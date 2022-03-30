# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras workflows."""

import collections
import datetime
import functools
import glob
import os
import random
import tempfile

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

from keras_declarative import hyperparams
from keras_declarative import config
from keras_declarative import io
from keras_declarative import objects
from keras_declarative import util


def train(config_file): # pylint: disable=missing-raises-doc
  """Create and train a new model.

  Args:
    config_file: A list of paths to the YAML configuration files.
  """
  # Get default config for this experiment.
  serialized_params = config.TrainWorkflowConfig()

  # Do overrides from from `config_file`.
  for file in config_file or []:
    serialized_params = hyperparams.override_params_dict(
        serialized_params, file, is_strict=False)

  # The following procedures do not support special objects.
  _set_global_config(serialized_params)
  exp_name, exp_dir = _setup_directory(serialized_params, config_file)
  ds_container = _make_datasets(serialized_params)

  # Deserialize special objects such as random number generators and tunable
  # hyperparameters.
  params = config.deserialize_special_objects(serialized_params)
  hp = config.find_hyperparameters(params)

  # Save external files.
  ext_objects = config.find_external_objects(params)
  external_filenames = []
  for obj in ext_objects:
    if obj.filename not in external_filenames:
      external_filenames.append(obj.filename)
  if external_filenames:
    ext_dir = os.path.join(exp_dir, 'external')
    tf.io.gfile.makedirs(ext_dir)
    for filename in external_filenames:
      tf.io.gfile.copy(str(filename), os.path.join(ext_dir, filename.name))

  if hp.space:
    # If a hyperparameter space is defined, launch tuning.
    hypermodel = HyperModel(params, exp_name, exp_dir, ds_container)
    tuner = _get_tuner(params, hypermodel, exp_dir)
    tuner.search(epochs=params.training.epochs,
                 callbacks=_get_callbacks(params, exp_dir, tuning=True))

  else:
    try:
      ds_container = _transform_datasets(params, ds_container, exp_name)
      model = _build_model(params, ds_container)
      model, _ = _train_model(params, exp_dir, model, ds_container)
      _test_model(params, model, ds_container, exp_dir)
      _clean_up(ds_container.cachefiles)

    except BaseException as err:
      _clean_up(ds_container.cachefiles)
      raise err

  print(f"Done. Results saved to {exp_dir}.")


def test(config_file): # pylint: disable=missing-raises-doc
  """Test an existing model.

  Args:
    config_file: A list of paths to the YAML configuration files.
  """
  # Get default config for this experiment.
  serialized_params = config.TestWorkflowConfig()

  # Do overrides from from `config_file`.
  for file in config_file or []:
    serialized_params = hyperparams.override_params_dict(
        serialized_params, file, is_strict=False)

  # Deserialize special objects such as random number generators and tunable
  # hyperparameters.
  params = config.deserialize_special_objects(serialized_params)

  try:
    _set_global_config(serialized_params)
    exp_name, exp_dir = _setup_directory(serialized_params, config_file)
    ds_container = _make_datasets(serialized_params)
    ds_container = _transform_datasets(params, ds_container, exp_name)
    model = _build_model(params, ds_container)
    _test_model(params, model, ds_container, exp_dir)
    _clean_up(ds_container.cachefiles)

  except BaseException as err:
    _clean_up(ds_container.cachefiles)
    raise err

  print(f"Done. Results saved to {exp_dir}.")


class HyperModel(kt.HyperModel):
  """Custom hypermodel for Keras Declarative."""
  def __init__(self, params, exp_name, exp_dir, ds_container, **kwargs):
    super().__init__(**kwargs)
    self.params = params
    self.exp_name = exp_name
    self.exp_dir = exp_dir
    self.ds_container = ds_container

  def build(self, hp):
    """Builds a model."""
    params = config.inject_hyperparameters(self.params, hp)
    self.transformed_ds_container = _transform_datasets(
        params, self.ds_container, self.exp_name)
    _clean_up(self.transformed_ds_container.cachefiles)
    return _build_model(params, self.transformed_ds_container)

  def fit(self, hp, model, *args, **kwargs): # pylint: disable=unused-argument
    """Trains a model."""
    _, history = _train_model(self.params, self.exp_dir, model,
                              self.transformed_ds_container,
                              *args, **kwargs)
    return history


def _set_global_config(params):
  """Set global configuration.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
  """
  if params.experiment.seed is not None:
    random.seed(params.experiment.seed)
    np.random.seed(params.experiment.seed)
    tf.random.set_seed(params.experiment.seed)


def _setup_directory(params, config_file):
  """Set up experiment directory.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    config_file: A list of paths to the YAML configuration files.

  Returns:
    The experiment name and the experiment directory.

  Raises:
    OSError: If the experiment directory already exists.
  """
  path = params.experiment.path or os.getcwd()

  if params.experiment.name:
    exp_name = params.experiment.name
  else:
    exp_name = os.path.splitext(os.path.basename(config_file[-1]))[0]
    exp_name += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  exp_dir = os.path.join(path, exp_name)
  if tf.io.gfile.exists(exp_dir):
    raise OSError(f"Directory {exp_dir} already exists.")
  tf.io.gfile.makedirs(exp_dir)
  hyperparams.save_params_dict_to_yaml(
      params, os.path.join(exp_dir, 'config.yaml'))

  return exp_name, exp_dir


def _make_datasets(params):
  """Makes datasets from the specified configuration.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.

  Returns:
    A `DatasetContainer`.

  Raises:
    ValueError: If `data.sources` is `None`.
  """
  if params.data.sources is None:
    raise ValueError("`data.sources` must be provided.")

  # Generate and concatenate datasets for each source.
  for index, source in enumerate(params.data.sources):
    if index == 0:
      ds_container = _make_datasets_from_source(source, params.data.specs)
    else:
      ds_container += _make_datasets_from_source(source, params.data.specs)

  # Set dataset options.
  if params.data.options is not None:
    options = tf.data.Options()
    if params.data.options.max_intra_op_parallelism is not None:
      options.threading.max_intra_op_parallelism = (
          params.data.options.max_intra_op_parallelism)
    if params.data.options.private_threadpool_size is not None:
      options.threading.private_threadpool_size = (
          params.data.options.private_threadpool_size)
    ds_container = ds_container.with_options(options)

  return ds_container


def _make_datasets_from_source(source, specs):
  """Make datasets from a generic source.

  Args:
    source: A `DataSourceConfig`.
    specs: A `DataSpecsConfig`.

  Returns:
    A `DatasetContainer`.

  Raises:
    ValueError: If passed an unsupported data source.
  """
  if source.type == 'dlex':
    return _make_dlex_datasets(source.dlex, specs)
  if source.type == 'tfds':
    return _make_tfds_datasets(source.tfds)
  raise ValueError(f"Unsupported source type: {source.type}")


def _make_dlex_datasets(source, specs):
  """Makes datasets from DLEX source files.

  Args:
    source: A `DataSourceConfig`.
    specs: A `DataSpecsConfig`.

  Returns:
    A `DatasetContainer`.
  """
  # Get filenames.
  files = io.get_distributed_hdf5_filenames(source.path, source.prefix)
  n = len(files)

  # Get splits.
  def _canonicalize_split(split):
    if util.is_percent(split):
      p = util.percent_to_float(split)
      return int(p * n)
    if not isinstance(split, int):
      raise ValueError(f"Invalid split: {split}")
    return split

  n_train = _canonicalize_split(source.split.train)
  n_val = _canonicalize_split(source.split.val)
  n_test = _canonicalize_split(source.split.test)

  # Split files.
  if source.split.mode == 'random':
    random.shuffle(files)
  train_files = files[0:n_train]
  val_files = files[n_train:n_train + n_val]
  test_files = files[n_train + n_val:n_train + n_val + n_test]

  # Shuffle files.
  random.shuffle(train_files)
  random.shuffle(val_files)
  random.shuffle(test_files)

  # Get specs.
  train_spec = _parse_spec_config(specs.train)
  val_spec = _parse_spec_config(specs.val)
  test_spec = _parse_spec_config(specs.test)

  # Prepend a slash to the spec names. This is needed for HDF5 dataset names.
  def _prepend_slashes(spec):
    return {f"/{k}": tf.TensorSpec.from_spec(v, name=f"{k}")
            for k, v in spec.items()}
  train_spec = _prepend_slashes(train_spec)
  val_spec = _prepend_slashes(val_spec)
  test_spec = _prepend_slashes(test_spec)

  # Create datasets.
  train_ds = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(list(map(str, train_files)), dtype=tf.string))
  val_ds = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(list(map(str, val_files)), dtype=tf.string))
  test_ds = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(list(map(str, test_files)), dtype=tf.string))

  # Add data read operation.
  def _read_hdf5(filename, spec=None):
    hdf5_io_tensor = tfio.IOTensor.from_hdf5(filename, spec=spec)
    tensors = {k: hdf5_io_tensor(k).to_tensor() for k in hdf5_io_tensor.keys}
    return {k: tf.ensure_shape(v, spec[k].shape) for k, v in tensors.items()}
  train_ds = train_ds.map(
      functools.partial(_read_hdf5, spec=train_spec))
  val_ds = val_ds.map(
      functools.partial(_read_hdf5, spec=val_spec))
  test_ds = test_ds.map(
      functools.partial(_read_hdf5, spec=test_spec))

  # Remove the slashes added previously.
  def _remove_slashes(structure):
    if isinstance(structure, dict):
      return {k[1:]: v for k, v in structure.items()}
    return structure
  train_ds = train_ds.map(_remove_slashes)
  val_ds = val_ds.map(_remove_slashes)
  test_ds = test_ds.map(_remove_slashes)

  # Get the dataset keys (obtained from filenames).
  train_keys = [f.stem for f in train_files]
  val_keys = [f.stem for f in val_files]
  test_keys = [f.stem for f in test_files]

  return util.DatasetContainer({
      'train': (train_keys, train_ds),
      'val': (val_keys, val_ds),
      'test': (test_keys, test_ds)
  })


def _make_tfds_datasets(source):
  """Makes datasets from a TFDS source.

  Args:
    source: A `DataSourceConfig`.

  Returns:
    A `DatasetContainer`.
  """
  name = source.name
  if source.version is not None:
    name = f"{name}:{source.version}"

  datasets, _ = tfds.load(
      name,
      split={k: v for k, v in source.split.as_dict().items() if v is not None},
      data_dir=source.data_dir,
      read_config=tfds.ReadConfig(try_autocache=False,
                                  add_tfds_id=True,
                                  interleave_cycle_length=1,
                                  interleave_block_length=1,
                                  skip_prefetch=True),
      with_info=True)

  # TODO(jmontalt): add support for example IDs.
  return util.DatasetContainer({k: (None, v) for k, v in datasets.items()})


def _transform_datasets(params, ds_container, exp_name):
  """Transforms the datasets.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    ds_container: A `DatasetContainer`.
    exp_name: A `str`. The experiment name.

  Returns:
    The transformed `DatasetContainer`.

  Raises:
    ValueError: If `data.sources` was not specified.
  """
  # Add user-specified transforms to each dataset.
  ds_container = _add_transforms(
      ds_container, 'train', params.data.transforms.train,
      params.data.options, exp_name)
  ds_container = _add_transforms(
      ds_container, 'val', params.data.transforms.val,
      params.data.options, exp_name)
  ds_container = _add_transforms(
      ds_container, 'test', params.data.transforms.test,
      params.data.options, exp_name)

  return ds_container


def _add_transforms(ds_container, ds_name, transforms, options, exp_name):
  """Add configured transforms to dataset.

  Args:
    ds_container: A `DatasetContainer`.
    ds_name: The name of the dataset currently being processed. One of
      `'train'`, `'val'` or `'test'`.
    transforms: A list of `DataTransformConfig`.
    options: A `DataOptionsConfig`.
    exp_name: A `str`. The experiment name.

  Returns:
    A `DatasetContainer`.

  Raises:
    ValueError: If a transform is not known or supported.
  """
  ds_container = ds_container.select(ds_name)

  for transform in transforms or []:
    if transform.type == 'apply':
      transformation_func = objects.get_layer(
          transform.apply.transformation_func)
      ds_container = ds_container.apply(transformation_func)

    elif transform.type == 'batch':
      ds_container = ds_container.batch(
          transform.batch.batch_size,
          drop_remainder=transform.batch.drop_remainder,
          num_parallel_calls=transform.batch.num_parallel_calls,
          deterministic=transform.batch.deterministic)

    elif transform.type == 'cache':
      cachefile = transform.cache.filename
      if cachefile is None:  # Auto-naming.
        cachefile = os.path.join(tempfile.gettempdir(),
                                 f'tfdata_cache-{exp_name}')
      # Delete the cache file if it exists.
      for file in glob.glob(cachefile + '*'):
        os.remove(file)
      ds_container = ds_container.cache(filename=cachefile)

    elif transform.type == 'filter':
      predicate = objects.get_predicate(transform.filter.predicate)
      ds_container = ds_container.filter(predicate)

    elif transform.type == 'flat_map':
      map_func = objects.get_layer(transform.flat_map.map_func)
      ds_container = ds_container.flat_map(map_func)

    elif transform.type == 'map':
      map_func = objects.get_layer(transform.map.map_func)
      ds_container = ds_container.map(
          _maybe_decorate_map_func(map_func, transform.map.component,
                                   transform.map.output),
          num_parallel_calls=transform.map.num_parallel_calls,
          deterministic=transform.map.deterministic)

    elif transform.type == 'prefetch':
      ds_container = ds_container.prefetch(transform.prefetch.buffer_size)

    elif transform.type == 'repeat':
      ds_container = ds_container.repeat(transform.repeat.count)

    elif transform.type == 'shuffle':
      if ds_name != 'train' and options.shuffle_training_only:
        continue
      ds_container = ds_container.shuffle(
          transform.shuffle.buffer_size,
          seed=transform.shuffle.seed,
          reshuffle_each_iteration=transform.shuffle.reshuffle_each_iteration)

    else:
      raise ValueError(f"Unknown transform type: {transform.type}")

  ds_container = ds_container.unselect()
  return ds_container


def _build_model(params, ds_container):
  """Train a model.

  Args:
    params: A `TrainWorkflowConfig`.
    ds_container: A `DatasetContainer`.

  Returns:
    A trained `tf.keras.Model`.
  """
  def _build_model_internal():

    if params.model.path is not None:
      # Load existing model.
      model = tf.keras.models.load_model(params.model.path)

    else: # Create new model architecture.
      # Currently we support only layers as network.
      layer = objects.get_layer(params.model.network)

      if params.model.input_spec:
        # User specified input spec explicitly, so use that.
        input_spec = _parse_spec_config(params.model.input_spec)

      else:
        # Model input spec not specified explicitly. Infer from training dataset.
        input_spec, _, _ = tf.keras.utils.unpack_x_y_sample_weight(
            ds_container.train_ds.element_spec)

      # Network.
      model = util.model_from_layers(layer, input_spec)

      optimizer = objects.get_optimizer(params.training.optimizer)
      loss = objects.get_nest(objects.get_loss)(params.training.loss)
      metrics = objects.get_nest(objects.get_metric)(params.training.metrics)

      model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=metrics or None,
                    loss_weights=params.training.loss_weights or None,
                    weighted_metrics=params.training.weighted_metrics or None,
                    run_eagerly=params.training.run_eagerly,
                    steps_per_execution=params.training.steps_per_execution)

    # Load weights, if they were specified.
    if params.model.weights is not None:
      model.load_weights(params.model.weights)

    # Print model summary.
    model.summary(line_length=80)

    return model

  strategy = objects.get_strategy(params.distribute.strategy)

  if strategy is not None:
    with strategy.scope():
      return _build_model_internal()

  return _build_model_internal()


def _train_model(params, exp_dir, model, ds_container, **kwargs):
  """Trains a Keras model.

  Args:
    params: A `TrainWorkflowConfig`.
    exp_dir: A `str`. The experiment directory.
    model: A `tf.keras.Model`.
    ds_container: A `DatasetContainer`.
    **kwargs: Keyword arguments to be passed to `fit`. Can be used by a tuner to
      override `fit` parameters.

  Returns:
    A trained `tf.keras.Model`.
  """
  # Get callbacks.
  callbacks = kwargs.get('callbacks') or _get_callbacks(
      params, exp_dir, ds_container)

  # Patch TensorBoardImages dataset.
  for callback in callbacks:
    if callback.__class__.__name__.startswith("TensorBoardImages"):
      dataset_name = getattr(callback, 'dataset_name', 'val')
      callback.x = ds_container.datasets[dataset_name]

  kwargs['x'] = ds_container.train_ds
  if 'epochs' not in kwargs:
    kwargs['epochs'] = params.training.epochs
  if 'verbose' not in kwargs:
    kwargs['verbose'] = params.training.verbose
  kwargs['callbacks'] = callbacks
  kwargs['validation_data'] = ds_container.val_ds

  print("Training...")
  history = model.fit(**kwargs)
  print("Training complete.")

  return model, history


def _test_model(params, model, ds_container, exp_dir):
  """Tests a model.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    model: A `tf.keras.Model`.
    ds_container: A `DatasetContainer`.
    exp_dir: A `str`. The experiment directory.
  """
  print("Predicting...")

  if isinstance(params.predict.datasets, str):
    dataset_keys = [params.predict.datasets]
  else:
    dataset_keys = params.predict.datasets

  datasets = {k: ds_container.datasets[k] for k in dataset_keys}

  pred_path = _get_pred_path(params, exp_dir)
  tf.io.gfile.makedirs(pred_path)

  if params.predict.save_images:
    image_path = _get_image_path(params, exp_dir)
    tf.io.gfile.makedirs(image_path)

  if params.predict.evaluate:
    eval_path = _get_eval_path(params, exp_dir)
    tf.io.gfile.makedirs(eval_path)

  feature_names = ['features/' + name for name in model.input_names]
  label_names = ['labels/' + name for name in model.output_names]
  pred_names = ['predictions/' + name for name in model.output_names]

  for ds_name, dataset in datasets.items():

    ds_pred_path = os.path.join(pred_path, ds_name)
    tf.io.gfile.makedirs(ds_pred_path)

    if params.predict.save_images:
      ds_image_path = os.path.join(image_path, ds_name)
      tf.io.gfile.makedirs(ds_image_path)

    if params.predict.evaluate:
      columns = ['name'] + model.metrics_names
      df = pd.DataFrame(columns=columns)

    # Get number of elements.
    n = dataset.cardinality().numpy()
    if n < 0:  # -1 means infinite, -2 means unknown
      n = None

    progbar = tf.keras.utils.Progbar(n)

    for element in dataset:
      x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(element)
      y_pred = model(x, training=False)

      x = _unstack_nested_tensors(x)
      y = _unstack_nested_tensors(y)
      y_pred = _unstack_nested_tensors(y_pred)

      # For each element in batch.
      for x_elem, y_elem, y_pred_elem in zip(x, y, y_pred):
        data = {}
        def _add_to_data(data, values, names, prefix):
          if isinstance(values, (tuple, list)):
            for elem_index, elem in enumerate(values):
              data[names[elem_index]] = elem
          elif isinstance(values, dict):
            for key, value in values.items():
              data[prefix + '/' + key] = value
          else:
            data[names[0]] = values
          return data
        data = _add_to_data(data, x_elem, feature_names, 'features')
        data = _add_to_data(data, y_elem, label_names, 'labels')
        data = _add_to_data(data, y_pred_elem, pred_names, 'predictions')

        basename = os.path.splitext(
            os.path.basename(ds_container.example_ids[ds_name].pop(0)))[0]
        file_path = os.path.join(ds_pred_path, basename + '.h5')
        io.write_hdf5(file_path, data)

        if params.predict.save_images:
          tfmri = _get_tensorflow_mri()
          image = _make_image(data)
          file_path = os.path.join(ds_image_path, basename + '.gif')
          tf.io.write_file(file_path, tfmri.io.encode_gif(image))

        if params.predict.evaluate:
          # Add batch dimensions.
          features = {k.replace('features/', ''): tf.expand_dims(v, 0)
                      for k, v in data.items() if k.startswith('features/')}
          labels = {k.replace('labels/', ''): tf.expand_dims(v, 0)
                    for k, v in data.items() if k.startswith('labels/')}
          results = model.evaluate(x=features, y=labels, verbose=0)
          df = df.append(dict(zip(columns, [basename] + results)),
                         ignore_index=True)

      progbar.add(1)

    if params.predict.evaluate:
      df.sort_values(by='name', inplace=True, ignore_index=True)
      df.to_csv(os.path.join(eval_path, ds_name + '.csv'))

  print("Prediction complete.")


def _make_image(data):
  """Makes a generic display image from an example.

  Concatenates all the images in `data` horizontally and returns a single image.

  Args:
    data: A `dict` containing images as values.

  Returns:
    An image suitable for display.
  """
  normalize = lambda x: tf.math.divide(
      x - tf.reduce_min(x), tf.reduce_max(x) - tf.reduce_min(x))
  images = list(data.values())
  image = tf.concat(images, axis=-2) # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
  image = normalize(image)
  image *= 255.0
  image = tf.cast(image, tf.uint8)
  return image


def _clean_up(cachefiles):
  """Clean up.

  Args:
    cachefiles: A list of cachefiles to be deleted.
  """
  for cachefile in cachefiles:
    for file in glob.glob(cachefile + '*'):
      os.remove(file)


def _parse_spec_config(spec_config):
  """Parse spec configuration.

  Converts a list of `TensorSpecConfig` to a nested structure of
  `tf.TensorSpec`.

  Args:
    spec_config: A list of `TensorSpecConfig`.

  Returns:
    A nested structure of `tf.TensorSpec`.

  Raises:
    ValueError: If `spec_config` is invalid.
  """
  if isinstance(spec_config, config.TensorSpecConfig):
    return tf.TensorSpec(spec_config.shape, spec_config.dtype, spec_config.name)

  names = [spec.name for spec in spec_config]
  if not (
      all(name is not None for name in names) or
      all(name is None for name in names)):
    raise ValueError(
        "Either all tensor specifications must have names or none must have "
        "names.")

  if spec_config[0].name is not None:
    return {spec.name: tf.TensorSpec(
        spec.shape, spec.dtype, spec.name) for spec in spec_config}

  return [tf.TensorSpec(spec.shape, spec.dtype) for spec in spec_config]


def _maybe_decorate_map_func(map_func, component=None, output=None):
  """Decorates a mapping function.

  Args:
    map_func: A callable.
    component: An `int` or `str`. The component `map_func` should be applied to.
      If `None` the function is applied to its unmodified input.
    output: An `int` or `str`. The destination where the output of `map_func`
      should be inserted.

  Returns:
    A (possibly decorated) function.
  """
  if component is None:
    return map_func

  if isinstance(component, (int, str)):
    component = [component]

  if output is None:
    # No output was provided, apply the function to each component and return
    # the result in the same location.
    def _map_func(*args):
      args = args[0] if len(args) == 1 else list(args)
      for c in component:
        args[c] = map_func(args[c])
      return tuple(args) if isinstance(args, list) else args
  else:
    # An output was specified, apply the function to the specified component and
    # return the result at the specified destination.
    def _map_func(args):
      args[output] = map_func([args[c] for c in component])
      return args

  return _map_func


def _get_callbacks(params, exp_dir, ds_container=None, tuning=False):
  """Gets the callbacks.

  Creates default callbacks and user callbacks.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    exp_dir: A `str`. The experiment directory.
    ds_container: (optional) A `DatasetContainer`.
    tuning: (optional) A `bool`. Whether this is being called by a tuner.

  Returns:
    A list of `tf.keras.callbacks.Callback`.
  """
  # Complete callback configs.
  callback_configs = params.training.callbacks

  checkpoint_kwargs = None
  tensorboard_kwargs = None
  images_class_name = None # Support subclasses.
  images_kwargs = None
  remaining_callback_configs = []

  for value in callback_configs:
    if isinstance(value, str) and value == 'ModelCheckpoint':
      checkpoint_kwargs = {}
      continue
    if (isinstance(value, config.ObjectConfig) and
        value.class_name == 'ModelCheckpoint'):
      checkpoint_kwargs = value.config.as_dict()
      continue

    if isinstance(value, str) and value == 'TensorBoard':
      tensorboard_kwargs = {}
      continue
    if (isinstance(value, config.ObjectConfig) and
        value.class_name == 'TensorBoard'):
      tensorboard_kwargs = value.config.as_dict()
      continue

    if isinstance(value, str) and value.startswith('TensorBoardImages'):
      images_class_name = value
      images_kwargs = {}
      continue
    if (isinstance(value, config.ObjectConfig) and
        value.class_name.startswith('TensorBoardImages')):
      images_class_name = value.class_name
      images_kwargs = value.config.as_dict()
      continue

    remaining_callback_configs.append(value)

  checkpoint_callback = _get_checkpoint_callback(params, exp_dir,
                                                 kwargs=checkpoint_kwargs)
  tensorboard_callback = _get_tensorboard_callback(params, exp_dir,
                                                   kwargs=tensorboard_kwargs)
  images_callback = _get_images_callback(params, exp_dir, ds_container,
                                         images_class_name,
                                         kwargs=images_kwargs)

  # Parse callback configs.
  callbacks = objects.get_list(objects.get_callback)(remaining_callback_configs)

  if checkpoint_callback and not tuning:
    # We do not add the checkpoint callback if tuning, as it will be added by
    # the tuner.
    callbacks.append(checkpoint_callback)
  if tensorboard_callback:
    callbacks.append(tensorboard_callback)
  if images_callback:
    callbacks.append(images_callback)

  return callbacks


def _unstack_nested_tensors(structure):
  """Make list of unstacked nested tensors.

  Args:
    structure: Nested structure of tensors whose first dimension is to be
      unstacked.

  Returns:
    A list of the unstacked nested tensors.
  """
  if structure is None:
    return structure

  flat_sequence = tf.nest.flatten(structure)
  unstacked_flat_sequence = [tf.unstack(tensor) for tensor in flat_sequence]

  return [
      tf.nest.pack_sequence_as(structure, sequence)
      for sequence in zip(*unstacked_flat_sequence)
  ]


def _get_checkpoint_callback(params, exp_dir, kwargs=None):
  """Gets the model checkpoint callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    exp_dir: A `str`. The experiment directory.
    kwargs: (optional) A `dict`. Keyword arguments for the callback.

  Returns:
    A `tf.keras.callbacks.ModelCheckpoint` instance, or `None`.
  """
  if not params.training.use_default_callbacks and kwargs is None:
    return None

  default_kwargs = dict(
      filepath=None,
      monitor='val_loss',
      verbose=1,
      save_best_only=True,
      save_weights_only=False,
      mode='auto',
      save_freq='epoch')

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  if kwargs['filepath'] is None:
    kwargs['filepath'] = _get_ckpt_path(exp_dir, kwargs)

  return tf.keras.callbacks.ModelCheckpoint(**kwargs)


def _get_tensorboard_callback(params, exp_dir, kwargs=None):
  """Gets the TensorBoard callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    exp_dir: A `str`. The experiment directory.
    kwargs: (optional) A `dict`. Keyword arguments for the callback.

  Returns:
    A `tf.keras.callbacks.TensorBoard` instance, or `None`.
  """
  if not params.training.use_default_callbacks and kwargs is None:
    return None

  default_kwargs = dict(
      log_dir=_get_logs_path(params, exp_dir),
      histogram_freq=0,
      write_graph=True,
      write_images=False,
      update_freq='epoch',
      profile_batch=0
  )

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  return tf.keras.callbacks.TensorBoard(**kwargs)


def _get_images_callback(params, exp_dir, ds_container,
                         class_name, kwargs=None):
  """Gets the images callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    exp_dir: A `str`. The experiment directory.
    ds_container: A `DatasetContainer`.
    class_name: A `str`. The class name of the images callback.
    kwargs: (optional) A `dict`. Keyword arguments for the callback.

  Returns:
    A `tfmri.callbacks.TensorBoardImages` instance, or `None`.

  Raises:
    ValueError: If `kwargs` is not `None` and TensorFlow MRI cannot be imported.
  """
  if kwargs is None:
    return None

  dataset_name = kwargs.pop('x', 'val')
  default_kwargs = dict(
      x=(ds_container.datasets[dataset_name]
         if ds_container is not None else None),
      log_dir=_get_logs_path(params, exp_dir)
  )

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  tfmri = _get_tensorflow_mri(
      message="TensorFlow MRI is needed to use the TensorBoardImages callback.",
      missing_ok=False)
  callback = getattr(tfmri.callbacks, class_name)(**kwargs)
  # Store user dataset selection to use when patching the callback during
  # hyperparameter tuning.
  callback.dataset_name = dataset_name
  return callback


def _get_ckpt_path(exp_dir, checkpoint_kwargs):  # pylint: disable=missing-param-doc,missing-any-param-doc
  """Gets path to checkpoints."""
  # Get relevant checkpoint configuration.
  monitor = checkpoint_kwargs['monitor']
  save_weights_only = checkpoint_kwargs['save_weights_only']

  # The checkpoint directory. Create if necessary.
  path = os.path.join(exp_dir, 'ckpt')
  tf.io.gfile.makedirs(path)

  # The checkpoint filename.
  filename = 'weights' if save_weights_only else 'model'
  filename += '.{epoch:04d}-{' + monitor + ':.4f}.h5'

  return os.path.join(path, filename)


def _get_logs_path(params, exp_dir): # pylint: disable=unused-argument
  """Gets path to TensorBoard logs."""
  return os.path.join(exp_dir, 'logs')


def _get_pred_path(params, exp_dir): # pylint: disable=unused-argument
  """Gets path to predictions."""
  return os.path.join(exp_dir, 'pred')


def _get_image_path(params, exp_dir): # pylint: disable=unused-argument
  """Gets path to images."""
  return os.path.join(exp_dir, 'image')


def _get_eval_path(params, exp_dir): # pylint: disable=unused-argument
  """Gets path to evaluation results."""
  return os.path.join(exp_dir, 'eval')


def _get_tuner(params, hypermodel, exp_dir): # pylint: disable=missing-function-docstring

  if params.tuning.tuner is None:
    return None

  name, kwargs = objects.class_and_config_for_serialized_object(
      params.tuning.tuner)

  tuners = {
      'RandomSearch': util.RandomSearch,
      'BayesianOptimization': util.BayesianOptimization,
      'Hyperband': util.Hyperband
  }

  if 'objective' in kwargs and isinstance(kwargs['objective'], dict):
    kwargs['objective'] = kt.Objective(**kwargs['objective'])

  kwargs['hypermodel'] = hypermodel
  kwargs['directory'] = exp_dir
  kwargs['project_name'] = 'tuning'

  return tuners[name](**kwargs)


def _get_tensorflow_mri(message=None, missing_ok=False):
  """Gets the TensorFlow MRI module.

  Args:
    message: An optional `str`. An error message to display if the module is not
      found.
    missing_ok: An optional `bool`. If `True`, do not raise an exception if the
      module is not found. In this case, returns `None`.

  Returns:
    The TensorFlow MRI module, or `None` if it is not installed and `missing_ok`
    is `True`.

  Raises:
    ValueError: If the module is not installed and `missing_ok` is `False`.
  """
  try:
    import tensorflow_mri # pylint: disable=import-outside-toplevel
    return tensorflow_mri
  except ImportError as err:
    if missing_ok:
      return None
    if message is None:
      message = (
          "TensorFlow MRI is required for an action you requested. Please "
          "install TensorFlow MRI to continue.")
    raise ValueError(message) from err


class defaultdict(collections.defaultdict): # pylint: disable=invalid-name

  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    ret = self[key] = self.default_factory(key) # pylint: disable=not-callable
    return ret
