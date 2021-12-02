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

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
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
  expname, expdir = _setup_directory(serialized_params, config_file)
  sources = _get_data_sources(serialized_params)

  # Deserialize special objects such as random number generators and tunable
  # hyperparameters.
  params = config.deserialize_special_objects(serialized_params)
  hp = config.find_hyperparameters(params)

  if hp.space:
    # If a hyperparameter space is defined, launch tuning.
    hypermodel = HyperModel(params, expname, expdir, sources)
    tuner = _get_tuner(params, hypermodel, expdir)
    tuner.search(epochs=params.training.epochs,
                 callbacks=_get_callbacks(params, expdir, tuning=True))

  else:
    try:
      datasets, cachefiles = _build_datasets(params, expname, sources)
      model = _build_model(params, datasets)
      model, _ = _train_model(params, expdir, model, datasets)
      _test_model(params, model, datasets, sources, expdir)
      _clean_up(cachefiles)

    except BaseException as err:
      _clean_up(cachefiles)
      raise err

  print(f"Done. Results saved to {expdir}.")


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
    expname, expdir = _setup_directory(serialized_params, config_file)
    sources = _get_data_sources(params)
    datasets, cachefiles = _build_datasets(params, expname, sources)
    model = _build_model(params, datasets)
    _test_model(params, model, datasets, sources, expdir)
    _clean_up(cachefiles)

  except BaseException as err:
    _clean_up(cachefiles)
    raise err

  print(f"Done. Results saved to {expdir}.")


class HyperModel(kt.HyperModel):
  """Custom hypermodel for Keras Declarative."""
  def __init__(self, params, expname, expdir, sources, **kwargs):
    super().__init__(**kwargs)
    self.params = params
    self.expname = expname
    self.expdir = expdir
    self.sources = sources
    self.datasets = None

  def build(self, hp):
    """Builds a model."""
    params = config.inject_hyperparameters(self.params, hp)
    self.datasets, cachefiles = _build_datasets(
        params, self.expname, self.sources)
    _clean_up(cachefiles)
    return _build_model(params, self.datasets)

  def fit(self, hp, model, *args, **kwargs): # pylint: disable=unused-argument
    """Trains a model."""
    _, history = _train_model(self.params, self.expdir, model,
                              self.datasets, *args, **kwargs)
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
  """
  path = params.experiment.path or os.getcwd()

  if params.experiment.name:
    expname = params.experiment.name
  else:
    expname = os.path.splitext(os.path.basename(config_file[-1]))[0]
    expname += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

  expdir = os.path.join(path, expname)
  tf.io.gfile.makedirs(expdir)
  hyperparams.save_params_dict_to_yaml(
      params, os.path.join(expdir, 'config.yaml'))

  return expname, expdir


def _get_data_sources(params):
  """Get training, validation and test HDF5 files.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.

  Returns:
    A tuple of three lists of HDF5 files.

  Raises:
    ValueError: If `data.sources` is `None`.
  """
  if params.data.sources is None:
    raise ValueError("`data.sources` must be provided.")

  train_sources = []
  val_sources = []
  test_sources = []

  for source in params.data.sources:
    files = io.get_distributed_hdf5_filenames(source.path, source.prefix)
    n = len(files)

    if source.split.mode == 'random':
      random.shuffle(files)

    get_n = lambda s, n=n: s if isinstance(s, int) else int(s * n)
    n_train = get_n(source.split.train)
    n_val = get_n(source.split.val)
    n_test = get_n(source.split.test)

    train_sources.extend(files[0:n_train])
    val_sources.extend(files[n_train:n_train + n_val])
    test_sources.extend(files[n_train + n_val:n_train + n_val + n_test])

  # Shuffle all the data.
  random.shuffle(train_sources)
  random.shuffle(val_sources)
  random.shuffle(test_sources)

  return train_sources, val_sources, test_sources


def _build_datasets(params, expname, sources):
  """Set up datasets.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    expname: A `str`. The experiment name.
    sources: A tuple of three lists. The training files, validation files, and
      test files.

  Returns:
    A tuple of three datasets (train, validation, test) and a list of cache
    files.

  Raises:
    ValueError: If `data.sources` was not specified.
  """
  train_sources, val_sources, test_sources = sources

  # Get specs.
  train_spec = _parse_spec_config(params.data.train_spec)
  val_spec = _parse_spec_config(params.data.val_spec)
  test_spec = _parse_spec_config(params.data.test_spec)

  # Some functions.
  def _read_hdf5(filename, spec=None):
    hdf5_io_tensor = tfio.IOTensor.from_hdf5(filename, spec=spec)
    tensors = {k: hdf5_io_tensor(k).to_tensor() for k in hdf5_io_tensor.keys}
    return {k: tf.ensure_shape(v, spec[k].shape) for k, v in tensors.items()}

  def _get_outputs(x):
    out = list(x.values())
    return out[0] if len(out) == 1 else out

  # Create datasets.
  train_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(train_sources, dtype=tf.string))
  val_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(val_sources, dtype=tf.string))
  test_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(test_sources, dtype=tf.string))

  # Add data read.
  train_dataset = train_dataset.map(
      functools.partial(_read_hdf5, spec=train_spec))
  val_dataset = val_dataset.map(
      functools.partial(_read_hdf5, spec=val_spec))
  test_dataset = test_dataset.map(
      functools.partial(_read_hdf5, spec=test_spec))

  # Extract outputs.
  train_dataset = train_dataset.map(_get_outputs)
  val_dataset = val_dataset.map(_get_outputs)
  test_dataset = test_dataset.map(_get_outputs)

  # Add user-specified transforms to each dataset.
  cachefiles = []
  train_dataset, cachefiles = _add_transforms(
      train_dataset, params.data.train_transforms, params.data.options,
      expname, cachefiles, 'train')
  val_dataset, cachefiles = _add_transforms(
      val_dataset, params.data.val_transforms, params.data.options,
      expname, cachefiles, 'val')
  test_dataset, cachefiles = _add_transforms(
      test_dataset, params.data.test_transforms, params.data.options,
      expname, cachefiles, 'test')

  # Set the dataset options.
  train_dataset = _set_dataset_options(train_dataset, params.data.options)
  val_dataset = _set_dataset_options(val_dataset, params.data.options)
  test_dataset = _set_dataset_options(test_dataset, params.data.options)

  # Pack and return.
  datasets = train_dataset, val_dataset, test_dataset

  return datasets, cachefiles


def _add_transforms(dataset, transforms, options, expname, cachefiles, dstype):
  """Add configured transforms to dataset.

  Args:
    dataset: A `tf.data.Dataset`.
    transforms: A list of `DataTransformConfig`.
    options: A `DataOptionsConfig`.
    expname: A `str`. The experiment name.
    cachefiles: A list of cache files.
    dstype: The type of this dataset. One of `'train'`, `'val'` or `'test'`.

  Returns:
    A `tf.data.Dataset` and a list of cache files.

  Raises:
    ValueError: If a transform is not known or supported.
  """
  for transform in transforms or []:
    if transform.type == 'batch':
      dataset = dataset.batch(
          transform.batch.batch_size,
          drop_remainder=transform.batch.drop_remainder,
          num_parallel_calls=transform.batch.num_parallel_calls,
          deterministic=transform.batch.deterministic)

    elif transform.type == 'cache':
      # Delete the cache file if it exists.
      cachefile = f'{transform.cache.filename}.{expname}.{dstype}'
      cachefiles.append(cachefile)
      for file in glob.glob(cachefile + '*'):
        os.remove(file)
      dataset = dataset.cache(filename=cachefile)

    elif transform.type == 'filter':
      predicate = objects.get_predicate(transform.filter.predicate)
      dataset = dataset.filter(predicate)

    elif transform.type == 'map':
      map_func = objects.get_layer(transform.map.map_func)
      dataset = dataset.map(
          _maybe_decorate_map_func(map_func, transform.map.component),
          num_parallel_calls=transform.map.num_parallel_calls,
          deterministic=transform.map.deterministic)

    elif transform.type == 'prefetch':
      dataset = dataset.prefetch(transform.prefetch.buffer_size)

    elif transform.type == 'shuffle':
      if dstype != 'train' and options.shuffle_training_only:
        continue
      dataset = dataset.shuffle(
          transform.shuffle.buffer_size,
          seed=transform.shuffle.seed,
          reshuffle_each_iteration=transform.shuffle.reshuffle_each_iteration)

    else:
      raise ValueError(f"Unknown transform type: {transform.type}")

  return dataset, cachefiles


def _set_dataset_options(dataset, options_config):
  """Set dataset options.

  Args:
    dataset: A `tf.data.Dataset`.
    options_config: A `DataOptionsConfig`.

  Returns:
    A `tf.data.Dataset` with the specified options set.
  """
  if options_config is None:
    return dataset

  options = tf.data.Options()

  if options_config.max_intra_op_parallelism is not None:
    options.threading.max_intra_op_parallelism = (
        options_config.max_intra_op_parallelism)
  if options_config.private_threadpool_size is not None:
    options.threading.private_threadpool_size = (
        options_config.private_threadpool_size)

  return dataset.with_options(options)


def _build_model(params, datasets):
  """Train a model.

  Args:
    params: A `TrainWorkflowConfig`.
    datasets: A tuple of three `tf.data.Dataset` (train, val, test).

  Returns:
    A trained `tf.keras.Model`.
  """
  train_dataset, _, _ = datasets

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
            train_dataset.element_spec)

      # Network.
      model = util.model_from_layers(layer, input_spec)

      optimizer = objects.get_optimizer(params.training.optimizer)
      loss = objects.get_list(objects.get_loss)(params.training.loss)
      metrics = objects.get_list(objects.get_metric)(params.training.metrics)

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


def _train_model(params, expdir, model, datasets, **kwargs):
  """Trains a Keras model.

  Args:
    params: A `TrainWorkflowConfig`.
    expdir: A `str`. The experiment directory.
    model: A `tf.keras.Model`.
    datasets: A tuple of three `tf.data.Dataset` (train, val, test).
    **kwargs: Keyword arguments to be passed to `fit`. Can be used by a tuner to
      override `fit` parameters.

  Returns:
    A trained `tf.keras.Model`.
  """
  train_dataset, val_dataset, _ = datasets

  # Get callbacks.
  callbacks = kwargs.get('callbacks') or _get_callbacks(
      params, expdir, datasets)

  # Patch TensorBoardImages dataset.
  for callback in callbacks:
    if callback.__class__.__name__ == "TensorBoardImages":
      callback.x = val_dataset

  kwargs['x'] = train_dataset
  if 'epochs' not in kwargs:
    kwargs['epochs'] = params.training.epochs
  if 'verbose' not in kwargs:
    kwargs['verbose'] = params.training.verbose
  kwargs['callbacks'] = callbacks
  kwargs['validation_data'] = val_dataset

  print("Training...")
  history = model.fit(**kwargs)
  print("Training complete.")

  return model, history


def _test_model(params, model, datasets, sources, expdir):
  """Tests a model.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    model: A `tf.keras.Model`.
    datasets: A tuple of three `tf.data.Dataset` (train, val, test).
    sources: A tuple of three lists of sources (train, val, test).
    expdir: A `str`. The experiment directory.
  """
  print("Predicting...")
  train_dataset, val_dataset, test_dataset = datasets
  train_sources, val_sources, test_sources = sources

  datasets = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset
  }

  sources = {
      'train': train_sources,
      'val': val_sources,
      'test': test_sources
  }

  if isinstance(params.predict.datasets, str):
    dataset_keys = [params.predict.datasets]
  else:
    dataset_keys = params.predict.datasets

  datasets = {k: datasets[k] for k in dataset_keys}

  pred_path = _get_pred_path(params, expdir)
  tf.io.gfile.makedirs(pred_path)

  if params.predict.save_images:
    image_path = _get_image_path(params, expdir)
    tf.io.gfile.makedirs(image_path)

  if params.predict.evaluate:
    eval_path = _get_eval_path(params, expdir)
    tf.io.gfile.makedirs(eval_path)

  input_names = defaultdict(lambda key: 'input_' + str(key))
  label_names = defaultdict(lambda key: 'label_' + str(key))
  pred_names = defaultdict(lambda key: 'pred_' + str(key))

  # TODO: add possibility of using user-specified names.

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
    if n < 0: # -1 means infinite, -2 means unknown
      n = None

    progbar = tf.keras.utils.Progbar(n)

    for element in dataset:
      x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(element)
      y_pred = model(x, training=False)

      x = _flatten_and_unbatch_nested_tensors(x)
      y = _flatten_and_unbatch_nested_tensors(y)
      y_pred = _flatten_and_unbatch_nested_tensors(y_pred)

      # For each element in batch.
      batch_size = len(x[0])
      for batch_index in range(batch_size):
        data = {}
        for elem_index, elem in enumerate(x):
          data[input_names[elem_index]] = elem[batch_index]
        for elem_index, elem in enumerate(y):
          data[label_names[elem_index]] = elem[batch_index]
        for elem_index, elem in enumerate(y_pred):
          data[pred_names[elem_index]] = elem[batch_index]

        basename = os.path.splitext(
            os.path.basename(sources[ds_name].pop(0)))[0]
        file_path = os.path.join(ds_pred_path, basename + '.h5')
        io.write_hdf5(file_path, data)

        if params.predict.save_images:
          tfmr = _get_tensorflow_mri()
          image = _make_image(data)
          file_path = os.path.join(ds_image_path, basename + '.gif')
          tf.io.write_file(file_path, tfmr.io.encode_gif(image))

        if params.predict.evaluate:
          inputs = [tf.expand_dims(data[v], 0) for v in input_names.values()]
          labels = [tf.expand_dims(data[v], 0) for v in label_names.values()]
          results = model.evaluate(x=inputs, y=labels, verbose=0)
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


def _maybe_decorate_map_func(map_func, component=None):
  """Decorates a mapping function.

  Args:
    map_func: A callable.
    component: An `int` or `str`. The component `map_func` should be applied to.
      If `None` the function is applied to its unmodified input.

  Returns:
    A (possibly decorated) function.
  """
  if component is None:
    return map_func

  if isinstance(component, (int, str)):
    component = [component]

  def _map_func(*args):
    args = list(args)
    for c in component:
      args[c] = map_func(args[c])
    return tuple(args)

  return _map_func


def _get_callbacks(params, expdir, datasets=None, tuning=False):
  """Gets the callbacks.

  Creates default callbacks and user callbacks.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    expdir: A `str`. The experiment directory.
    datasets: (optional) A tuple of three `tf.data.Dataset` (train, val, test).
    tuning: (optional) A `bool`. Whether this is being called by a tuner.

  Returns:
    A list of `tf.keras.callbacks.Callback`.
  """
  val_dataset = datasets[1] if datasets is not None else None

  # Complete callback configs.
  callback_configs = params.training.callbacks

  checkpoint_kwargs = None
  tensorboard_kwargs = None
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

    if isinstance(value, str) and value == 'TensorBoardImages':
      images_kwargs = {}
      continue
    if (isinstance(value, config.ObjectConfig) and
        value.class_name == 'TensorBoardImages'):
      images_kwargs = value.config.as_dict()
      continue

    remaining_callback_configs.append(value)

  checkpoint_callback = _get_checkpoint_callback(params, expdir,
                                                 kwargs=checkpoint_kwargs)
  tensorboard_callback = _get_tensorboard_callback(params, expdir,
                                                   kwargs=tensorboard_kwargs)
  images_callback = _get_images_callback(params, expdir, val_dataset,
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


def _flatten_and_unbatch_nested_tensors(structure):
  """Flattens and unbatches a nest of tensors.

  The output is a list of lists. The outer list corresponds to the number of
  elements in the input structure. The inner lists contain the unbatched
  tensors.
  """
  if structure is None:
    return structure
  return [tf.unstack(x) for x in tf.nest.flatten(structure)]


def _get_checkpoint_callback(params, expdir, kwargs=None):
  """Gets the model checkpoint callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    expdir: A `str`. The experiment directory.
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
    kwargs['filepath'] = _get_ckpt_path(expdir, kwargs)

  return tf.keras.callbacks.ModelCheckpoint(**kwargs)


def _get_tensorboard_callback(params, expdir, kwargs=None):
  """Gets the TensorBoard callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    expdir: A `str`. The experiment directory.
    kwargs: (optional) A `dict`. Keyword arguments for the callback.

  Returns:
    A `tf.keras.callbacks.TensorBoard` instance, or `None`.
  """
  if not params.training.use_default_callbacks and kwargs is None:
    return None

  default_kwargs = dict(
      log_dir=_get_logs_path(params, expdir),
      histogram_freq=0,
      write_graph=True,
      write_images=False,
      update_freq='epoch',
      profile_batch=0
  )

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  return tf.keras.callbacks.TensorBoard(**kwargs)


def _get_images_callback(params, expdir, dataset, kwargs=None):
  """Gets the images callback.

  Args:
    params: A `TrainWorkflowConfig` or `TestWorkflowConfig`.
    expdir: A `str`. The experiment directory.
    dataset: A `tf.data.Dataset`. The validation dataset.
    kwargs: (optional) A `dict`. Keyword arguments for the callback.

  Returns:
    A `tfmr.callbacks.TensorBoardImages` instance, or `None`.

  Raises:
    ValueError: If `kwargs` is not `None` and TensorFlow MRI cannot be imported.
  """
  if kwargs is None:
    return None

  default_kwargs = dict(
      x=dataset,
      log_dir=_get_logs_path(params, expdir)
  )

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  tfmr = _get_tensorflow_mri(
      message="TensorFlow MRI is needed to use the TensorBoardImages callback.",
      missing_ok=False)
  return tfmr.callbacks.TensorBoardImages(**kwargs)


def _get_ckpt_path(expdir, checkpoint_kwargs): # pylint: disable=missing-param-doc
  """Gets path to checkpoints."""
  # Get relevant checkpoint configuration.
  monitor = checkpoint_kwargs['monitor']
  save_weights_only = checkpoint_kwargs['save_weights_only']

  # The checkpoint directory. Create if necessary.
  path = os.path.join(expdir, 'ckpt')
  tf.io.gfile.makedirs(path)

  # The checkpoint filename.
  filename = 'weights' if save_weights_only else 'model'
  filename += '.{epoch:04d}-{' + monitor + ':.4f}.h5'

  return os.path.join(path, filename)


def _get_logs_path(params, expdir): # pylint: disable=unused-argument
  """Gets path to TensorBoard logs."""
  return os.path.join(expdir, 'logs')


def _get_pred_path(params, expdir): # pylint: disable=unused-argument
  """Gets path to predictions."""
  return os.path.join(expdir, 'pred')


def _get_image_path(params, expdir): # pylint: disable=unused-argument
  """Gets path to images."""
  return os.path.join(expdir, 'image')


def _get_eval_path(params, expdir): # pylint: disable=unused-argument
  """Gets path to evaluation results."""
  return os.path.join(expdir, 'eval')


def _get_tuner(params, hypermodel, expdir): # pylint: disable=missing-function-docstring

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
  kwargs['directory'] = expdir
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
