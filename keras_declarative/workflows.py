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
import itertools
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from official.modeling import hyperparams

from keras_declarative import config
from keras_declarative import io
from keras_declarative import objects
from keras_declarative import util


def train_model(config_file):
  """Create and train a new model.
  
  Args:
    config_file: A list of paths to the YAML configuration files.
  """
  # Get default config for this experiment.
  params = config.TrainModelWorkflowConfig()

  # Do overrides from from `config_file`.
  for file in config_file or []:
    params = hyperparams.override_params_dict(params, file, is_strict=False)

  _set_global_config(params)
  expdir = _setup_directory(params, config_file)
  datasets, files = _setup_datasets(params)
  model = _train_model(params, datasets, expdir)
  _do_predictions(params, model, datasets, files, expdir)


def _set_global_config(params):
  """Set global configuration.
  
  Args:
    params: A `config.TrainModelWorkflowConfig`.
  """
  if params.experiment.seed is not None:
    random.seed(params.experiment.seed)
    np.random.seed(params.experiment.seed)
    tf.random.set_seed(params.experiment.seed)


def _setup_directory(params, config_file):
  """Set up experiment directory.

  Args:
    params: A `config.TrainModelWorkflowConfig`.

  Returns:
    The experiment directory.
  """
  path = params.experiment.path or os.getcwd()
  expname = params.experiment.name or os.path.splitext(os.path.basename(config_file[-1]))[0]
  expname += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  expdir = os.path.join(path, expname)
  tf.io.gfile.makedirs(expdir)
  return expdir


def _setup_datasets(params):
  """Set up datasets.
  
  Args:
    params: A `config.TrainModelWorkflowConfig`.

  Returns:
    A tuple of three datasets (train, validation, test) and a tuple of three
    lists of files.
  """
  if params.data.sources is None:
    raise ValueError("`data.sources` must be provided.")

  train_files = []
  val_files = []
  test_files = []

  for source in params.data.sources:
    files = io.get_distributed_hdf5_filenames(source.path, source.prefix)
    n = len(files)

    if source.split.mode == 'random':
      random.shuffle(files)

    get_n = lambda s: s if isinstance(s, int) else int(s * n)
    n_train = get_n(source.split.train)
    n_val = get_n(source.split.val)
    n_test = get_n(source.split.test)

    train_files.extend(files[0:n_train])
    val_files.extend(files[n_train:n_train + n_val])
    test_files.extend(files[n_train + n_val:n_train + n_val + n_test])

  # Shuffle all the data.
  random.shuffle(train_files)
  random.shuffle(val_files)
  random.shuffle(test_files)

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
    out = [v for v in x.values()]
    return out[0] if len(out) == 1 else out

  # Create datasets.
  train_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(train_files, dtype=tf.string))
  val_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(val_files, dtype=tf.string))
  test_dataset = tf.data.Dataset.from_tensor_slices(
      tf.convert_to_tensor(test_files, dtype=tf.string))

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

  def _add_transforms(dataset, transforms, options, dstype):
    """Add configured transforms to dataset.
    
    Args:
      dataset: A `tf.data.Dataset`.
      transforms: A list of `DataTransformConfig`.
      options: A `DataOptionsConfig`.
      dstype: The type of this dataset. One of `'train'`, `'val'` or `'test'`.

    Returns:
      A `tf.data.Dataset`.
    """
    for transform in transforms or []:
      if transform.type == 'batch':
        dataset = dataset.batch(transform.batch.batch_size,
                                drop_remainder=transform.batch.drop_remainder,
                                num_parallel_calls=transform.batch.num_parallel_calls,
                                deterministic=transform.batch.deterministic)

      elif transform.type == 'cache':
        dataset = dataset.cache(filename=transform.cache.filename)

      elif transform.type == 'map':
        map_func = objects.get_layer(transform.map.map_func)
        dataset = dataset.map(_get_map_func(map_func, transform.map.component),
                              num_parallel_calls=transform.map.num_parallel_calls,
                              deterministic=transform.map.deterministic)

      elif transform.type == 'shuffle':
        if dstype != 'train' and options.shuffle_training_only:
          continue
        dataset = dataset.shuffle(transform.shuffle.buffer_size,
                                  seed=transform.shuffle.seed,
                                  reshuffle_each_iteration=transform.shuffle.reshuffle_each_iteration)

      else:
        raise ValueError(f"Unknown transform type: {transform.type}")

    return dataset

  # Add user-specified transforms to each dataset.
  train_dataset = _add_transforms(
      train_dataset, params.data.train_transforms, params.data.options, 'train')
  val_dataset = _add_transforms(
      val_dataset, params.data.val_transforms, params.data.options, 'val')
  test_dataset = _add_transforms(
      test_dataset, params.data.test_transforms, params.data.options, 'test')

  # Set the dataset options.
  train_dataset = _set_dataset_options(train_dataset, params.data.options)
  val_dataset = _set_dataset_options(val_dataset, params.data.options)
  test_dataset = _set_dataset_options(test_dataset, params.data.options)

  # Pack and return.
  datasets = train_dataset, val_dataset, test_dataset
  files = train_files, val_files, test_files
  return datasets, files


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
    options.threading.max_intra_op_parallelism = options_config.max_intra_op_parallelism
  if options_config.private_threadpool_size is not None:
    options.threading.private_threadpool_size = options_config.private_threadpool_size
  
  return dataset.with_options(options)


def _train_model(params, datasets, expdir):
  """Create and train a model.
  
  Args:
    params: A `config.TrainModelWorkflowConfig`.
  """
  train_dataset, val_dataset, _ = datasets

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

  # Print model summary.
  model.summary(line_length=80)

  optimizer = objects.get_optimizer(params.training.optimizer)
  loss = objects.get_list(objects.get_loss)(params.training.loss)
  metrics = objects.get_list(objects.get_metric)(params.training.metrics)
  callbacks = _process_callbacks(params, expdir, datasets)

  model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics or None,
                loss_weights=params.training.loss_weights or None,
                weighted_metrics=params.training.weighted_metrics or None,
                run_eagerly=params.training.run_eagerly,
                steps_per_execution=params.training.steps_per_execution)

  print("Training...")
  model.fit(x=train_dataset,
            epochs=params.training.epochs,
            verbose=params.training.verbose,
            callbacks=callbacks,
            validation_data=val_dataset)
  print("Training complete.")

  return model


def _do_predictions(params, model, datasets, files, expdir):

  print("Predicting...")
  train_dataset, val_dataset, test_dataset = datasets
  train_files, val_files, test_files = files

  datasets = {
      'train': train_dataset,
      'val': val_dataset,
      'test': test_dataset
  }

  files = {
      'train': train_files,
      'val': val_files,
      'test': test_files
  }

  if isinstance(params.predict.datasets, str):
    dataset_keys = [params.predict.datasets]
  else:
    dataset_keys = params.predict.datasets

  datasets = {k: datasets[k] for k in dataset_keys}

  pred_path = _get_pred_path(params, expdir)
  tf.io.gfile.makedirs(pred_path)

  input_names = defaultdict(lambda key: 'input_' + str(key))
  output_names = defaultdict(lambda key: 'output_' + str(key))

  # TODO: add possibility of using specified names.

  for name, dataset in datasets.items():
    path = os.path.join(pred_path, name)
    tf.io.gfile.makedirs(path)

    progbar = tf.keras.utils.Progbar(dataset.cardinality().numpy())

    for element in dataset:
      x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(element)
      y_pred = model(x, training=False)

      x = _flatten_and_unbatch_nested_tensors(x)
      y = _flatten_and_unbatch_nested_tensors(y) or itertools.repeat(None)
      y_pred = _flatten_and_unbatch_nested_tensors(y_pred) or itertools.repeat(None)

      # For each element in batch.
      for e_x, e_y, e_y_pred in zip(x, y, y_pred):
        d = {}
        d.update({input_names[idx]: value for idx, value in enumerate(e_x)})
        d.update({output_names[idx]: value for idx, value in enumerate(e_y)})
        d.update({output_names[idx] + '_pred': value for idx, value in enumerate(e_y_pred)})
        d = {k: v.numpy() for k, v in d.items()}

        file_path = os.path.join(path, os.path.basename(files[name].pop(0)))
        io.write_hdf5(file_path, d)

      progbar.add(1)

  print("Prediction complete.")


def _parse_spec_config(spec_config):
  """Parse spec configuration.
  
  Converts a list of `TensorSpecConfig` to a nested structure of
  `tf.TensorSpec`.

  Args:
    spec_config: A list of `TensorSpecConfig`.

  Returns:
    A nested structure of `tf.TensorSpec`.
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
    return {spec.name: tf.TensorSpec(spec.shape, spec.dtype, spec.name) for spec in spec_config}
  else:
    return [tf.TensorSpec(spec.shape, spec.dtype) for spec in spec_config]


def _get_map_func(map_func, component=None):

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


def _process_callbacks(params, expdir, datasets):

  _, val_dataset, _ = datasets

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
    elif (isinstance(value, config.ObjectConfig) and
        value.class_name == 'ModelCheckpoint'):
      checkpoint_kwargs = value.config.as_dict()
      continue

    if isinstance(value, str) and value == 'TensorBoard':
      tensorboard_kwargs = {}
      continue
    elif (isinstance(value, config.ObjectConfig) and
        value.class_name == 'TensorBoard'):
      tensorboard_kwargs = value.config.as_dict()
      continue

    if isinstance(value, str) and value == 'TensorBoardImages':
      images_kwargs = {}
      continue
    elif (isinstance(value, config.ObjectConfig) and
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

  if checkpoint_callback:
    callbacks.append(checkpoint_callback)
  if tensorboard_callback:
    callbacks.append(tensorboard_callback)
  if images_callback:
    callbacks.append(images_callback)

  return callbacks


def _flatten_and_unbatch_nested_tensors(structure):

  if structure is None:
    return structure
  return [tf.unstack(x) for x in tf.nest.flatten(structure)]


def _get_checkpoint_callback(params, expdir, kwargs=None):
  
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

  if kwargs is None:
    return None

  default_kwargs = dict(
      x=dataset,
      log_dir=_get_logs_path(params, expdir)
  )

  kwargs = {**default_kwargs, **kwargs} if kwargs else default_kwargs

  try:
    import tensorflow_mri as tfmr
    return tfmr.callbacks.TensorBoardImages(**kwargs)
  except ImportError as err:
    raise ValueError("TensorFlow MRI is required to use the TensorBoardImages "
                     "callback.") from err


def _get_ckpt_path(expdir, checkpoint_kwargs):

  # Get relevant checkpoint configuration.
  monitor = checkpoint_kwargs['monitor']
  save_best_only = checkpoint_kwargs['save_best_only']
  save_weights_only = checkpoint_kwargs['save_weights_only']

  # The checkpoint directory. Create if necessary.
  path = os.path.join(expdir, 'ckpt')
  tf.io.gfile.makedirs(path)

  # The checkpoint filename.
  filename = 'weights' if save_weights_only else 'model'
  filename += '.{epoch:04d}-{' + monitor + ':.4f}.h5'

  return os.path.join(path, filename)


def _get_logs_path(params, expdir):

  return os.path.join(expdir, 'logs')


def _get_pred_path(params, expdir):

  return os.path.join(expdir, 'pred')


class defaultdict(collections.defaultdict):

  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret
