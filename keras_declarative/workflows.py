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
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import model_util

from keras_declarative import config
from keras_declarative import objects
from keras_declarative import util


def train_model(config_file):
  """Create and train a new model.
  
  Args:
    config_file: Path to the YAML configuration file.
  """
  # Get default config for this experiment.
  params = config.TrainModelWorkflowConfig()

  # Do overrides from from `config_file`.
  for file in config_file or []:
    params = hyperparams.override_params_dict(params, file, is_strict=False)
  print(params)

  _set_global_config(params)
  expdir = _setup_experiment(params, config_file)
  datasets, files = _setup_datasets(params)
  model = _train_model(params, datasets, expdir)
  predict = _do_predictions(params, model, datasets, files, expdir)


def _set_global_config(params):
  """Set global configuration."""
  if params.general.seed is not None:
    random.seed(params.general.seed)
    np.random.seed(params.general.seed)
    tf.random.set_seed(params.general.seed)


def _setup_experiment(params, config_file):
  """Set up experiment directory."""
  if params.general.path is None:
    raise ValueError("`general.path` must be provided.")
  expname = params.general.name or os.path.splitext(os.path.basename(config_file[-1]))[0]
  expname += '_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  expdir = os.path.join(params.general.path, expname)
  tf.io.gfile.makedirs(expdir)
  return expdir


def _setup_datasets(params):
  """Set up datasets."""
  if params.data.sources is None:
    raise ValueError("`data.sources` must be provided.")

  train_files = []
  val_files = []
  test_files = []

  for source in params.data.sources:
    files = io_util.get_distributed_hdf5_filenames(source.path, source.prefix)
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

  # Convert to tensor.
  train_files = tf.convert_to_tensor(train_files, dtype=tf.string)
  val_files = tf.convert_to_tensor(val_files, dtype=tf.string)
  test_files = tf.convert_to_tensor(test_files, dtype=tf.string)

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
  train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
  val_dataset = tf.data.Dataset.from_tensor_slices(val_files)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_files)

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

  def _add_transforms(dataset, transforms):
    for transform in transforms:
      if transform.type == 'batch':
        dataset = dataset.batch(transform.batch.batch_size,
                                drop_remainder=transform.batch.drop_remainder,
                                num_parallel_calls=transform.batch.num_parallel_calls,
                                deterministic=transform.batch.deterministic)
      elif transform.type == 'map':
        map_func = objects.get_layer(transform.map.map_func)
        dataset = dataset.map(_get_map_func(map_func, transform.map.component),
                              num_parallel_calls=transform.map.num_parallel_calls,
                              deterministic=transform.map.deterministic)

    return dataset

  train_dataset = _add_transforms(train_dataset, params.data.train_transforms)
  val_dataset = _add_transforms(val_dataset, params.data.val_transforms)
  test_dataset = _add_transforms(test_dataset, params.data.test_transforms)

  datasets = train_dataset, val_dataset, test_dataset
  files = train_files, val_files, test_files

  return datasets, files


def _train_model(params, datasets, expdir):
  """Create and train a model."""
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
  model = model_util.model_from_layers(layer, input_spec)

  print(model.summary())

  optimizer = objects.get_optimizer(params.training.optimizer)
  loss = objects.get_list(objects.get_loss)(params.training.loss)
  metrics = objects.get_list(objects.get_metric)(params.training.metrics)
  callbacks = _process_callbacks(params, expdir)

  model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics or None,
                loss_weights=params.training.loss_weights or None,
                weighted_metrics=params.training.weighted_metrics or None,
                run_eagerly=params.training.run_eagerly,
                steps_per_execution=params.training.steps_per_execution)

  model.fit(x=train_dataset,
            epochs=params.training.epochs,
            verbose=params.training.verbose,
            callbacks=callbacks,
            validation_data=val_dataset)

  return model


def _do_predictions(params, model, datasets, files, expdir):

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

      x = _flatten_structure_and_unbatch(x)
      y = _flatten_structure_and_unbatch(y) or itertools.repeat(None)
      y_pred = _flatten_structure_and_unbatch(y_pred) or itertools.repeat(None)

      # For each element in batch.
      for e_x, e_y, e_y_pred in zip(x, y, y_pred):
        d = {}
        d.update({name: value for name, value in zip(input_names, e_x)})
        d.update({name: value for name, value in zip(output_names, e_y)})
        d.update({name + '_pred': value for name, value in zip(output_names, e_y_pred)})
        io_util.write_hdf5(os.path.join(pred_path, files[name].pop(0)), d)

      progbar.add(1)


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

    def _map_func(x):
      x[component] = map_func(x[component])
      return x

    return _map_func


def _process_callbacks(params, expdir):

  # TODO: Complete callback configs.
  callback_configs = params.training.callbacks

  # Parse callback configs.
  callbacks = objects.get_list(objects.get_callback)(callback_configs)

  # Add default training callbacks.
  if params.training.use_default_callbacks:
    callbacks = _add_default_training_callbacks(callbacks, params, expdir)

  return callbacks


def _add_default_training_callbacks(callbacks, params, expdir):

  # Add model checkpoint callback, if there isn't one already.
  if not any(
      isinstance(c, tf.keras.callbacks.ModelCheckpoint) for c in callbacks):
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=_get_ckpt_path(params, expdir),
            monitor=_CKPT_MONITOR_DEFAULT,
            verbose=1,
            save_best_only=_CKPT_SAVE_BEST_ONLY_DEFAULT,
            save_weights_only=_CKPT_SAVE_WEIGHTS_ONLY_DEFAULT,
            mode='auto',
            save_freq='epoch'))

  # Add Tensorboard callback, if there isn't one already.
  if not any(
      isinstance(c, tf.keras.callbacks.TensorBoard) for c in callbacks):
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=_get_logs_path(params, expdir),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=2))

  return callbacks


def _flatten_structure_and_unbatch(structure):
  if structure is None:
    return structure
  return [tf.unstack(x) for x in tf.nest.flatten(structure)]


def _get_ckpt_path(params, expdir, ckpt_config=None):

  # Get relevant checkpoint configuration.
  ckpt_config = ckpt_config or {}
  monitor = ckpt_config.get(
      'monitor', _CKPT_MONITOR_DEFAULT)
  save_best_only = ckpt_config.get(
      'save_best_only', _CKPT_SAVE_BEST_ONLY_DEFAULT)
  save_weights_only = ckpt_config.get(
      'save_weights_only', _CKPT_SAVE_WEIGHTS_ONLY_DEFAULT)

  # The checkpoint directory. Create if necessary.
  path = os.path.join(expdir, 'ckpt')
  tf.io.gfile.makedirs(path)

  # The checkpoint filename.
  filename = 'weights' if save_weights_only else 'model'
  filename += '.{epoch:03d}-{' + monitor + ':.3f}.h5'

  return os.path.join(path, filename)


def _get_logs_path(params, expdir):

  return os.path.join(expdir, 'logs')


def _get_pred_path(params, expdir):

  return os.path.join(expdir, 'pred')


_CKPT_MONITOR_DEFAULT = 'val_loss'
_CKPT_SAVE_BEST_ONLY_DEFAULT = True
_CKPT_SAVE_WEIGHTS_ONLY_DEFAULT = False


class defaultdict(collections.defaultdict):

  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret
