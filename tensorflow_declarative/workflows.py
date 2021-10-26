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
"""TF workflows."""

import datetime
import functools
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import test
import tensorflow_io as tfio

from official.modeling import hyperparams
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import model_util

from tensorflow_declarative import config
from tensorflow_declarative import objects


def new_model(config_file):
  """Create and train a new model.
  
  Args:
    config_file: Path to the YAML configuration file.
  """
  # Get default config for this experiment.
  params = config.NewModelWorkflowConfig()

  # Do overrides from from `config_file`.
  for file in config_file or []:
    params = hyperparams.override_params_dict(params, file, is_strict=False)
  print(params)

  _set_global_config(params)
  expdir = _setup_experiment(params, config_file)
  datasets = _setup_datasets(params)
  model = _setup_model(params, datasets[0], datasets[1])


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
    return {k: hdf5_io_tensor(k).to_tensor() for k in hdf5_io_tensor.keys}
  
  def _set_static_shape(x, spec=None):
    return {k: tf.ensure_shape(v, spec[k].shape) for k, v in x.items()}

  def _get_outputs(x):
    out = [v for v in x.values()]
    return out[0] if len(out) == 1 else out

  # Create datasets.
  train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
  val_dataset = tf.data.Dataset.from_tensor_slices(val_files)
  test_dataset = tf.data.Dataset.from_tensor_slices(test_files)

  # Add data read.
  train_dataset = train_dataset.map(functools.partial(_read_hdf5, spec=train_spec))
  val_dataset = val_dataset.map(functools.partial(_read_hdf5, spec=val_spec))
  test_dataset = test_dataset.map(functools.partial(_read_hdf5, spec=test_spec))

  # Add static shape info.
  train_dataset = train_dataset.map(functools.partial(_set_static_shape, spec=train_spec))
  val_dataset = val_dataset.map(functools.partial(_set_static_shape, spec=val_spec))
  test_dataset = test_dataset.map(functools.partial(_set_static_shape, spec=test_spec))
  
  # Extract outputs.
  train_dataset = train_dataset.map(_get_outputs)
  val_dataset = val_dataset.map(_get_outputs)
  test_dataset = test_dataset.map(_get_outputs)

  def _add_transforms(dataset, transforms):
    for transform in transforms:
      if transform.type == 'map':
        map_func = objects.get_layer(transform.map.map_func)
        dataset = dataset.map(map_func,
                              num_parallel_calls=transform.map.num_parallel_calls,
                              deterministic=transform.map.deterministic)
      print(transform, dataset.element_spec)
    return dataset

  train_dataset = _add_transforms(train_dataset, params.data.train_transforms)
  val_dataset = _add_transforms(val_dataset, params.data.val_transforms)
  test_dataset = _add_transforms(test_dataset, params.data.test_transforms)

  return train_dataset, val_dataset, test_dataset


def _setup_model(params, train_dataset, val_dataset):
  """Create and train a model."""
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
  callbacks = objects.get_list(objects.get_callback)(params.training.callbacks)

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
