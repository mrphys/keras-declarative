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
"""Utilities."""

import keras_tuner as kt
import tensorflow as tf


def model_from_layers(layers, input_spec):
  """Create a Keras model from the given layers and input specification.

  Args:
    layers: A `tf.keras.layers.Layer` or a list thereof.
    input_spec: A nested structure of `tf.TensorSpec` objects.

  Returns:
    A `tf.keras.Model`.
  """
  if isinstance(layers, tf.keras.layers.Layer):
    layers = [layers]

  # If `input_spec` is a dict, set the `TensorSpec` names to the value of the
  # corresponding dict keys.
  if isinstance(input_spec, dict):
    for k, v in input_spec.items():
      input_spec[k] = tf.TensorSpec.from_spec(v, name=k)

  # Generate inputs with the passed specification.
  def _make_input(spec):

    if spec.shape == None: # pylint: disable=singleton-comparison
      return tf.keras.Input(shape=None, batch_size=None, dtype=spec.dtype)

    return tf.keras.Input(shape=spec.shape[1:],
                          batch_size=spec.shape[0],
                          name=spec.name,
                          dtype=spec.dtype)

  inputs = tf.nest.map_structure(_make_input, input_spec)

  # Forward pass.
  outputs = inputs
  for layer in layers:
    outputs = layer(outputs)

  # If output is a dictionary, create a dummy identity layer for each output.
  # This has no effect on model output but results in nicer output naming
  # throughout Keras functions.
  if isinstance(outputs, dict):
    outputs = {k: tf.keras.layers.Layer(name=k)(v) for k, v in outputs.items()}

  # Build model using functional API.
  return tf.keras.Model(inputs=inputs, outputs=outputs)


class DatasetContainer():
  """A container for datasets.

  Holds multiple `tf.data.Datasets` and their corresponding example keys.

  Args:
    keys_and_datasets: A dict of (example keys, dataset) tuples keyed by the
      dataset names.
  """
  def __init__(self, keys_and_datasets):
    self._keys_and_datasets = {k: list(v) for k, v in keys_and_datasets.items()}
    self._cachefiles = []
    self._selected_names = None

  @property
  def datasets(self):
    return {k: v[1] for k, v in self._keys_and_datasets.items()}

  @property
  def example_ids(self):
    return {k: v[0] for k, v in self._keys_and_datasets.items()}

  @property
  def names(self):
    return list(self._keys_and_datasets.keys())

  @property
  def cachefiles(self):
    return self._cachefiles

  @property
  def train_ds(self):
    return self.datasets['train']

  @property
  def val_ds(self):
    return self.datasets['val']

  @property
  def test_ds(self):
    return self.datasets['test']

  @property
  def train_keys(self):
    return self.example_ids['train']

  @property
  def val_keys(self):
    return self.example_ids['val']

  @property
  def test_keys(self):
    return self.example_ids['test']

  def __len__(self):
    """Returns the number of datasets held by this container."""
    return len(self._datasets)

  def __getitem__(self, name):
    """Returns the keys-dataset pair for the given dataset name."""
    return tuple(self._keys_and_datasets[name])

  def __repr__(self):
    return f"DatasetContainer(names={self.names})"

  def __add__(self, other):
    """Combines two dataset containers."""
    if not isinstance(other, DatasetContainer):
      raise TypeError(f"`other` must be a DatasetContainer, got {type(other)}.")

    if self.names != other.names:
      raise ValueError(f"Dataset names must match, got {self.names} and "
                       f"{other.names}.")

    return DatasetContainer({
        name: (self.example_ids[name] + other.example_ids[name],
               self.datasets[name].concatenate(other.datasets[name]))
        for name in self.names
    })

  def select(self, names):
    """Selects the specified datasets for future transformations."""
    if not isinstance(names, (list, tuple)):
      names = [names]
    self._selected_names = names
    return self

  def unselect(self):
    self._selected_names = None
    return self

  def apply(self, *args, **kwargs):
    """Calls `apply` on one or more datasets held by this container."""
    return self._transform_datasets('apply', *args, **kwargs)

  def batch(self, *args, **kwargs):
    """Calls `batch` on one or more datasets held by this container."""
    return self._transform_datasets('batch', *args, **kwargs)

  def cache(self, filename='', name=None):  # pylint: disable=missing-param-doc
    """Calls `cache` on one or more datasets held by this container."""
    # We need to copy-paste here to adapt the behaviour of
    # `_transform_datasets`.
    keys_and_datasets = {name: [keys, ds] for name, (keys, ds)
                         in self._keys_and_datasets.items()}
    for ds_name in (self._selected_names or self.names):
      cachefile = filename + f'-{ds_name}' if filename else filename
      if cachefile:
        self._cachefiles.append(cachefile)
      keys_and_datasets[ds_name][1] = self.datasets[ds_name].cache(
          filename=cachefile, name=name)
    new_container = DatasetContainer(keys_and_datasets)
    new_container._cachefiles = self._cachefiles  # pylint: disable=protected-access
    new_container._selected_names = self._selected_names  # pylint: disable=protected-access
    return new_container

  def flat_map(self, *args, **kwargs):
    """Calls `flat_map` on one or more datasets held by this container."""
    return self._transform_datasets('flat_map', *args, **kwargs)

  def map(self, *args, **kwargs):
    """Calls `map` on one or more datasets held by this container."""
    return self._transform_datasets('map', *args, **kwargs)

  def prefetch(self, *args, **kwargs):
    """Calls `prefetch` on one or more datasets held by this container."""
    return self._transform_datasets('prefetch', *args, **kwargs)

  def shuffle(self, *args, **kwargs):
    """Calls `shuffle` on one or more datasets held by this container."""
    return self._transform_datasets('shuffle', *args, **kwargs)

  def with_options(self, *args, **kwargs):
    """Calls `with_options` on one or more datasets held by this container."""
    return self._transform_datasets('with_options', *args, **kwargs)

  def _transform_datasets(self, method, *args, **kwargs):  # pylint: disable=missing-param-doc
    """Calls method `method` on one or more datasets held by this container."""
    keys_and_datasets = {name: [keys, ds] for name, (keys, ds)
                         in self._keys_and_datasets.items()}
    for ds_name in (self._selected_names or self.names):
      keys_and_datasets[ds_name][1] = getattr(self.datasets[ds_name], method)(
          *args, **kwargs)
    new_container = DatasetContainer(keys_and_datasets)
    new_container._cachefiles = self._cachefiles  # pylint: disable=protected-access
    new_container._selected_names = self._selected_names  # pylint: disable=protected-access
    return new_container


class ExternalObject():
  """An object loaded from an external module.

  This class is used to wrap objects that are loaded from an external module and
  are not managed by Keras Declarative.
  """
  def __init__(self, obj, args, kwargs, filename):
    self._obj = obj
    self._args = args
    self._kwargs = kwargs
    self._filename = filename

  def __call__(self, *args, **kwargs):
    return self._obj(*self._args, **self._kwargs)(*args, **kwargs)

  @property
  def obj(self):
    return self._obj

  @property
  def filename(self):
    return self._filename


class TunablePlaceholder():
  """A hyperparameter placeholder.

  Args:
    type_: A string. The type of the hyperparameter.
    kwargs: A dictionary. The keyword arguments defining the hyperparameter.
  """
  def __init__(self, type_, kwargs):
    self.type_ = type_
    self.kwargs = kwargs

  def __call__(self, hp):
    return getattr(hp, self.type_)(**self.kwargs)


class TunerMixin(kt.Tuner):  # pylint: disable=abstract-method
  """Mixin for Keras Tuner tuners."""
  def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
    super()._configure_tensorboard_dir(callbacks, trial, execution)

    for callback in callbacks:
      if callback.__class__.__name__.startswith("TensorBoardImages"):
        # Patch TensorBoardImages log_dir.
        logdir = self._get_tensorboard_dir(
            callback.log_dir, trial.trial_id, execution
        )
        callback.log_dir = logdir

  def run_trial(self, trial, *args, **kwargs):
    try:
      return super().run_trial(trial, *args, **kwargs)
    except Exception as e: # pylint: disable=broad-except
      print(f"Trial '{trial.trial_id}' failed with exception:", e)
      trial.status = kt.engine.trial.TrialStatus.INVALID
      return {self.oracle.objective.name: float("-inf")
              if self.oracle.objective.direction == "max" else float("inf")}


class RandomSearch(TunerMixin, kt.RandomSearch):  # pylint: disable=abstract-method
  """Random search tuner with mixins."""


class BayesianOptimization(TunerMixin, kt.BayesianOptimization):  # pylint: disable=abstract-method
  """Bayesian optimization tuner with mixins."""


class Hyperband(TunerMixin, kt.Hyperband):  # pylint: disable=abstract-method
  """Hyperband tuner with mixins."""
  def _build_hypermodel(self, hp):
    """Builds a hypermodel.

    There seems to be a bug in the `kt.Hyperband` implementation. It overrides a
    `_build_model` function which does not seem to exist in the `kt.Tuner` base
    class. As a result, model weights are not loaded from previous trials as
    would be expected.

    This function is the same as `Hyperband._build_model` but has the
    correct name `_build_hypermodel`.

    Args:
      hp: A `kt.HyperParameters` object.

    Returns:
      A `tf.keras.Model`.
    """
    model = super()._build_hypermodel(hp)

    # Load previous checkpoint if requested.
    if "tuner/trial_id" in hp.values:
      trial_id = hp.values["tuner/trial_id"]
      history_trial = self.oracle.get_trial(trial_id)

      model.load_weights(
          self._get_checkpoint_fname(
              history_trial.trial_id, history_trial.best_step
          )
      )
    return model


def is_float(s):
  """Returns `True` is string is a valid representation of a float."""
  try:
    float(s)
    return True
  except ValueError:
    return False


def is_percent(s):
  """Returns `True` is string is a valid representation of a percentage."""
  return s.endswith("%") and is_float(s[:-1])


def percent_to_float(s):
  """Converts a percentage string to a float."""
  return float(s[:-1]) / 100
