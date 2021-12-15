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

  # Build model using functional API.
  return tf.keras.Model(inputs=inputs, outputs=outputs)


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


class TunerMixin(kt.Tuner):
  """Mixin for Keras Tuner tuners."""
  def _configure_tensorboard_dir(self, callbacks, trial, execution=0):
    super()._configure_tensorboard_dir(callbacks, trial, execution)

    for callback in callbacks:
      if callback.__class__.__name__ == "TensorBoardImages":
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


class RandomSearch(TunerMixin, kt.RandomSearch):
  """Random search tuner with mixins."""


class BayesianOptimization(TunerMixin, kt.BayesianOptimization):
  """Bayesian optimization tuner with mixins."""


class Hyperband(TunerMixin, kt.Hyperband):
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
