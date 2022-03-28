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
"""Keras objects registry.

Keras Declarative maintains its own object registry. There are a few differences
with respect to the Keras registry:

  * It includes non-serializable objects such as callbacks.
  * It does not prepend package prefixes to object names.
  * It supports objects of type `ObjectConfig` as identifiers.
"""

import inspect

import tensorflow as tf

from keras_declarative import config as config_module
from keras_declarative import hyperparams
from keras_declarative import predicates
from keras_declarative import util


def get_list(get_fn):
  """Returns a function that retrieves a list of objects.

  Args:
    get_fn: The get function to be used for individual identifiers.

  Returns:
    A function that retrieves an object or a list of objects.
  """
  def get_list_fn(identifier):
    """Retrieves a list of objects.

    Args:
      identifier: An object identifier. Must be a string, a dictionary, an
        `ObjectConfig` or `None`.

    Returns:
      A list of Keras objects as class instances.
    """
    if isinstance(identifier, list):
      return [get_fn(ident) for ident in identifier]
    return get_fn(identifier)
  return get_list_fn


def get_nest(get_fn):
  """Returns a function that retrieves a nested structure of objects.

  Nests include lists and dictionaries.

  Args:
    get_fn: The get function to be used for individual identifiers.

  Returns:
    A function that retrieves an object or a list of objects.
  """
  def get_nest_fn(identifier):
    """Retrieves a nested structure of objects.

    Args:
      identifier: An object identifier. Must be a string, a dictionary, an
        `ObjectConfig` or `None`.

    Returns:
      A list of Keras objects as class instances.
    """
    if isinstance(identifier, hyperparams.ParamsDict):
      identifier = identifier.as_dict()
    def _parse_nest(nest):
      if is_object_config(nest):
        return get_fn(nest)
      if isinstance(nest, dict):
        return {key: _parse_nest(value) for key, value in nest.items()}
      if isinstance(nest, list):
        return [_parse_nest(value) for value in nest]
      return get_fn(nest)
    return _parse_nest(identifier)
  return get_nest_fn


def get_callback(identifier):
  """Retrieve a Keras callback as a class instance.

  Args:
    identifier: A callback identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A Keras callback as a class instance.
  """
  return _get(identifier, _CALLBACK_OBJECTS, 'callback')


def get_layer(identifier):
  """Retrieve a Keras layer as a class instance.

  Args:
    identifier: A layer identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A Keras layer as a class instance.
  """
  return _get(identifier, _LAYER_OBJECTS, 'layer')


def get_loss(identifier):
  """Retrieve a Keras loss as a class instance.

  Args:
    identifier: A loss identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A Keras loss as a class instance.
  """
  return _get(identifier, _LOSS_OBJECTS, 'loss')


def get_metric(identifier):
  """Retrieve a Keras metric as a class instance.

  Args:
    identifier: A metric identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A Keras metric as a class instance.
  """
  return _get(identifier, _METRIC_OBJECTS, 'metric')


def get_optimizer(identifier):
  """Retrieve a Keras optimizer as a class instance.

  Args:
    identifier: An optimizer identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A Keras optimizer as a class instance.
  """
  return _get(identifier, _OPTIMIZER_OBJECTS, 'optimizer')


def get_predicate(identifier):
  """Retrieve a predicate as a class instance.

  Args:
    identifier: A predicate identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A predicate as a class instance.
  """
  return _get(identifier, _PREDICATE_OBJECTS, 'predicate')


def get_strategy(identifier):
  """Retrieve a TF distribution strategy as a class instance.

  Args:
    identifier: A strategy identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.

  Returns:
    A TF distribution strategy as a class instance.
  """
  return _get(identifier, _STRATEGY_OBJECTS, 'strategy')


def _get(identifier, objects, objtype):
  """Retrieve an object as a class instance.

  Args:
    identifier: An object identifier. Must be a string, a dictionary, an
      `ObjectConfig` or `None`.
    objects: A dictionary with the registered objects.
    objtype: A string with the type of object being retrieved. This is only used
      to format error messages.

  Returns:
    An instance of the object identified by `identifier`.

  Raises:
    ValueError: If the identifier is invalid.
    RuntimeError: If an error occurs while initializing the object.
  """
  # If object is an external object, don't try to resolve it.
  if isinstance(identifier, util.ExternalObject):
    return identifier

  if isinstance(identifier, config_module.ObjectConfig):
    identifier = identifier.as_dict()

  if not identifier: # Might be `None` or an empty dict.
    return None

  class_name, config = class_and_config_for_serialized_object(identifier)

  if class_name not in objects:
    raise ValueError(f"No known {objtype} with name: {class_name}")
  obj = objects[class_name]

  try:
    return obj(**config)
  except Exception as e:
    raise RuntimeError(
        f"An error occurred while initializing {class_name} with parameters: "
        f"{config}") from e


def class_and_config_for_serialized_object(identifier):
  """Returns the class name and config for a serialized object.

  Args:
    identifier: An object identifier. Must be a string, a dictionary or an
      `ObjectConfig`.

  Returns:
    A tuple containing the class name and its keyword arguments.

  Raises:
    ValueError: If the identifier is invalid.
  """
  if isinstance(identifier, config_module.ObjectConfig):
    identifier = identifier.as_dict()

  if isinstance(identifier, str):
    class_name, config = identifier, {}

  elif isinstance(identifier, dict):
    if 'class_name' not in identifier or 'config' not in identifier:
      raise ValueError(
          f"Invalid identifier: {identifier}. Value is not a valid "
          f"configuration dictionary.")
    class_name = identifier['class_name']
    config = identifier['config']

  else:
    raise ValueError(
        f"Invalid identifier: {identifier}. Value must be a string, a "
        f"dictionary or an `ObjectConfig`.")

  return class_name, config


def is_object_config(config):
  """Check if input is a valid object configuration dict.

  Args:
    config: The object to check.

  Returns:
    True if input is a valid object configuration dict, false otherwise.
  """
  # A str or None are valid object configs.
  if isinstance(config, (str, type(None))):
    return True

  # Otherwise, must be a dict or an object of type `ParamsDict`.
  if not isinstance(config, (dict, hyperparams.ParamsDict)):
    return False

  # If a dict, must have two keys: class_name and config.
  d = config.as_dict() if isinstance(config, hyperparams.ParamsDict) else config
  if set(d.keys()) != {'class_name', 'config'}:
    return False

  return True


def _find_objects(modules, objtype):
  """Finds objects of a certain type on the given modules.

  Args:
    modules: A list of modules to search for objects.
    objtype: The type of objects to be searched for.

  Returns:
    A dictionary containing the found objects.
  """
  objects = {}
  for module in modules:
    members = inspect.getmembers(module)
    for name, value in members:
      if inspect.isclass(value) and issubclass(value, objtype):
        objects[name] = value
  return objects


_CALLBACK_MODULES = [
  tf.keras.callbacks
]

_LAYER_MODULES = [
  tf.keras.layers
]

_LOSS_MODULES = [
  tf.keras.losses,
]

_METRIC_MODULES = [
  tf.keras.metrics,
]

_OPTIMIZER_MODULES = [
  tf.keras.optimizers
]

_PREDICATE_MODULES = [
  predicates
]

_STRATEGY_MODULES = [
  tf.distribute
]


# Try to discover objects from TensorFlow MRI, if it is installed.
try:
  import tensorflow_mri as tfmri
  _CALLBACK_MODULES.append(tfmri.callbacks)
  _LAYER_MODULES.extend([tfmri.layers])
  _LOSS_MODULES.append(tfmri.losses)
  _METRIC_MODULES.append(tfmri.metrics)
except ImportError:
  pass


# Try to discover objects from TF Playground, if it is installed.
try:
  import tf_playground as tfpg
  _CALLBACK_MODULES.append(tfpg.callbacks)
  _LAYER_MODULES.append(tfpg.layers)
  _LOSS_MODULES.append(tfpg.losses)
  _METRIC_MODULES.append(tfpg.metrics)
except ImportError:
  pass


_CALLBACK_OBJECTS = None
_LAYER_OBJECTS = None
_LOSS_OBJECTS = None
_METRIC_OBJECTS = None
_OPTIMIZER_OBJECTS = None
_PREDICATE_OBJECTS = None
_STRATEGY_OBJECTS = None


def discover_objects(custom_modules=None):
  """Discover Keras objects.

  By default, this function searches for Keras objects in core TensorFlow and
  TensorFlow MRI (if installed).

  Args:
    custom_modules: A list of custom modules to be searched for Keras objects.
  """
  global _CALLBACK_OBJECTS
  global _LAYER_OBJECTS
  global _LOSS_OBJECTS
  global _METRIC_OBJECTS
  global _OPTIMIZER_OBJECTS
  global _PREDICATE_OBJECTS
  global _STRATEGY_OBJECTS

  custom_modules = custom_modules or []

  _CALLBACK_OBJECTS = _find_objects(_CALLBACK_MODULES + custom_modules,
                                    tf.keras.callbacks.Callback)

  _LAYER_OBJECTS = _find_objects(_LAYER_MODULES + custom_modules,
                                tf.keras.layers.Layer)

  _LOSS_OBJECTS = _find_objects(_LOSS_MODULES + custom_modules,
                                tf.keras.losses.Loss)

  _METRIC_OBJECTS = _find_objects(_METRIC_MODULES + custom_modules,
                                  tf.keras.metrics.Metric)

  _OPTIMIZER_OBJECTS = _find_objects(_OPTIMIZER_MODULES + custom_modules,
                                     tf.keras.optimizers.Optimizer)

  _PREDICATE_OBJECTS = _find_objects(_PREDICATE_MODULES, predicates.Predicate)

  _STRATEGY_OBJECTS = _find_objects(_STRATEGY_MODULES, tf.distribute.Strategy)

discover_objects()
