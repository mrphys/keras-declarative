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
"""TFDP objects."""

import inspect

import tensorflow as tf

from tensorflow_declarative import config as config_module


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
  """
  if identifier is None:
    return None
  
  if isinstance(identifier, config_module.ObjectConfig):
    identifier = {
      'class_name': identifier.class_name,
      'config': identifier.config.as_dict()
    }

  if isinstance(identifier, str):
    class_name, config = identifier, {}

  elif isinstance(identifier, dict):
    if 'class_name' not in identifier or 'config' not in identifier:
      raise ValueError(
          f"Invalid identifier: {identifier}. Value is not a valid {objtype} "
          f"configuration dictionary.")
    class_name = identifier['class_name']
    config = identifier['config']

  else:
    raise ValueError(
        f"Invalid identifier: {identifier}. Value must be a string, a "
        f"dictionary or an `ObjectConfig`.")

  if class_name not in objects:
    raise ValueError(f"No known {objtype} with name: {class_name}")
  obj = objects[class_name]

  try:
    return obj(**config)
  except Exception as e:
    raise RuntimeError(
        f"An error occurred while initializing {class_name} with parameters: "
        f"{config}") from e


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


try:
  import tensorflow_mri as tfmr
  _CALLBACK_MODULES.append(tfmr.callbacks)
  _LAYER_MODULES.append(tfmr.layers)
  _LOSS_MODULES.append(tfmr.losses)
  _METRIC_MODULES.append(tfmr.metrics)
except ImportError:
  pass


_CALLBACK_OBJECTS = _find_objects(_CALLBACK_MODULES,
                                  tf.keras.callbacks.Callback)

_LAYER_OBJECTS = _find_objects(_LAYER_MODULES,
                               tf.keras.layers.Layer)

_LOSS_OBJECTS = _find_objects(_LOSS_MODULES,
                              tf.keras.losses.Loss)

_METRIC_OBJECTS = _find_objects(_METRIC_MODULES,
                                tf.keras.metrics.Metric)

_OPTIMIZER_OBJECTS = _find_objects(_OPTIMIZER_MODULES,
                                   tf.keras.optimizers.Optimizer)
