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
"""Predicates."""

import tensorflow as tf


class Predicate():
  """A predicate."""
  def __init__(self):
    """Initializes the predicate."""

  def __call__(self, inputs):
    """Gets predicate result for the given inputs."""
    return tf.constant(True)


class ShapePredicate(Predicate):
  """A predicate used to verify shapes."""
  def __init__(self,
               inclusive_min_shape=None,
               inclusive_max_shape=None,
               exclusive_min_shape=None,
               exclusive_max_shape=None):
    """Initializes the predicate."""
    super().__init__()
    self._inclusive_min_shape = inclusive_min_shape
    self._inclusive_max_shape = inclusive_max_shape
    self._exclusive_min_shape = exclusive_min_shape
    self._exclusive_max_shape = exclusive_max_shape

  def __call__(self, inputs):
    """Gets predicate result for the given inputs."""
    result = tf.constant(True)

    if self._inclusive_min_shape is not None:
      result = tf.math.logical_and(result, tf.math.reduce_all(
          tf.shape(inputs) >= self._inclusive_min_shape))

    if self._inclusive_max_shape is not None:
      result = tf.math.logical_and(result, tf.math.reduce_all(
          tf.shape(inputs) <= self._inclusive_max_shape))

    if self._exclusive_min_shape is not None:
      result = tf.math.logical_and(result, tf.math.reduce_all(
          tf.shape(inputs) < self._exclusive_min_shape))

    if self._exclusive_max_shape is not None:
      result = tf.math.logical_and(result, tf.math.reduce_all(
          tf.shape(inputs) < self._exclusive_max_shape))

    return result
