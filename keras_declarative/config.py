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
"""Configuration."""

import dataclasses
from typing import List, Union, Optional

from official.modeling import hyperparams


# NOTE: This file is more easily read from the bottom up, as the more generic
# configuration elements are at the bottom and become more specific towards the
# top.


@dataclasses.dataclass
class DataSplitConfig(hyperparams.Config):
  """Data split configuration."""
  train: float = 0.0
  val: float = 0.0
  test: float = 0.0
  mode: str = 'random'


@dataclasses.dataclass
class DataSourceConfig(hyperparams.Config):
  """Data source configuration."""
  path: str = None
  prefix: str = None
  split: DataSplitConfig = DataSplitConfig()


@dataclasses.dataclass
class TensorSpecConfig(hyperparams.Config):
  """Tensor specification configuration."""
  name: Optional[Union[str, int]] = None
  shape: List[int] = None
  dtype: str = 'float32'


@dataclasses.dataclass
class GenericConfig(hyperparams.Config):
  """Generic config."""
  def _set(self, k, v):
    self.__dict__[k] = v


@dataclasses.dataclass
class ObjectConfig(hyperparams.Config):
  """Object configuration."""
  class_name: str = None
  config: GenericConfig = GenericConfig()


@dataclasses.dataclass
class BatchTransformConfig(hyperparams.Config):
  """Batch transform configuration."""
  batch_size: int = None
  drop_remainder: bool = False
  num_parallel_calls: Optional[int] = None
  deterministic: Optional[bool] = None


@dataclasses.dataclass
class CacheTransformConfig(hyperparams.Config):
  """Cache transform configuration."""
  filename: str = ''


@dataclasses.dataclass
class MapTransformConfig(hyperparams.Config):
  """Data transform configuration."""
  map_func: ObjectConfig = ObjectConfig()
  num_parallel_calls: Optional[int] = None
  deterministic: Optional[bool] = None
  component: Optional[Union[int, str]] = None


@dataclasses.dataclass
class ShuffleTransformConfig(hyperparams.Config):
  """Shuffle transform configuration."""
  buffer_size: int = None
  seed: Optional[int] = None
  reshuffle_each_iteration: Optional[bool] = None


@dataclasses.dataclass
class DataTransformConfig(hyperparams.OneOfConfig):
  """Data transform configuration."""
  type: str = None
  batch: BatchTransformConfig = BatchTransformConfig()
  cache: CacheTransformConfig = CacheTransformConfig()
  map: MapTransformConfig = MapTransformConfig()
  shuffle: ShuffleTransformConfig = ShuffleTransformConfig()


class DataOptionsConfig(hyperparams.Config):
  """Data options configuration."""
  shuffle_training_only: bool = True
  max_intra_op_parallelism: int = None
  private_threadpool_size: int = None


@dataclasses.dataclass
class DataConfig(hyperparams.Config):
  """Data configuration."""
  sources: List[DataSourceConfig] = dataclasses.field(default_factory=list)
  train_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  val_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  test_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  train_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long
  val_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long
  test_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long
  options: DataOptionsConfig = DataOptionsConfig()


@dataclasses.dataclass
class ExperimentConfig(hyperparams.Config):
  """Experiment configuration.

  Attributes:
    name: The name of this experiment. Defaults to
      `<config_filename>_<datetime>`.
    path: The path to this experiment. Results will be saved in a new directory
      `path/name`. Defaults to current working directory.
    seed: A global seed to be used for all random number generators.
  """
  name: Optional[str] = None
  path: Optional[str] = None
  seed: Optional[int] = None


@dataclasses.dataclass
class NewModelConfig(hyperparams.Config):
  """Model configuration.

  Attributes:
    network: A `str` or `ObjectConfig` defining a `tf.keras.layers.Layer` or a
      list thereof, implementing a sequential network architecture.
    input_spec: A list of `TensorSpecConfig` defining the model input
      specification. If not specified, we will attempt to infer the input
      specification from the training dataset.
  """
  network: List[ObjectConfig] = ObjectConfig()
  input_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ExistingModelConfig(hyperparams.Config):
  """Existing model configuration.
  
  Attributes:
    path: A `str`. Path to an existing model. Defaults to `None`. If not `None`,
      loads this model ignoring the remaining arguments.
  """
  path: str = None


@dataclasses.dataclass
class ModelConfig(hyperparams.OneOfConfig):
  """Model configuration."""
  type: str = 'new'
  new: NewModelConfig = NewModelConfig()
  existing: ExistingModelConfig = ExistingModelConfig()


@dataclasses.dataclass
class TrainingConfig(hyperparams.Config):
  """Training configuration.

  See `tf.keras.Model.compile` and `tf.keras.Model.fit` for more information
  about these attributes.

  Attributes:
    optimizer: A `str` or `ObjectConfig` defining a
      `tf.keras.optimizers.Optimizer`.
    loss: A `str` or `ObjectConfig` defining a `tf.keras.losses.Loss` or a list
      thereof.
    metrics: A list of `str` or `ObjectConfig` defining a list of
      `tf.keras.metrics.Metric`.
    loss_weights: A list of `float` scalars to weight the different loss
      functions.
    weighted_metrics: A list of `str` or `ObjectConfig` defining a list of
      `tf.keras.metrics.Metric`.
    run_eagerly: A `bool`. If true, run the model eagerly, without creating a
      graph.
    steps_per_execution: An `int`. The number of batches to run during each
      graph call.
    epochs: An `int`. The number of epochs to train the model.
    verbose: An `int`. The verbosity mode.
    callbacks: A list of `str` or `ObjectConfig` defining a list of
      `tf.keras.callbacks.Callback`.
    use_default_callbacks: A `bool`. If true, a `ModelCheckpoint` callback and a
      `TensorBoard` callback will be added automatically, without the need to
      specify them explicitly.
  """
  optimizer: ObjectConfig = 'RMSprop'
  loss: List[ObjectConfig] = dataclasses.field(default_factory=list)
  metrics: List[ObjectConfig] = dataclasses.field(default_factory=list)
  loss_weights: List[float] = dataclasses.field(default_factory=list)
  weighted_metrics: List[ObjectConfig] = dataclasses.field(default_factory=list)
  run_eagerly: bool = None
  steps_per_execution: int = None
  epochs: int = 1
  verbose: Union[str, int] = 1
  callbacks: List[ObjectConfig] = dataclasses.field(default_factory=list)
  use_default_callbacks: bool = True


@dataclasses.dataclass
class PredictConfig(hyperparams.Config):
  """Prediction configuration.

  Attributes:
    datasets: A string or list of strings with the datasets to obtain and store
      predictions for. Can include the strings `'train'`, `'val'` and `'test'`.
  """
  datasets: List[str] = 'test'


@dataclasses.dataclass
class TrainModelWorkflowConfig(hyperparams.Config):
  """Train model workflow configuration.

  Attributes:
    experiment: An `ExperimentConfig`. General experiment configuration.
    data: A `DataConfig`. The dataset/s configuration.
    model: A `ModelConfig`. The model configuration.
    training: A `TrainingConfig`. The training configuration.
    predict: A `PredictConfig`. The prediction configuration.
  """
  experiment: ExperimentConfig = ExperimentConfig()
  data: DataConfig = DataConfig()
  model: ModelConfig = ModelConfig()
  training: TrainingConfig = TrainingConfig()
  predict: PredictConfig = PredictConfig()


@dataclasses.dataclass
class TestModelWorkflowConfig(hyperparams.Config):
  """Test model workflow configuration.
  
  Attributes:
    experiment: An `ExperimentConfig`. General experiment configuration.
    data: A `DataConfig`. The dataset/s configuration.
    model: An `ExistingModelConfig`. The model configuration.
    predict: A `PredictConfig`. The prediction configuration.
  """
  experiment: ExperimentConfig = ExperimentConfig()
  data: DataConfig = DataConfig()
  model: ExistingModelConfig = ExistingModelConfig()
  predict: PredictConfig = PredictConfig()
