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



@dataclasses.dataclass
class DataSplitConfig(hyperparams.Config):
  """Data split configuration."""
  train: float = None
  val: float = None
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
  train_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list)
  val_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list)
  test_transforms: List[DataTransformConfig] = dataclasses.field(default_factory=list)
  options: DataOptionsConfig = DataOptionsConfig()


@dataclasses.dataclass
class ExperimentConfig(hyperparams.Config):
  """Experiment configuration."""
  name: str = None
  path: str = None
  seed: Optional[int] = None


@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
  """Model configuration."""
  network: ObjectConfig = ObjectConfig()
  input_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TrainingConfig(hyperparams.Config):
  """Training configuration."""
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
  """Prediction configuration."""
  datasets: List[str] = 'test'


@dataclasses.dataclass
class TrainModelWorkflowConfig(hyperparams.Config):
  """Train model workflow configuration."""
  experiment: ExperimentConfig = ExperimentConfig()
  data: DataConfig = DataConfig()
  model: ModelConfig = ModelConfig()
  training: TrainingConfig = TrainingConfig()
  predict: PredictConfig = PredictConfig()
