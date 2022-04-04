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

import copy
import dataclasses
import importlib.util
import pathlib
from typing import List, Union, Optional

import keras_tuner as kt
import tensorflow as tf

from keras_declarative import hyperparams
from keras_declarative import util

# NOTE: This file is more easily read from the bottom up, as the more generic
# configuration elements are at the bottom and become more specific towards the
# top.


@dataclasses.dataclass
class DlexDataSplitConfig(hyperparams.Config):
  """Data split configuration (DLEX).

  Attributes:
    train: The training split. Can be an integer (e.g. 50) to use a fixed
      number of examples, or a percentage (e.g. 50%) to use a fixed percentage
      of the total number of examples.
    val: The validation split. Can be an integer (e.g. 50) to use a fixed
      number of examples, or a percentage (e.g. 50%) to use a fixed percentage
      of the total number of examples.
    test: The test split. Can be an integer (e.g. 50) to use a fixed number of
      examples, or a percentage (e.g. 50%) to use a fixed percentage of the
      total number of examples.
  """
  train: int = 0
  val: int = 0
  test: int = 0
  mode: str = 'random'


@dataclasses.dataclass
class TfdsDataSplitConfig(hyperparams.Config):
  """Data split configuration (TFDS).

  Attributes:
    train: A TFDS split. See https://www.tensorflow.org/datasets/splits.
    val: A TFDS split. See https://www.tensorflow.org/datasets/splits.
    test: A TFDS split. See https://www.tensorflow.org/datasets/splits.
  """
  train: str = None
  val: str = None
  test: str = None


@dataclasses.dataclass
class DlexDataSourceConfig(hyperparams.Config):
  """DLEX data source configuration.

  Attributes:
    path: Path to the directory containing the DLEX files.
    prefix: The prefix of the DLEX files.
    split: The split configuration.
  """
  path: str = None
  prefix: str = None
  split: DlexDataSplitConfig = DlexDataSplitConfig()


@dataclasses.dataclass
class TfdsDataSourceConfig(hyperparams.Config):
  """TFDS data source configuration.

  Attributes:
    name: The name of the TFDS dataset.
    version: The version of the TFDS dataset.
    split: The split configuration.
    data_dir: The TFDS data directory.
  """
  name: str = None
  version: str = None
  split: TfdsDataSplitConfig = TfdsDataSplitConfig()
  data_dir: str = None


@dataclasses.dataclass
class DataSourceConfig(hyperparams.OneOfConfig):
  """Data source configuration.

  Attributes:
    type: The type of data source.
    dlex: The DLEX data source configuration.
    tfds: The TFDS data source configuration.
  """
  type: str = None
  dlex: DlexDataSourceConfig = DlexDataSourceConfig()
  tfds: TfdsDataSourceConfig = TfdsDataSourceConfig()


@dataclasses.dataclass
class TensorSpecConfig(hyperparams.Config):
  """Tensor specification configuration."""
  name: Optional[Union[str, int]] = None
  shape: List[int] = None
  dtype: str = 'float32'


@dataclasses.dataclass
class DataSpecsConfig(hyperparams.Config):
  """Specs configuration."""
  train: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  val: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  test: List[TensorSpecConfig] = dataclasses.field(default_factory=list)


class ObjectConfig(hyperparams.ParamsDict):
  """Object configuration."""


@dataclasses.dataclass
class ApplyTransformConfig(hyperparams.Config):
  """Apply transform configuration."""
  transformation_func: ObjectConfig = ObjectConfig()


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
class FilterTransformConfig(hyperparams.Config):
  """Shuffle transform configuration."""
  predicate: ObjectConfig = ObjectConfig()


@dataclasses.dataclass
class FlatMapTransformConfig(hyperparams.Config):
  """Flat map transform configuration."""
  map_func: ObjectConfig = ObjectConfig()


@dataclasses.dataclass
class MapTransformConfig(hyperparams.Config):
  """Map transform configuration."""
  map_func: ObjectConfig = ObjectConfig()
  num_parallel_calls: Optional[int] = None
  deterministic: Optional[bool] = None
  component: Optional[Union[int, str]] = None
  output: Optional[Union[int, str]] = None


@dataclasses.dataclass
class PrefetchTransformConfig(hyperparams.Config):
  """Prefetch transform configuration."""
  buffer_size: int = None


@dataclasses.dataclass
class RepeatTransformConfig(hyperparams.Config):
  """Repeat transform configuration."""
  count: int = None


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
  apply: ApplyTransformConfig = ApplyTransformConfig()
  batch: BatchTransformConfig = BatchTransformConfig()
  cache: CacheTransformConfig = CacheTransformConfig()
  filter: FilterTransformConfig = FilterTransformConfig()
  flat_map: FlatMapTransformConfig = FlatMapTransformConfig()
  map: MapTransformConfig = MapTransformConfig()
  prefetch: PrefetchTransformConfig = PrefetchTransformConfig()
  repeat: RepeatTransformConfig = RepeatTransformConfig()
  shuffle: ShuffleTransformConfig = ShuffleTransformConfig()


class DataOptionsConfig(hyperparams.Config):
  """Data options configuration."""
  shuffle_training_only: bool = True
  max_intra_op_parallelism: int = None
  private_threadpool_size: int = None


@dataclasses.dataclass
class DataTransformsConfig(hyperparams.Config):
  """Data transforms configuration."""
  train: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long
  val: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long
  test: List[DataTransformConfig] = dataclasses.field(default_factory=list) # pylint: disable=line-too-long


@dataclasses.dataclass
class DataConfig(hyperparams.Config):
  """Data configuration."""
  sources: List[DataSourceConfig] = dataclasses.field(default_factory=list)
  specs: DataSpecsConfig = DataSpecsConfig()
  transforms: DataTransformsConfig = DataTransformsConfig()
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
class AppConfig(hyperparams.Config):
  """App configuration."""
  name: str = None
  config: ObjectConfig = ObjectConfig()
  preprocess_input: bool = True
  decode_predictions: bool = True


@dataclasses.dataclass
class ModelConfig(hyperparams.Config):
  """Model configuration.

  Attributes:
    type: A `str`. The type of model. One of `'app'`, `'layers'` or `'model'`.
    app: An `AppConfig`. The app configuration.
    network: A `str` or `ObjectConfig` defining a `tf.keras.layers.Layer` or a
      list thereof, implementing a sequential network architecture.
    input_spec: A list of `TensorSpecConfig` defining the model input
      specification. If not specified, we will attempt to infer the input
      specification from the training dataset.
    path: A `str`. Path to an existing model. Defaults to `None`. If not `None`,
      loads this model ignoring the remaining arguments.
    weights: A `str`. Path to model weights.
  """
  type: str = None
  app: AppConfig = AppConfig()
  network: List[ObjectConfig] = ObjectConfig()
  input_spec: List[TensorSpecConfig] = dataclasses.field(default_factory=list)
  path: str = None
  weights: str = None


@dataclasses.dataclass
class TrainingConfig(hyperparams.Config):
  """Training configuration.

  See `tf.keras.Model.compile` and `tf.keras.Model.fit` for more information
  about these attributes.

  Attributes:
    optimizer: A `str` or `ObjectConfig` defining a
      `tf.keras.optimizers.Optimizer`.
    loss: A nested structure of `str` or `ObjectConfig` defining one or more
      `tf.keras.losses.Loss`.
    metrics: A nested structure of `str` or `ObjectConfig` defining a list of
      `tf.keras.metrics.Metric`.
    loss_weights: A list or dict of `float` scalars to weight the different loss
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
  optimizer: ObjectConfig = ObjectConfig()
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
    evaluate: A `bool`. Whether to evaluate the model using the specified
      datasets.
    save_images: A `bool`. If true, saves processed images of the predictions.
      3D images are saved as GIF files.
  """
  datasets: List[str] = 'test'
  evaluate: bool = True
  save_images: bool = False


@dataclasses.dataclass
class TuningConfig(hyperparams.Config):
  """Tuning configuration.

  Attributes:
    tuner: An `ObjectConfig` definining the tuner configuration. For a list of
      valid tuners and their configurations, see
      https://keras.io/api/keras_tuner/tuners/.
  """
  tuner: ObjectConfig = ObjectConfig()


@dataclasses.dataclass
class DistributeConfig(hyperparams.Config):
  """Distribute configuration.

  Attribute:
    strategy: An `ObjectConfig` defining the distribute strategy configuration.
  """
  strategy: ObjectConfig = ObjectConfig()


@dataclasses.dataclass
class TrainWorkflowConfig(hyperparams.Config):
  """Train model workflow configuration.

  Attributes:
    experiment: An `ExperimentConfig`. General experiment configuration.
    data: A `DataConfig`. The dataset/s configuration.
    model: A `ModelConfig`. The model configuration.
    training: A `TrainingConfig`. The training configuration.
    predict: A `PredictConfig`. The prediction configuration.
    tuning: A `TuningConfig`. The tuning configuration.
    distribute: A `DistributeConfig`. The distribute configuration.
  """
  experiment: ExperimentConfig = ExperimentConfig()
  data: DataConfig = DataConfig()
  model: ModelConfig = ModelConfig()
  training: TrainingConfig = TrainingConfig()
  predict: PredictConfig = PredictConfig()
  tuning: TuningConfig = TuningConfig()
  distribute: DistributeConfig = DistributeConfig()


@dataclasses.dataclass
class TestWorkflowConfig(hyperparams.Config):
  """Test model workflow configuration.

  Attributes:
    experiment: An `ExperimentConfig`. General experiment configuration.
    data: A `DataConfig`. The dataset/s configuration.
    model: A `ModelConfig`. The model configuration.
    predict: A `PredictConfig`. The prediction configuration.
    distribute: A `DistributeConfig`. The distribute configuration.
  """
  experiment: ExperimentConfig = ExperimentConfig()
  data: DataConfig = DataConfig()
  model: ModelConfig = ModelConfig()
  predict: PredictConfig = PredictConfig()
  distribute: DistributeConfig = DistributeConfig()


def deserialize_special_objects(params):
  """Deserialize special objects.

  Special objects include random numbers, random number generators, tunable
  hyperparameters and external module objects.

  Note that the output of this function can no longer be safely serialized and
  should not be written to YAML.

  Args:
    params: A `hyperparams.Config`.

  Returns:
    A non-serializable `hyperparams.Config`.
  """
  for k, v in params.__dict__.items():

    if is_special_config(v):
      params.__dict__[k] = _parse_special_config(v)

    elif isinstance(v, hyperparams.ParamsDict):
      params.__dict__[k] = deserialize_special_objects(v)

    elif isinstance(v, hyperparams.Config.SEQUENCE_TYPES):
      for i, e in enumerate(v):

        if is_special_config(e):
          params.__dict__[k][i] = _parse_special_config(e)

        if isinstance(e, hyperparams.ParamsDict):
          params.__dict__[k][i] = deserialize_special_objects(e)

  return params


def _parse_special_config(config):
  """Parse a special object configuration.

  Args:
    config: A `hyperparams.ParamsDict` defining a valid special configuration.

  Returns:
    The corresponding special object.

  Raises:
    ValueError: If `config` is not a valid special configuration.
  """
  if not is_special_config(config):
    raise ValueError(f"Not a valid special configuration: {config}")

  d = config.as_dict() if isinstance(config, hyperparams.ParamsDict) else config
  obj_type, obj_config = next(iter(d.items()))
  obj_type = obj_type[1:]

  if obj_type == 'rng':
    return _get_rng(obj_config)

  if obj_type == 'random':
    return _get_rng(obj_config)()

  if obj_type == 'tunable':
    return _get_tunable(obj_config)

  if obj_type == 'external':
    return _get_external(obj_config)

  raise ValueError(f"Unknown special object type: {obj_type}")


def _get_rng(config):
  """Get a random number generator from the given config.

  Args:
    config: An RNG config.

  Returns:
    A callable with no arguments which returns random numbers according to the
    specified configuration.

  Raises:
    ValueError: If `config` is not a valid RNG config.
  """
  if 'type' not in config:
    raise ValueError(f"Invalid RNG config: {config}")
  rng_type = config['type']

  if rng_type not in config:
    rng_kwargs = {}
  else:
    rng_kwargs = config[rng_type]

  rng = tf.random.get_global_generator()

  rng_func = {
      'binomial': rng.binomial,
      'normal': rng.normal,
      'truncated_normal': rng.truncated_normal,
      'uniform': rng.uniform
  }

  if rng_type not in rng_func:
    raise ValueError(f"Unknown RNG type: {rng_type}")

  return lambda: rng_func[rng_type](**rng_kwargs) # pylint: disable=unnecessary-lambda


def _get_tunable(config):
  """Get a tunable placeholder from the given config.

  Args:
    config: A tunable config dictionary.

  Returns:
    A `TunablePlaceholder`.

  Raises:
    ValueError: If `config` is not a valid tunable config.
  """
  if 'type' not in config:
    raise ValueError(f"Invalid hyperparameter config: {config}")
  tunable_type = config['type']

  if tunable_type not in config:
    tunable_kwargs = {}
  else:
    tunable_kwargs = config[tunable_type]

  types_dict = {
      'boolean': 'Boolean',
      'choice': 'Choice',
      'fixed': 'Fixed',
      'float': 'Float',
      'int': 'Int'
  }

  if tunable_type not in types_dict:
    raise ValueError(f"Unknown hyperparameter type: {tunable_type}")

  return util.TunablePlaceholder(types_dict[tunable_type], tunable_kwargs)


def _get_external(config):
  """Get an external object from the given config.

  Args:
    config: An external config dictionary.

  Returns:
    An `ExternalObject`.

  Raises:
    ValueError: If `config` is not a valid external config.
  """
  if 'filename' not in config:
    raise ValueError(f"Invalid module config: {config}. Missing filename.")
  if 'object_name' not in config:
    raise ValueError(f"Invalid module config: {config}. Missing object_name.")

  # Load the specified module.
  path = pathlib.Path(config['filename'])
  spec = importlib.util.spec_from_file_location(path.stem, str(path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  # Retrieve the specified object from module.
  obj = getattr(module, config['object_name'])

  args = config.get('args') or []
  kwargs = config.get('kwargs') or {}
  kwargs = deserialize_special_objects(hyperparams.ParamsDict(kwargs)).as_dict()

  # Initialize object.
  return util.ExternalObject(obj, args, kwargs, filename=path)


def is_special_config(config):
  """Check if input is a valid special config.

  Args:
    config: A `hyperparams.ParamsDict`.

  Returns:
    True if input is a valid special config, false otherwise.
  """
  # Must by an object of type `ParamsDict`.
  if not isinstance(config, (dict, hyperparams.ParamsDict)):
    return False

  # Must have one key.
  d = config.as_dict() if isinstance(config, hyperparams.ParamsDict) else config
  if not len(d) == 1:
    return False

  # Key must be a string starting with dollar sign $.
  k = next(iter(d)) # First key in dict.
  if not isinstance(k, str) or not k.startswith('$'):
    return False

  return True


def find_hyperparameters(params, hp=None):
  """Find all tunable hyperparameters in config.

  Args:
    params: A `hyperparams.Config`.
    hp: A `kt.HyperParameters` object.

  Returns:
    A non-serializable `hyperparams.Config`.
  """
  if hp is None:
    hp = kt.HyperParameters()

  for v in params.__dict__.values():

    if isinstance(v, util.TunablePlaceholder):
      _ = v(hp)

    elif isinstance(v, hyperparams.ParamsDict):
      hp = find_hyperparameters(v, hp=hp)

    elif isinstance(v, util.ExternalObject):
      hp = find_hyperparameters(hyperparams.ParamsDict(v._kwargs), hp=hp)  # pylint: disable=protected-access

    elif isinstance(v, hyperparams.Config.SEQUENCE_TYPES):
      for e in v:

        if isinstance(e, util.TunablePlaceholder):
          _ = e(hp)

        if isinstance(e, hyperparams.ParamsDict):
          hp = find_hyperparameters(e, hp=hp)

  return hp


def find_external_objects(params, ext_objects=None):
  """Find all external objects in config.

  Args:
    params: A `hyperparams.Config`.
    ext_objects: A list of `util.ExternalObject`.

  Returns:
    A list of `util.ExternalObject`.
  """
  ext_objects = []

  def _find_external_objects(p):

    for v in p.__dict__.values():
      if isinstance(v, util.ExternalObject):
        ext_objects.append(v)

      elif isinstance(v, hyperparams.ParamsDict):
        _find_external_objects(v)

      elif isinstance(v, hyperparams.Config.SEQUENCE_TYPES):
        for e in v:
          if isinstance(e, util.ExternalObject):
            ext_objects.append(v)

          if isinstance(e, hyperparams.ParamsDict):
            _find_external_objects(e)

  _find_external_objects(params)
  return ext_objects


def inject_hyperparameters(params, hp):
  """Injects current hyperparameters into params dict.

  Args:
    params: A `hyperparams.Config`, potentially containing hyperparameter
      placeholders.
    hp: A `kt.HyperParameters` object.

  Returns:
    A `hyperparams.Config` with the injected hyperparameter values.
  """
  params = copy.deepcopy(params)

  for k, v in params.__dict__.items():

    if isinstance(v, util.TunablePlaceholder):
      params.__dict__[k] = v(hp)

    elif isinstance(v, hyperparams.ParamsDict):
      params.__dict__[k] = inject_hyperparameters(v, hp)

    elif isinstance(v, util.ExternalObject):
      params.__dict__[k]._kwargs = inject_hyperparameters(  # pylint: disable=protected-access
          hyperparams.ParamsDict(v._kwargs), hp).as_dict()  # pylint: disable=protected-access

    elif isinstance(v, hyperparams.Config.SEQUENCE_TYPES):
      for i, e in enumerate(v):

        if isinstance(v, util.TunablePlaceholder):
          params.__dict__[k][i] = e(hp)

        if isinstance(e, hyperparams.ParamsDict):
          params.__dict__[k][i] = inject_hyperparameters(e, hp)

  return params
