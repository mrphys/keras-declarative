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
"""Command line interface (CLI) services."""

import sys

from absl import app
from absl import flags

from keras_declarative import workflows


FLAGS = flags.FLAGS


def define_flags():

  flags.DEFINE_multi_string(
      name='config_file',
      default=None,
      help="YAML/JSON files specifying overrides. The override order follows "
           "the order of args. Note that each file can be used as an override "
           "template to override the default parameters specified in Python. "
           "If the same parameter is specified in both `--config_file` and "
           "`--params_override`, `config_file` will be used first, followed by "
           "params_override.",
      required=True)


def train():
  """Create and train a new model (CLI)."""
  define_flags()
  app.run(_train)


def test():
  """Test an existing model (CLI)."""
  define_flags()
  app.run(_test)


def _train(argv): # pylint: disable=unused-argument
  """Create and train a new model (CLI, internal)."""
  workflows.train(FLAGS.config_file)
  sys.exit(0)


def _test(argv): # pylint: disable=unused-argument
  """Test an existing model (CLI, internal)."""
  workflows.test(FLAGS.config_file)
  sys.exit(0)
