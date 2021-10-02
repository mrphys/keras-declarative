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

import yaml

from tensorflow_declarative import objects


def new_model(config_file):
  # Load configuration.
  with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

  config['compile']['loss'] = _parse_loss(config['compile']['loss'])

  
  print(config)

  # model.compile(**config['compile'])
  # model.fit(**config['fit'])


def _parse_loss(identifiers):
  if isinstance(identifiers, list):
    return [objects.get_loss(identifier) for identifier in identifiers]
  return objects.get_loss(identifiers)
