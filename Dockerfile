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

FROM tensorflow/tensorflow:2.6.0-gpu

ARG TF_VERSION=2.6.0
ARG TFMR_VERSION=0.7.0
ARG TFFT_VERSION=0.3.2

# Note: this will stop being necessary once TensorFlow images start using with
# Ubuntu 20.04, which ships with Python 3.8 by default. 
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.8 python3.8-dev
RUN python3.8 -m pip install --upgrade pip

RUN apt-get install -y libopenexr-dev

RUN python3.8 -m pip install tensorflow==${TF_VERSION}
RUN python3.8 -m pip install tensorflow-mri==${TFMR_VERSION}
RUN python3.8 -m pip install tensorflow-nufft==${TFFT_VERSION}
