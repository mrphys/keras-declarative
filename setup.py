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
"""Setup Keras Declarative."""

from os import path
from setuptools import find_packages
from setuptools import setup

ROOT = path.abspath(path.dirname(__file__))

ABOUT = {}
with open(path.join(ROOT, "keras_declarative/__about__.py")) as f:
    exec(f.read(), ABOUT)

with open(path.join(ROOT, "README.rst"), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

with open(path.join(ROOT, "requirements.txt")) as f:
    REQUIRED_PACKAGES = [line.strip() for line in f.readlines()]

setup(
    name=ABOUT['__title__'],
    version=ABOUT['__version__'],
    description=ABOUT['__summary__'],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    author=ABOUT['__author__'],
    author_email=ABOUT['__email__'],
    url=ABOUT['__uri__'],
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'keras.train = keras_declarative.cli:train',
            'keras.test = keras_declarative.cli:test'
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license=ABOUT['__license__'],
    keywords=['tensorflow', 'keras', 'machine learning', 'ml']   
)
