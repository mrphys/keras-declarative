Keras Declarative
=================

  This package is in alpha stage.
  This README is a work in progress.


Creating Experiments
--------------------

There are two types of experiments: **training** experiments, which involve
training a new or existing model, and **testing** experiments, which involve
testing an existing model.

.. code-block:: console

  keras.train --config_file path/to/config/file.yaml

  keras.test --config_file path/to/config/file.yaml


Specifying Keras Objects
------------------------

Keras Declarative automatically discovers objects, such as layers and loss
functions, which are available in core TensorFlow, TensorFlow MRI and TF
Playground. This means these objects can be used in the configuration file
without explicit registration.

Serializable Keras objects, such as layers and loss functions, are specified by
a class name and a configuration dictionary. If no parameters should be passed
to the object, the configuration dictionary may be omitted.

.. code-block:: yaml

  # class name and parameters
  training:
    optimizer:
      class_name: Adam
      config:
        learning_rate: 0.001

  # class name only (instantiated with default parameters)
  training:
    optimizer: Adam


External Modules
----------------

It is possible to use objects defined in external modules within Keras
Declarative, i.e., objects that are not part of either core TensorFlow,
TensorFlow MRI or TF Playground. This is particularly useful to define
preprocessing functions but can be used for any other purpose. External modules
can be specified with the ``$external`` directive.

Any external modules used during an experiment will be automatically saved to
the experiment folder to enable reproducibility.

.. code-block:: yaml

  data:
    transforms:
      train:
        - type: map
          map:
            map_func:
              $external:
                # Use a preprocessing function defined in an external module.
                filename: /path/to/preprocessing_fn.py
                object_name: preprocessing_fn
                # Any parameters that should be passed to this object may be
                # specified here.
                args: null
                kwargs: null


Anchors and Aliases
-------------------

If you need to repeat a node more than once, you can anchor it once with the
``&`` character and then alias it any number of times using the ``*`` character.
For example, to use the same list of data transforms for the training and
validation sets, you may use:

.. code-block:: yaml

  data:
    transforms:
      train: &transforms
        # Define the list of transforms here.

      val: *transforms  # Reusing training transforms here.


Hyperparameter Tuning
---------------------

Keras Declarative can configure the Keras Tuner to automatically tune one or
more parameters.

Most parameters can be set as tunable using the ``$tunable`` directive. For
example, to tune the kernel size of a U-Net model, you might use:

.. code-block:: yaml

  model:
    network:
      class_name: UNet
      config:
        scales: 3
        base_filters: 32
        kernel_size:
          $tunable:
            type: int
            int:
              name: kernel_size
              min_value: 3
              max_value: 7
              step: 2

Valid tunable types are `boolean`, `choice`, `fixed`, `float` and `int`. For
more details, see https://keras.io/api/keras_tuner/hyperparameters/.

The tuner type and options can be specified with ``tuning.tuner`` parameter:

.. code-block:: yaml

  tuning:
    tuner:
      class_name: Hyperband
      config:
        objective:
          name: val_ssim
          direction: max
        max_epochs: 100

Available tuners are ``RandomSearch``, ``BayesianOptimization`` and
``Hyperband``. For more details about these tuners and their options, see
https://keras.io/api/keras_tuner/tuners/.

Note that some parameters cannot be tuned. These include all parameters
under ``experiment`` and under ``data.sources``.
