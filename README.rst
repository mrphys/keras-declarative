Keras Declarative
=================

  This package is in alpha stage.
  This README is a work in progress.

Object Discovery
----------------

Keras Declarative automatically discovers objects available in some libraries,
including core TensorFlow and TensorFlow MRI.

Anchors and Aliases
-------------------

If you need to repeat a node more than once, you can anchor it once with the
``&`` character and then alias it any number of times using the ``*`` character.
For example, to use the same list of data transforms for the training and
validation sets, you may use:

.. code-block:: yaml

  data:
    train_transforms: &transforms
      # Define the list of transforms here.

    val_transforms: *transforms

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

For a list of available tuners and their options, see
https://keras.io/api/keras_tuner/tuners/.

Note that some parameters cannot be tuned. These include all parameters
under `experiment` and under `data.sources`.
