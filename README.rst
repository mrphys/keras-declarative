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

  train_transforms: &transforms
    # Define the list of transforms here.

  val_transforms: *transforms

Hyperparameter Tuning
---------------------

Most parameters can be set as tunable using the ``$tunable`` directive.

.. code-block:: yaml

  filters:
    $tunable:
      type: int
      int:
        name: filters
        min_value: 16
        max_value: 64
        step: 16

Valid types are `boolean`, `choice`, `fixed`, `float` and `int`. For a list of
valid options, see https://keras.io/api/keras_tuner/hyperparameters/.

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
