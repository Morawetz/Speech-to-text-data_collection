Module dags.src.models.initial.new_model
========================================

Functions
---------

    
`my_model(encoder, max_length)`
:   

Classes
-------

`CTCLayer(name=None, **kwargs)`
:   This is the class from which all layers inherit.
    
    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables), defined
    either in the constructor `__init__()` or in the `build()` method.
    
    Users will just instantiate a layer and then treat it as a callable.
    
    Args:
      trainable: Boolean, whether the layer's variables should be trainable.
      name: String name of the layer.
      dtype: The dtype of the layer's computations and weights. Can also be a
        `tf.keras.mixed_precision.Policy`, which allows the computation and weight
        dtype to differ. Default of `None` means to use
        `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
        unless set to different value.
      dynamic: Set this to `True` if your layer should only be run eagerly, and
        should not be used to generate a static computation graph.
        This would be the case for a Tree-RNN or a recursive network,
        for example, or generally for any layer that manipulates tensors
        using Python control flow. If `False`, we assume that the layer can
        safely be used to generate a static computation graph.
    
    Attributes:
      name: The name of the layer (string).
      dtype: The dtype of the layer's weights.
      variable_dtype: Alias of `dtype`.
      compute_dtype: The dtype of the layer's computations. Layers automatically
        cast inputs to this dtype which causes the computations and output to also
        be in this dtype. When mixed precision is used with a
        `tf.keras.mixed_precision.Policy`, this will be different than
        `variable_dtype`.
      dtype_policy: The layer's dtype policy. See the
        `tf.keras.mixed_precision.Policy` documentation for details.
      trainable_weights: List of variables to be included in backprop.
      non_trainable_weights: List of variables that should not be
        included in backprop.
      weights: The concatenation of the lists trainable_weights and
        non_trainable_weights (in this order).
      trainable: Whether the layer should be trained (boolean), i.e. whether
        its potentially-trainable weights should be returned as part of
        `layer.trainable_weights`.
      input_spec: Optional (list of) `InputSpec` object(s) specifying the
        constraints on inputs that can be accepted by the layer.
    
    We recommend that descendants of `Layer` implement the following methods:
    
    * `__init__()`: Defines custom layer attributes, and creates layer state
      variables that do not depend on input shapes, using `add_weight()`.
    * `build(self, input_shape)`: This method can be used to create weights that
      depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
      will automatically build the layer (if it has not been built yet) by
      calling `build()`.
    * `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
      sure `build()` has been called. `call()` performs the logic of applying the
      layer to the input tensors (which should be passed in as argument).
      Two reserved keyword arguments you can optionally use in `call()` are:
        - `training` (boolean, whether the call is in inference mode or training
          mode). See more details in [the layer/model subclassing guide](
          https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
        - `mask` (boolean tensor encoding masked timesteps in the input, used
          in RNN layers). See more details in [the layer/model subclassing guide](
          https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
      A typical signature for this method is `call(self, inputs)`, and user could
      optionally add `training` and `mask` if the layer need them. `*args` and
      `**kwargs` is only useful for future extension when more input parameters
      are planned to be added.
    * `get_config(self)`: Returns a dictionary containing the configuration used
      to initialize this layer. If the keys differ from the arguments
      in `__init__`, then override `from_config(self)` as well.
      This method is used when saving
      the layer or a model that contains this layer.
    
    Examples:
    
    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).
    
    ```python
    class SimpleDense(Layer):
    
      def __init__(self, units=32):
          super(SimpleDense, self).__init__()
          self.units = units
    
      def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
    
      def call(self, inputs):  # Defines the computation from inputs to outputs
          return tf.matmul(inputs, self.w) + self.b
    
    # Instantiates the layer.
    linear_layer = SimpleDense(4)
    
    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(tf.ones((2, 2)))
    assert len(linear_layer.weights) == 2
    
    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```
    
    Note that the method `add_weight()` offers a shortcut to create weights:
    
    ```python
    class SimpleDense(Layer):
    
      def __init__(self, units=32):
          super(SimpleDense, self).__init__()
          self.units = units
    
      def build(self, input_shape):
          self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True)
          self.b = self.add_weight(shape=(self.units,),
                                   initializer='random_normal',
                                   trainable=True)
    
      def call(self, inputs):
          return tf.matmul(inputs, self.w) + self.b
    ```
    
    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:
    
    ```python
    class ComputeSum(Layer):
    
      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                   trainable=False)
    
      def call(self, inputs):
          self.total.assign_add(tf.reduce_sum(inputs, axis=0))
          return self.total
    
    my_sum = ComputeSum(2)
    x = tf.ones((2, 2))
    
    y = my_sum(x)
    print(y.numpy())  # [2. 2.]
    
    y = my_sum(x)
    print(y.numpy())  # [4. 4.]
    
    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```
    
    For more information about creating layers, see the guide
    [Making new Layers and Models via subclassing](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models)

    ### Ancestors (in MRO)

    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector

    ### Methods

    `call(self, y_true, y_pred)`
    :   This is where the layer's logic lives.
        
        Note here that `call()` method in `tf.keras` is little bit different
        from `keras` API. In `keras` API, you can pass support masking for
        layers as additional arguments. Whereas `tf.keras` has `compute_mask()`
        method to support masking.
        
        Args:
          inputs: Input tensor, or dict/list/tuple of input tensors.
            The first positional `inputs` argument is subject to special rules:
            - `inputs` must be explicitly passed. A layer cannot have zero
              arguments, and `inputs` cannot be provided via the default value
              of a keyword argument.
            - NumPy array or Python scalar values in `inputs` get cast as tensors.
            - Keras mask metadata is only collected from `inputs`.
            - Layers are built (`build(input_shape)` method)
              using shape info from `inputs` only.
            - `input_spec` compatibility is only checked against `inputs`.
            - Mixed precision input casting is only applied to `inputs`.
              If a layer has tensor arguments in `*args` or `**kwargs`, their
              casting behavior in mixed precision should be handled manually.
            - The SavedModel input specification is generated using `inputs` only.
            - Integration with various ecosystem packages like TFMOT, TFLite,
              TF.js, etc is only supported for `inputs` and not for tensors in
              positional and keyword arguments.
          *args: Additional positional arguments. May contain tensors, although
            this is not recommended, for the reasons above.
          **kwargs: Additional keyword arguments. May contain tensors, although
            this is not recommended, for the reasons above.
            The following optional keyword arguments are reserved:
            - `training`: Boolean scalar tensor of Python boolean indicating
              whether the `call` is meant for training or inference.
            - `mask`: Boolean input mask. If the layer's `call()` method takes a
              `mask` argument, its default value will be set to the mask generated
              for `inputs` by the previous layer (if `input` did come from a layer
              that generated a corresponding mask, i.e. if it came from a Keras
              layer with masking support).
        
        Returns:
          A tensor or list/tuple of tensors.

`LogMelgramLayer(num_fft, hop_length, **kwargs)`
:   This is the class from which all layers inherit.
    
    A layer is a callable object that takes as input one or more tensors and
    that outputs one or more tensors. It involves *computation*, defined
    in the `call()` method, and a *state* (weight variables), defined
    either in the constructor `__init__()` or in the `build()` method.
    
    Users will just instantiate a layer and then treat it as a callable.
    
    Args:
      trainable: Boolean, whether the layer's variables should be trainable.
      name: String name of the layer.
      dtype: The dtype of the layer's computations and weights. Can also be a
        `tf.keras.mixed_precision.Policy`, which allows the computation and weight
        dtype to differ. Default of `None` means to use
        `tf.keras.mixed_precision.global_policy()`, which is a float32 policy
        unless set to different value.
      dynamic: Set this to `True` if your layer should only be run eagerly, and
        should not be used to generate a static computation graph.
        This would be the case for a Tree-RNN or a recursive network,
        for example, or generally for any layer that manipulates tensors
        using Python control flow. If `False`, we assume that the layer can
        safely be used to generate a static computation graph.
    
    Attributes:
      name: The name of the layer (string).
      dtype: The dtype of the layer's weights.
      variable_dtype: Alias of `dtype`.
      compute_dtype: The dtype of the layer's computations. Layers automatically
        cast inputs to this dtype which causes the computations and output to also
        be in this dtype. When mixed precision is used with a
        `tf.keras.mixed_precision.Policy`, this will be different than
        `variable_dtype`.
      dtype_policy: The layer's dtype policy. See the
        `tf.keras.mixed_precision.Policy` documentation for details.
      trainable_weights: List of variables to be included in backprop.
      non_trainable_weights: List of variables that should not be
        included in backprop.
      weights: The concatenation of the lists trainable_weights and
        non_trainable_weights (in this order).
      trainable: Whether the layer should be trained (boolean), i.e. whether
        its potentially-trainable weights should be returned as part of
        `layer.trainable_weights`.
      input_spec: Optional (list of) `InputSpec` object(s) specifying the
        constraints on inputs that can be accepted by the layer.
    
    We recommend that descendants of `Layer` implement the following methods:
    
    * `__init__()`: Defines custom layer attributes, and creates layer state
      variables that do not depend on input shapes, using `add_weight()`.
    * `build(self, input_shape)`: This method can be used to create weights that
      depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
      will automatically build the layer (if it has not been built yet) by
      calling `build()`.
    * `call(self, inputs, *args, **kwargs)`: Called in `__call__` after making
      sure `build()` has been called. `call()` performs the logic of applying the
      layer to the input tensors (which should be passed in as argument).
      Two reserved keyword arguments you can optionally use in `call()` are:
        - `training` (boolean, whether the call is in inference mode or training
          mode). See more details in [the layer/model subclassing guide](
          https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_training_argument_in_the_call_method)
        - `mask` (boolean tensor encoding masked timesteps in the input, used
          in RNN layers). See more details in [the layer/model subclassing guide](
          https://www.tensorflow.org/guide/keras/custom_layers_and_models#privileged_mask_argument_in_the_call_method)
      A typical signature for this method is `call(self, inputs)`, and user could
      optionally add `training` and `mask` if the layer need them. `*args` and
      `**kwargs` is only useful for future extension when more input parameters
      are planned to be added.
    * `get_config(self)`: Returns a dictionary containing the configuration used
      to initialize this layer. If the keys differ from the arguments
      in `__init__`, then override `from_config(self)` as well.
      This method is used when saving
      the layer or a model that contains this layer.
    
    Examples:
    
    Here's a basic example: a layer with two variables, `w` and `b`,
    that returns `y = w . x + b`.
    It shows how to implement `build()` and `call()`.
    Variables set as attributes of a layer are tracked as weights
    of the layers (in `layer.weights`).
    
    ```python
    class SimpleDense(Layer):
    
      def __init__(self, units=32):
          super(SimpleDense, self).__init__()
          self.units = units
    
      def build(self, input_shape):  # Create the state of the layer (weights)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)
    
      def call(self, inputs):  # Defines the computation from inputs to outputs
          return tf.matmul(inputs, self.w) + self.b
    
    # Instantiates the layer.
    linear_layer = SimpleDense(4)
    
    # This will also call `build(input_shape)` and create the weights.
    y = linear_layer(tf.ones((2, 2)))
    assert len(linear_layer.weights) == 2
    
    # These weights are trainable, so they're listed in `trainable_weights`:
    assert len(linear_layer.trainable_weights) == 2
    ```
    
    Note that the method `add_weight()` offers a shortcut to create weights:
    
    ```python
    class SimpleDense(Layer):
    
      def __init__(self, units=32):
          super(SimpleDense, self).__init__()
          self.units = units
    
      def build(self, input_shape):
          self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                   initializer='random_normal',
                                   trainable=True)
          self.b = self.add_weight(shape=(self.units,),
                                   initializer='random_normal',
                                   trainable=True)
    
      def call(self, inputs):
          return tf.matmul(inputs, self.w) + self.b
    ```
    
    Besides trainable weights, updated via backpropagation during training,
    layers can also have non-trainable weights. These weights are meant to
    be updated manually during `call()`. Here's a example layer that computes
    the running sum of its inputs:
    
    ```python
    class ComputeSum(Layer):
    
      def __init__(self, input_dim):
          super(ComputeSum, self).__init__()
          # Create a non-trainable weight.
          self.total = tf.Variable(initial_value=tf.zeros((input_dim,)),
                                   trainable=False)
    
      def call(self, inputs):
          self.total.assign_add(tf.reduce_sum(inputs, axis=0))
          return self.total
    
    my_sum = ComputeSum(2)
    x = tf.ones((2, 2))
    
    y = my_sum(x)
    print(y.numpy())  # [2. 2.]
    
    y = my_sum(x)
    print(y.numpy())  # [4. 4.]
    
    assert my_sum.weights == [my_sum.total]
    assert my_sum.non_trainable_weights == [my_sum.total]
    assert my_sum.trainable_weights == []
    ```
    
    For more information about creating layers, see the guide
    [Making new Layers and Models via subclassing](
      https://www.tensorflow.org/guide/keras/custom_layers_and_models)

    ### Ancestors (in MRO)

    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.training.tracking.tracking.AutoTrackable
    * tensorflow.python.training.tracking.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector

    ### Methods

    `build(self, input_shape)`
    :   Creates the variables of the layer (optional, for subclass implementers).
        
        This is a method that implementers of subclasses of `Layer` or `Model`
        can override if they need a state-creation step in-between
        layer instantiation and layer call.
        
        This is typically used to create the weights of `Layer` subclasses.
        
        Args:
          input_shape: Instance of `TensorShape`, or list of instances of
            `TensorShape` if the layer expects a list of inputs
            (one instance per input).

    `call(self, input)`
    :   Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)

    `get_config(self)`
    :   Returns the config of the layer.
        
        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.
        
        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).
        
        Note that `get_config()` does not guarantee to return a fresh copy of dict
        every time it is called. The callers should make a copy of the returned dict
        if they want to modify it.
        
        Returns:
            Python dictionary.