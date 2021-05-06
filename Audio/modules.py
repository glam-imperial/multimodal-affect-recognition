def conv1d(inputs, 
           filters=None, 
           size=1, 
           rate=1, 
           padding="SAME", 
           use_bias=False,
           activation_fn=None,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    
    with tf.variable_scope(scope):
        if padding.lower()=="causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list[-1]

        params = {"inputs":inputs, "filters":filters, "kernel_size":size,
                "dilation_rate":rate, "padding":padding, "activation":activation_fn, 
                "use_bias":use_bias, "reuse":reuse}

        outputs = tf.layers.conv1d(**params)
    return outputs


def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None,
                num_filters=64):
    '''Applies a series of conv1d separately.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is, 
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.
    
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    
    with tf.variable_scope(scope, reuse=reuse):
        outputs = conv1d(inputs, num_filters, 1) 
        outputs = tf.layers.dropout(outputs)
        outputs = normalize(outputs, type="bn", is_training=is_training, 
                            activation_fn=tf.nn.relu)
        
        for k in range(2, K+1): # k = 2...K
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs, num_filters, k, 1)
                outputs = tf.layers.dropout(outputs)
                
                output = normalize(output, type="bn", is_training=is_training, 
                            activation_fn=tf.nn.relu)
                outputs = tf.concat((outputs, output), -1)
        
    return outputs
