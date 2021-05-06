from __future__ import print_function
from __future__ import division


import tensorflow as tf

class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
            One can use attributes to read/write dictionary content.
        '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self



def conv2d(input_, output_dim, k_h, k_w, padding = 'VALID', name = 'conv2d'):
    '''
         :param input_: shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]
         :param output_dim: [kernel_features], which is # of kernels with this width
         :param k_h: 1
         :param k_w: kernel width, n-grams
         :param name: name scope
         :return: shape = [reduced_length, output_dim]
    '''

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim])
        b = tf.get_variable('b', [output_dim])

    return tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding= padding) + b

def linear(input_, output_size, scope = None):
    '''
        input needs to be a 2D matrix, with its shape[1] is valid
        :param intput_: shape = [batch_size * num_unroll_steps, sum(kernel_features)]
        :param output_size: shape = [batch_size * num_unroll_steps, sum(kernel_features)] cause it needs to plus the original input
        :param scope: variable scope
        :return:

    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    :param input_: a tensor or a list of 2D, batch x n, Tensors
    :param output_size: second dim of W[i], output_size
    :param scope: variable scope for the created scope, default linaer!
    :return: A 2D Tensor with shape [batch, output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.

    '''



    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))

    input_size = shape[1]

    with tf.variable_scope(scope or 'SimpleLinear'):

        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype= input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype = input_.dtype)

    return tf.matmul(input_, matrix, transpose_b= True)  + bias_term


def highway(input_, size, num_layers = 1, bias = -2.0, f = tf.nn.relu, scope = 'Highway'):
    '''

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

        :param input_: [batch_size * num_unroll_steps, sum(kernel_features)]
        :param size: this is the output_dim of the kernels, which should be matched for the input_ dim shape[1]
        :param num_layers: how many highway layers you want
        :param bias: transform gate bias
        :param f: linear activation
        :param scope: highway
        :return:
            t = sigmoid(W_Ty + b_T)
            z = t * f(W_Hy + b_H) + (1. - t) * y
    '''


    with tf.variable_scope(scope):

        for idx in range(num_layers):

            g = f(linear(input_, size, scope = 'highway_lin_%d' % idx))

            t = tf.sigmoid(linear(input_, size, scope = 'highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_

            input_ = output

    return output


def tdnn(input_, kernels, kernel_features, training = None, scope = 'TDNN'):
    '''

    :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)

        :param input_: shape = [batch_size * num_unroll_steps, max_sent_length, embed_size]
        :param kernels: n-grams
        :param kernel_features: how many features we want to use for each n-gram kernel
        :param scope: variable scope
        :return:
    '''

    assert len(kernels) == len(kernel_features)

    max_sent_length = input_.get_shape()[1]


    input_ = tf.expand_dims(input_, 1)
    # input_, shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]

    layers = []

    with tf.variable_scope(scope):

        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):

            reduced_length = max_sent_length - kernel_size + 1

            conv = conv2d(input_, kernel_feature_size, 1, kernel_size, name= 'kernel_%d' % kernel_size)
            # conv, shape = [batch_size * num_unroll_steps, 1, reduced_length, kernel_feature_size]


            pool = tf.nn.max_pool(tf.tanh(conv), [1, 1, reduced_length, 1],  [1, 1, 1, 1], 'VALID')
            # pool, shape = [batch_size * num_unroll_steps, 1, 1, kernel_feature_size]

            layers.append(tf.squeeze(pool, [1, 2]))

    if len(kernels) > 1:
        output = tf.concat(layers, 1) # shape = [batch_size * num_unroll_steps, sum(kernel_features)]
    else:
        output = layers[0]

    output = tf.layers.dropout(output, rate = 0.3, training = training)

    return output

def advanced_tdnn(input_, dropout, kernels, kernel_features, scope = 'Advanced_TDNN'):  #, dropout, trnn_size, num_trnn_layers, batch_size, num_unroll_steps, training = None
    '''

        :input:           input float tensor of shape [(batch_size*num_unroll_steps) x max_word_length x embed_size]
        :kernels:         array of kernel sizes
        :kernel_features: array of kernel feature sizes (parallel to kernels)

            :param input_: shape = [batch_size * num_unroll_steps, max_sent_length, embed_size]
            :param kernels: n-grams
            :param kernel_features: how many features we want to use for each n-gram kernel
            :param scope: variable scope
            :return:
        '''

    assert len(kernels) == len(kernel_features)

    layers = []

    max_sent_length = input_.get_shape()[1]

    layers.append(input_)

    input_ = tf.expand_dims(input_, 1)
    # input_, shape = [batch_size * num_unroll_steps, 1, max_sent_length, embed_size]


    with tf.variable_scope(scope):

        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):
            reduced_length = max_sent_length - kernel_size + 1

            conv = tf.nn.tanh(conv2d(input_, kernel_feature_size, 1, kernel_size, 'SAME', name='kernel_%d' % kernel_size))
            # conv, shape = [batch_size * num_unroll_steps, 1, reduced_length, kernel_feature_size]
            conv = tf.squeeze(conv, [1])
            #attention, outputs = trnn(conv, dropout, trnn_size, num_trnn_layers, batch_size * num_unroll_steps)
            #attention, shape = [batch_size * num_unroll_steps, reduced_length]
            #outputs, shape = [batch_size * num_unroll_steps, reduced_length(max_time), kernel_feature_size]
            #attention = tf.expand_dims(attention, axis=2)
            #average_pool = tf.reduce_mean(tf.multiply(outputs, attention), 1)

            layers.append(conv)

    if len(kernels) > 1:
        output = tf.concat(layers, 2)  # shape = [batch_size * num_unroll_steps, sum(kernel_features)]
    else:
        output = layers[0]

    output = tf.nn.dropout(output, keep_prob= 1. - dropout)

    return output

def multi_linear_attention(input_, dropout, dropout_trnn, trnn_size, num_trnn_layers, batch_size, num_heads, scope = 'multi_linear_attention'):

    features = []

    assert len(trnn_size) == len(num_trnn_layers)

    with tf.variable_scope(scope):

        for idx in range(num_heads):

            matrix = tf.get_variable("Matrix_%d" % idx, [input_.get_shape()[-1], 512], dtype=input_.dtype)
            trnn_input = tf.tensordot(input_, matrix, axes= [2, 0])

            trnn_input = tf.nn.dropout(trnn_input, keep_prob= 1. - dropout)

            attention, output = trnn(trnn_input, dropout_trnn, trnn_size[idx], num_trnn_layers[idx], batch_size,
                                     'MultiLinearAttention_TRNN_{}'.format(idx + 1))

            attention = tf.expand_dims(attention, axis=2)
            features.append(tf.reduce_sum(tf.multiply(output, attention), 1))

    if num_heads> 1:
        output = tf.concat(features, 1)
    else:
        output = features[0]

    #print (output)
    return output

def trnn(input_, dropout, trnn_size, num_trnn_layers, batch_size, scope= None):

    # input_, shape = [batch_size * num_unroll_steps, max_sentence_length, embedding_size]
    with tf.variable_scope(scope or 'LSTM_trnn'):

        #selector = tf.get_variable('selector', [trnn_size], dtype= input_.dtype)

        if num_trnn_layers > 1:
            # cell = tf.contrib.rnn.MultiRNNCell([create_cnn_cell() for _ in range(num_rnn_layers)])
            fw_cell = []
            bw_cell = []
            for _ in range(num_trnn_layers):
                f, b = create_cnn_cell(dropout, trnn_size)
                fw_cell.append(f)
                bw_cell.append(b)
            fw_cell = tf.contrib.rnn.MultiRNNCell(fw_cell)
            bw_cell = tf.contrib.rnn.MultiRNNCell(bw_cell)
        else:

            fw_cell, bw_cell = create_cnn_cell(dropout, trnn_size)

        initial_fw_state = fw_cell.zero_state(batch_size, dtype='float32')
        initial_bw_state = bw_cell.zero_state(batch_size, dtype='float32')

        #print (input_)

        outputs, final_rnn_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_,
                                                                  initial_state_fw=initial_fw_state,
                                                                  initial_state_bw=initial_bw_state,
                                                                  dtype='float32')
        outputs_fw, outputs_bw = outputs

        #outputs shape = [batch_size, max_time, rnn_output_size]

        outputs = tf.concat([outputs_fw, outputs_bw], axis = 2)

        outputs_conv = tf.expand_dims(outputs, 3)
        conv = tf.squeeze(conv2d(outputs_conv, 1, 1, trnn_size * 2, name='trnn_conv2d'), [2,3])

        conv = tf.multiply(conv, 1. / tf.sqrt(tf.cast(outputs.get_shape()[-1], tf.float32)))

        #print(conv), attention, shape = [batch_size * num_unroll_steps, 30]
        attention = tf.nn.softmax(conv, name = 'attention')

        return attention, outputs

def create_cnn_cell(dropout, rnn_size):
    fw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0, reuse=False)
    bw_cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True, forget_bias=1.0, reuse=False)

    #if dropout > 0.0:
    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1. - dropout)
    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=1. - dropout)
    return fw_cell, bw_cell

def inference_graph(word_vocab_size = 1223,
                    kernels = [2, 3, 4],
                    kernel_features = [100, 100, 100],
                    rnn_size = 256,
                    dropout = 0.0,
                    num_rnn_layers = 1,
                    num_highway_layers = 1,
                    num_unroll_steps = 60,
                    max_sent_length = 30,
                    embed_size = 120,
                    trnn_size=[256, 256],
                    num_trnn_layers=[2, 2],
                    num_heads = 2):

        '''

        :param training:
        :param kernels:
        :param kernel_features:
        :param rnn_size:
        :param dropout:
        :param num_rnn_layers:
        :param num_highway_layers:
        :param num_unroll_steps:
        :param max_sent_length:
        :param batch_size:
        :param embed_size:
        :return:
        '''



        assert len(kernels) == len(kernel_features)

        input_ = tf.placeholder(tf.int32, shape = [None, num_unroll_steps, max_sent_length], name = 'input')

        batch_size = tf.placeholder(tf.int32, shape = [], name = 'batch_size')
        training = tf.placeholder(tf.bool, shape = [], name = 'training')

        dropout_LSTM = tf.placeholder(tf.float32, shape = [], name = 'dropout_LSTM')
        dropout_text = tf.placeholder(tf.float32, shape = [], name = 'dropoiut_text')
        dropout_atdnn =tf.placeholder(tf.float32, shape = [], name = 'dropout_atdnn')
        dropout_trnn = tf.placeholder(tf.float32, shape = [], name = 'dropout_trnn')
        dropout_mlattention = tf.placeholder(tf.float32, shape = [], name = 'dropout_mlattention')



        sequence_length = tf.placeholder(tf.int32, shape = [None], name = 'seq_len')

        with tf.variable_scope('Embedding'):

            word_embedding = tf.placeholder(tf.float32, shape=[word_vocab_size, embed_size], name='word_embedding')

            input_embedded = tf.nn.embedding_lookup(word_embedding, input_)



            input_embedded = tf.reshape(input_embedded, [-1, max_sent_length, embed_size])

            input_embedded = tf.nn.dropout(input_embedded, keep_prob= 1. - dropout_text)

        input_cnn = advanced_tdnn(input_embedded, dropout_atdnn, kernels, kernel_features)
        input_cnn = multi_linear_attention(input_cnn, dropout_mlattention, dropout_trnn, trnn_size, num_trnn_layers, batch_size * num_unroll_steps, num_heads)


        if num_highway_layers > 0:
            input_cnn = highway(input_cnn, input_cnn.get_shape()[-1], num_layers= num_highway_layers)


        with tf.variable_scope('LSTM'):


            def create_cnn_cell():

                cell = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple = True, forget_bias = 1.0, reuse = False)
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = 1. - dropout_LSTM)

                return cell

            if num_rnn_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell([create_cnn_cell() for _ in range(num_rnn_layers)])
            else:
                cell = create_cnn_cell()

            initial_state = cell.zero_state(batch_size, dtype='float32')


            input_cnn = tf.reshape(input_cnn, [-1, num_unroll_steps, input_cnn.get_shape()[-1]])

            outputs, final_rnn_state = tf.nn.dynamic_rnn(cell, input_cnn, initial_state=initial_state,
                                                         sequence_length=sequence_length, dtype='float32')

            outputs = tf.reshape(outputs, shape=[-1, outputs.get_shape()[-1]])

            predictions_arousal = tf.nn.tanh(linear(outputs, 1, scope='prediction_arousal_linear'))
            predictions_valence = tf.nn.tanh(linear(outputs, 1, scope='prediction_valence_linear'))
            predictions_liking = tf.nn.tanh(linear(outputs, 1, scope='prediction_liking_linear'))

        return adict(input = input_,
                 training = training,

                 word_embedding = word_embedding,
                 input_embedded = input_embedded,
                 input_cnn = input_cnn,
                 initial_state = initial_state,
                 final_rnn_state = final_rnn_state,
                 rnn_outputs = outputs,
                 predictions_arousal=predictions_arousal,
                 predictions_valence=predictions_valence,
                 predictions_liking=predictions_liking,
                 sequence_length = sequence_length,
                 batch_size = batch_size,
                 dropout_LSTM = dropout_LSTM,
                 dropout_text = dropout_text,
                 dropout_atdnn = dropout_atdnn,
                 dropout_trnn = dropout_trnn,
                 dropout_mlattention = dropout_mlattention)


def loss_graph_ccc_arousal_valence(predictions, groundtruth):

    with tf.variable_scope('CCC_LOSS_AV'):

        groundtruth = tf.reshape(groundtruth, shape = [-1, 3])

        loss = 0

        for i in range(2):
            pred_mean, pred_var = tf.nn.moments(predictions[:, i], [0])
            gt_mean, gt_var = tf.nn.moments(groundtruth[:, i], [0])

            mean_cent_prod = tf.reduce_mean((predictions[:, i] - pred_mean) * (groundtruth[:, i] - gt_mean))

            loss += 1. - 2. * mean_cent_prod / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

        loss = loss / 2.

        return adict(
            AV_CCC = loss
        )

def metric_graph():
    with tf.variable_scope('CCC'):
        pred = tf.placeholder(tf.float32, [None, 3], name = 'pred')
        label = tf.placeholder(tf.float32, [None, 3], name= 'label')

        metric = {0: 0.0, 1: 0.0, 2: 0.0}

        for i in [0, 1, 2]:

            pred_mean, pred_var = tf.nn.moments(pred[:, i], [0])
            gt_mean, gt_var = tf.nn.moments(label[:, i], [0])

            mean_cent_prod = tf.reduce_mean((pred[:,i] - pred_mean) * (label[:,i] - gt_mean))
            #cov = tf.contrib.metrics.streaming_covariance(pred[:, i], label[:, i])[0]



            metric[i] = 2. * mean_cent_prod / (pred_var + gt_var + tf.square(pred_mean - gt_mean))

    return adict(
        eval_predictions = pred,
        eval_labels = label,
        eval_metric_arousal = metric[0],
        eval_metric_valence = metric[1],
        eval_metric_liking = metric[2]
    )


def training_graph(loss, learning_rate, max_grad_norm=3.0):
    ''' Builds training graph. '''

    # global_step is very useful for certain training phrase
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        #learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        # tvars is all the variables that could be trained
        tvars = tf.trainable_variables()

        # get gradients for all variables that could be trained, max is 5.0
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)

        # following is the training process, which is just straightforward training
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate, beta1= 0.9, beta2= 0.99)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)