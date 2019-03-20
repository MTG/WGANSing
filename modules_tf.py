from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib import rnn
import config


tf.logging.set_verbosity(tf.logging.INFO)

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d"):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    # try:
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1], name = name)

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    # if with_w:
    #   return deconv, w, biases
    # else:
  return deconv
def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def bi_dynamic_stacked_RNN(x, input_lengths, scope='RNN'):
    with tf.variable_scope(scope):
    # x = tf.layers.dense(x, 128)

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, _state1, state2  = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=[cell,cell2],
            cells_bw=[cell,cell2],
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

    return outputs

def bi_static_stacked_RNN(x, scope='RNN'):
    """
    Input and output in batch major format
    """
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        output = x
        num_layer = 2
        # for n in range(num_layer):
        lstm_fw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)
        lstm_bw = tf.nn.rnn_cell.LSTMCell(config.lstm_size, state_is_tuple=True)

        _initial_state_fw = lstm_fw.zero_state(config.batch_size, tf.float32)
        _initial_state_bw = lstm_bw.zero_state(config.batch_size, tf.float32)

        output, _state1, _state2 = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw, lstm_bw, output, 
                                                  initial_state_fw=_initial_state_fw,
                                                  initial_state_bw=_initial_state_bw, 
                                                  scope='BLSTM_')
        output = tf.stack(output)
        output_fw = output[0]
        output_bw = output[1]
        output = tf.transpose(output, [1,0,2])


        # output = tf.layers.dense(output, config.output_features, activation=tf.nn.relu) # Remove this to use cbhg

        return output




def bi_dynamic_RNN(x, input_lengths, scope='RNN'):
    """
    Stacked dynamic RNN, does not need unpacking, but needs input_lengths to be specified
    """

    with tf.variable_scope(scope):

        cell = tf.nn.rnn_cell.LSTMCell(num_units=config.lstm_size, state_is_tuple=True)

        outputs, states  = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell,
            cell_bw=cell,
            dtype=config.dtype,
            sequence_length=input_lengths,
            inputs=x)

        outputs = tf.concat(outputs, axis=2)

    return outputs


def RNN(x, scope='RNN'):
    with tf.variable_scope(scope):
        x = tf.unstack(x, config.max_phr_len, 1)

        lstm_cell = rnn.BasicLSTMCell(num_units=config.lstm_size)

        # Get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=config.dtype)
        outputs=tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

    return outputs


def highwaynet(inputs, scope='highway', units=config.highway_units):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.relu,
        name='H')
        T = tf.layers.dense(
        inputs,
        units=units,
        activation=tf.nn.sigmoid,
        name='T',
        bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv(inputs, kernel_size, filters=config.conv_filters, activation=config.conv_activation, training=True, scope='conv'):
  with tf.variable_scope(scope):
    x = tf.layers.conv1d(inputs,filters=filters,kernel_size=kernel_size,activation=activation,padding='same')
    return tf.layers.batch_normalization(x, training=training)


# def build_encoder(inputs):
#     embedding_encoder = variable_scope.get_variable("embedding_encoder", [config.vocab_size, config.inp_embedding_size], dtype=config.dtype)

def conv_bank(inputs, scope='conv_bank', num_layers=config.num_conv_layers, training=True):
    with tf.variable_scope(scope):
        outputs = [conv(inputs, k, training=training, scope='conv_%d' % k) for k in range(1, num_layers+1)]
        outputs = tf.concat(outputs, axis=-1)
        outputs = tf.layers.max_pooling1d(outputs,pool_size=2,strides=1,padding='same')
    return outputs




def cbhg(inputs, scope='cbhg', training=True):
    with tf.variable_scope(scope):
        # Prenet
        if training:
            dropout = config.dropout_rate
        else:
            dropout = 0.0
        # tf.summary.histogram('inputs', inputs)
        prenet_out = tf.layers.dropout(tf.layers.dense(inputs, config.lstm_size*2), rate=dropout)
        prenet_out = tf.layers.dropout(tf.layers.dense(prenet_out, config.lstm_size), rate=dropout)
        # tf.summary.histogram('prenet_output', prenet_out)

        # Conv Bank
        x = conv_bank(prenet_out, training=training)


        # Projections
        x = conv(x, config.projection_size, config.conv_filters, training=training, scope='proj_1')
        x = conv(x, config.projection_size, config.conv_filters,activation=None, training=training, scope='proj_2')

        assert x.shape[-1]==config.highway_units

        x = x+prenet_out

        for i in range(config.highway_layers):
            x = highwaynet(x, scope='highway_%d' % (i+1))
        x = bi_static_stacked_RNN(x)
        x = tf.layers.dense(x, config.output_features)

        output = tf.layers.dense(x, 128, activation=tf.nn.relu) # Remove this to use cbhg
        harm = tf.layers.dense(output, 60, activation=tf.nn.relu)
        ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
        f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
        f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
        vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)
        phonemes = tf.layers.dense(output, 41, activation=tf.nn.relu)
    return harm, ap, f0, vuv, phonemes
        


def nr_wavenet_block(inputs, dilation_rate = 2, name = "name"):

    con_pad_forward = tf.pad(inputs, [[0,0],[dilation_rate,dilation_rate],[0,0]],"CONSTANT")

    con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 3, dilation_rate = dilation_rate, padding = 'valid', name = name+"1", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    sig = tf.sigmoid(con_sig_forward)

    con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 3, dilation_rate = dilation_rate, padding = 'valid', name = name+"3", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    tanh = tf.tanh(con_tanh_forward)

    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"5")

    residual = skip + inputs

    return skip, residual

def nr_wavenet_block_d(conditioning,filters=2, scope = 'nr_wavenet_block', name = "name"):

    with tf.variable_scope(scope):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])

        # con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
        # con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
        con_sig_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"1")
        # con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid', name = name+"2")
        # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

        sig = tf.sigmoid(con_sig_forward)


        con_tanh_forward = tf.layers.conv1d(conditioning, config.wavenet_filters, filters, padding = 'valid', name = name+"2")

        tanh = tf.tanh(con_tanh_forward)


        outputs = tf.multiply(sig,tanh)

        skip = tf.layers.conv1d(outputs,config.wavenet_filters,1, name = name+"5")

        residual = skip

    return skip, residual


def nr_wavenet(inputs, num_block = config.wavenet_layers):
    prenet_out = tf.layers.dense(inputs, config.lstm_size*2)
    prenet_out = tf.layers.dense(prenet_out, config.lstm_size)

    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1)
    skips = []
    skip, residual = nr_wavenet_block(first_conv, dilation_rate=1, scope = "nr_wavenet_block_0")
    output = skip
    for i in range(num_block):
        skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1), scope = "nr_wavenet_block_"+str(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,config.wavenet_filters,1)

    output = tf.nn.relu(output)

    harm = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)

    return harm, ap, f0, vuv

# def final_net(encoded, f0, phones):

#     encoded_embedding = tf.layers.dense(encoded, 128)


    
#     embed_1 = tf.layers.dense(f0, 64)

#     embed_ph = tf.layers.dense(phones, 64)

#     inputs_2 = tf.concat([embed_1, embed_ph], axis = -1)

#     conv1 = tf.layers.conv1d(inputs=inputs_2, filters=128, kernel_size=2, padding='same', activation=tf.nn.relu)

#     maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, padding='same')

#     conv2 = tf.layers.conv1d(inputs=maxpool1, filters=64, kernel_size=4, padding='same', activation=tf.nn.relu)

#     maxpool2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding='same')

#     conv3 = tf.layers.conv1d(inputs=maxpool2, filters=32, kernel_size=4, padding='same', activation=tf.nn.relu)

#     encoded = tf.layers.max_pooling1d(conv3, pool_size=2, strides=2, padding='same')

#     encoded = tf.concat([tf.reshape(encoded, [config.batch_size, -1]), encoded_embedding], axis = -1)

#     upsample1 = tf.image.resize_images(tf.reshape(encoded, [30,4,1,-1]), size=(16,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#     # import pdb;pdb.set_trace()

#     conv4 = tf.layers.conv2d(inputs=upsample1, filters=32, kernel_size=(2,1), padding='same', activation=tf.nn.relu)
#     # Now 7x7x16
#     upsample2 = tf.image.resize_images(conv4, size=(32,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     # Now 14x14x16
#     conv5 = tf.layers.conv2d(inputs=upsample2, filters=64, kernel_size=(2,1), padding='same', activation=tf.nn.relu)
#     # Now 14x14x32
#     upsample3 = tf.image.resize_images(conv5, size=(64,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     # Now 28x28x32
#     conv6 = tf.layers.conv2d(inputs=upsample3, filters=64, kernel_size=(2,1), padding='same', activation=tf.nn.relu)

#     upsample4 = tf.image.resize_images(conv6, size=(config.max_phr_len,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     # Now 28x28x32
#     conv7 = tf.layers.conv2d(inputs=upsample4, filters=128, kernel_size=(2,1), padding='same', activation=tf.nn.relu)

#     # encoded_embedding = tf.reshape(tf.tile(encoded_embedding, [1,config.max_phr_len]), [config.batch_size, config.max_phr_len, 32])

#     # encoded_embedding = tf.reshape(tf.image.resize_images(tf.reshape(encoded_embedding, [30,1,1,32]), size=(config.max_phr_len,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), [config.batch_size, config.max_phr_len, 32])
# # 
#     # import pdb;pdb.set_trace()

#     output_2 = tf.reshape(conv7, [30, config.max_phr_len, 128])

#     output_1 = bi_static_stacked_RNN(output_2, scope = 'RNN_3')

#     output_1 = tf.layers.dense(output_1, 256)

#     final_voc = tf.layers.dense(output_1, 64)

#     return final_voc


def final_net(singer_label, f0_notation, phones):

    # import pdb;pdb.set_trace()

    # singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "f_condi"), [config.batch_size,1,1,-1], name = "f_condi_reshape")

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    phones = tf.layers.dense(phones, config.wavenet_filters, name = "F_phone")

    f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "F_f0")

    inputs = tf.concat([phones, f0_notation, singer_label], axis = -1)

    inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "F_in")

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    conv1 =  tf.nn.relu(tf.layers.conv2d(inputs, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_1"))
    # import pdb;pdb.set_trace()

    # conv2 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_2"))

    # conv3 =  tf.nn.relu(tf.layers.conv2d(conv2, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_3") )

    # conv4 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_4"))

    conv5 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_5"))
    # import pdb;pdb.set_trace()

    conv6 =  tf.nn.relu(tf.layers.conv2d(conv5, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_6"))

    conv7 = tf.nn.relu(tf.layers.conv2d(conv6, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_7"))

    conv8 = tf.nn.relu(tf.layers.conv2d(conv7, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_8"))

    conv9 =  tf.nn.relu(tf.layers.conv2d(conv8, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "F_9") + conv8 )

    conv10 =  tf.nn.relu(tf.layers.conv2d(conv9, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "F_10") + conv9)

    conv11 =  tf.nn.relu(tf.layers.conv2d(conv10, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "F_11") + conv10)

    conv12 =  tf.nn.relu(tf.layers.conv2d(conv11, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "F_12") + conv11)

    deconv1 = tf.nn.relu(deconv2d(conv12, [config.batch_size, 8, 1, config.wavenet_filters], name = "F_dec1") +  conv7)

    deconv2 = tf.nn.relu(deconv2d(deconv1, [config.batch_size, 16, 1, config.wavenet_filters], name = "F_dec2") + conv6)

    deconv3 = tf.nn.relu(deconv2d(deconv2, [config.batch_size, 32, 1, config.wavenet_filters], name = "F_dec3") + conv5)

    deconv4 = tf.nn.relu(deconv2d(deconv3, [config.batch_size, 64, 1, config.wavenet_filters], name = "F_dec4") + conv1)

    deconv5 = tf.nn.relu(deconv2d(deconv4, [config.batch_size, 128, 1, config.wavenet_filters], name = "F_dec5") +  inputs)

    # deconv6 = deconv2d(deconv5, [config.batch_size, 256, 1, config.wavenet_filters], name = "G_dec6") + conv2

    # deconv7 = deconv2d(deconv6, [config.batch_size, 512, 1, config.wavenet_filters], name = "G_dec7") + conv1

    # deconv8 = deconv2d(deconv7, [config.batch_size, 1024, 1, config.wavenet_filters], name = "G_dec8") + inputs

    output = tf.nn.relu(tf.layers.conv2d(deconv5 , config.wavenet_filters, 1, strides=1,  padding = 'same', name = "F_o"))

    output = tf.layers.conv2d(output, 64, 1, strides=1,  padding = 'same', name = "F_o_2", activation = None)

    output = tf.reshape(output, [config.batch_size, config.max_phr_len, -1])

    output = tf.layers.dense(output, 64, name = "F_F")

    return output


def phone_network(inputs):

    inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "P_in")

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    conv1 =  tf.nn.relu(tf.layers.conv2d(inputs, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "P_1"))

    conv5 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "P_5"))
    # import pdb;pdb.set_trace()

    conv6 =  tf.nn.relu(tf.layers.conv2d(conv5, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "P_6"))

    conv7 = tf.nn.relu(tf.layers.conv2d(conv6, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "P_7"))

    conv8 = tf.nn.relu(tf.layers.conv2d(conv7, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "P_8"))

    conv9 =  tf.nn.relu(tf.layers.conv2d(conv8, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "P_9") + conv8)

    conv10 =  tf.nn.relu(tf.layers.conv2d(conv9, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "P_10") + conv9)

    conv11 =  tf.nn.relu(tf.layers.conv2d(conv10, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "P_11") + conv10)

    conv12 =  tf.nn.relu(tf.layers.conv2d(conv11, config.wavenet_filters, (4,1), strides=1,  padding = 'same', name = "P_12") + conv11)

    deconv1 = tf.nn.relu(deconv2d(conv12, [config.batch_size, 8, 1, config.wavenet_filters], name = "P_dec1") +  conv7)

    deconv2 = tf.nn.relu(deconv2d(deconv1, [config.batch_size, 16, 1, config.wavenet_filters], name = "P_dec2") + conv6)

    deconv3 = tf.nn.relu(deconv2d(deconv2, [config.batch_size, 32, 1, config.wavenet_filters], name = "P_dec3") + conv5)

    deconv4 = tf.nn.relu(deconv2d(deconv3, [config.batch_size, 64, 1, config.wavenet_filters], name = "P_dec4") + conv1)

    deconv5 = tf.nn.relu(deconv2d(deconv4, [config.batch_size, 128, 1, config.wavenet_filters], name = "P_dec5") +  inputs)

    output = tf.nn.relu(tf.layers.conv2d(deconv5 , config.wavenet_filters, 1, strides=1,  padding = 'same', name = "P_o"))

    output = tf.layers.conv2d(output, 64, 1, strides=1,  padding = 'same', name = "P_o_2", activation = None)

    output = tf.reshape(output, [config.batch_size, config.max_phr_len, -1])

    output = tf.layers.dense(output, 42, name = "P_F")

    return output

# def phone_network(inputs):

#     prenet_out = tf.layers.dense(inputs, config.wavenet_filters, name = "dense_p_1")

#     num_block = config.wavenet_layers

#     receptive_field = 2**num_block

#     first_conv = tf.layers.conv1d(prenet_out, config.wavenet_filters, 1, name = "conv_p_1")
#     skips = []
#     skip, residual = nr_wavenet_block(first_conv, dilation_rate=1,  name = "p_nr_wavenet_block_0")
#     output = skip
#     for i in range(num_block):
#         skip, residual = nr_wavenet_block(residual, dilation_rate=2**(i+1),  name = "p_nr_wavenet_block_"+str(i+1) )
#         skips.append(skip)
#     for skip in skips:
#         output+=skip
#     output = output+first_conv

#     output = tf.layers.conv1d(output,config.wavenet_filters,1, name = "conv_p_2")

#     output = tf.layers.dense(output, 42, name = "dense_p_3")

#     return output




def GAN_discriminator(inputs, singer_label, phones, f0_notation):

    # inputs = tf.concat([inputs, conds], axis = -1)

    singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "g_condi"), [config.batch_size,1,1,-1], name = "g_condi_reshape")

    phones = tf.layers.dense(phones, config.wavenet_filters, name = "G_phone", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "G_f0", kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    # # conds = tf.concat([phones, f0_notation], axis = -1)

    # # conds = tf.layers.dense(conds, config.wavenet_filters, name = "G_conds")    

    inputs = tf.concat([phones, f0_notation, singer_label, rand], axis = -1)

    # inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "G_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    # rand = tf.layers.dense(rand, config.wavenet_filters, name = "G_rand", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    conv1 =  tf.nn.relu(tf.layers.conv2d(inputs, 32, (3,1), strides=(2,1),  padding = 'same', name = "G_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv5 =  tf.nn.relu(tf.layers.conv2d(conv1, 64, (3,1), strides=(2,1),  padding = 'same', name = "G_5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    # import pdb;pdb.set_trace()

    conv6 =  tf.nn.relu(tf.layers.conv2d(conv5, 128, (3,1), strides=(2,1),  padding = 'same', name = "G_6", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv7 = tf.nn.relu(tf.layers.conv2d(conv6, 256, (3,1), strides=(2,1),  padding = 'same', name = "G_7", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv8 = tf.nn.relu(tf.layers.conv2d(conv7, 512, (3,1), strides=(2,1),  padding = 'same', name = "G_8", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))


    return conv8

# def GAN_discriminator(inputs, conds):
#     # singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "d_condi"), [config.batch_size,1,1,-1], name = "d_condi_reshape")
#     # singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

#     # phones = tf.layers.dense(phones, config.wavenet_filters, name = "D_phone", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

#     # f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "D_f0", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

#     inputs = tf.concat([inputs, conds], axis = -1)

#     # inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "D_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

#     # import pdb;pdb.set_trace()

#     inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

#     inputs = tf.layers.conv2d(inputs, 512, 1, strides=1,  padding = 'same', name = "D_pre")

#     conv1 =  selu(tf.layers.conv2d(inputs, 256, (3,1), dilation_rate=(2,1),  padding = 'same', name = "D_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv2 =  selu(tf.layers.conv2d(conv1, 128, (3,1), strides=(2,1),  padding = 'same', name = "D_2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv3 =  selu(tf.layers.conv2d(conv2, 64, (3,1), strides=(2,1),  padding = 'same', name = "D_3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv4 =  selu(tf.layers.conv2d(conv3, 32, (3,1), strides=(2,1),  padding = 'same', name = "D_4", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv5 =  selu(tf.layers.conv2d(conv4, 16, (3,1), strides=(2,1),  padding = 'same', name = "D_5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv6 =  selu(tf.layers.conv2d(conv5, 8, (3,1), strides=(2,1),  padding = 'same', name = "D_6", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))


#     conv1_1 =  selu(tf.layers.conv2d(inputs, 64, (32,1), strides=(16,1),  padding = 'same', name = "D_1_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv2_1 =  selu(tf.layers.conv2d(conv1_1, 32, (3,1), strides=(2,1),  padding = 'same', name = "D_2_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv3_1 =  selu(tf.layers.conv2d(conv2_1, 16, (1,1), strides=(1,1),  padding = 'same', name = "D_3_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv4_1 =  selu(tf.layers.conv2d(conv3_1, 8, (1,1), strides=(1,1),  padding = 'same', name = "D_4_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))


#     conv1_2 =  selu(tf.layers.conv2d(inputs, 64, (64,1), strides=(32,1),  padding = 'same', name = "D_1_2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv2_2 =  selu(tf.layers.conv2d(conv1_2, 8, (1,1), strides=(1,1),  padding = 'same', name = "D_2_2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))



#     conv1_3 =  selu(tf.layers.conv2d(inputs, 32, (128,1), strides=(1,1),  padding = 'valid', name = "D_1_3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     # conv2_3 =  selu(tf.layers.conv2d(conv1_3, 12, (1,1), strides=(1,1),  padding = 'same', name = "D_2_3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     conv2_3 = tf.reshape(conv1_3, [30,4,1,8])

#     # import pdb;pdb.set_trace()

#     # conv2 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_2"))

#     # conv3 =  tf.nn.relu(tf.layers.conv2d(conv2, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_3") )

#     # conv4 =  tf.nn.relu(tf.layers.conv2d(conv1, config.wavenet_filters, (4,1), strides=(2,1),  padding = 'same', name = "F_4"))

#     # conv5 =  selu(tf.layers.conv2d(conv1, 256, (32,1), strides=(2,1),  padding = 'same', name = "D_5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    

#     # conv6 =  selu(tf.layers.conv2d(conv5, 128, (16,1), strides=(2,1),  padding = 'same', name = "D_6", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     # conv7 = selu(tf.layers.conv2d(conv6, 64, (8,1), strides=(2,1),  padding = 'same', name = "D_7", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     # conv8 = selu(tf.layers.conv2d(conv7, 32, (4,1), strides=(2,1),  padding = 'same', name = "D_8", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

#     # import pdb;pdb.set_trace()

#     # conv9 =  selu(tf.layers.conv2d(conv8, 512, (3,1), strides=1,  padding = 'same', name = "D_9", kernel_initializer=tf.random_normal_initializer(stddev=0.02)) + conv8)

#     # conv10 =  selu(tf.layers.conv2d(conv9, 512, (3,1), strides=1,  padding = 'same', name = "D_10", kernel_initializer=tf.random_normal_initializer(stddev=0.02)) + conv9)

#     # conv11 =  selu(tf.layers.conv2d(conv10, 512, (3,1), strides=1,  padding = 'same', name = "D_11", kernel_initializer=tf.random_normal_initializer(stddev=0.02)) + conv10)

#     # conv12 =  selu(tf.layers.conv2d(conv11, 512, (3,1), strides=1,  padding = 'same', name = "D_12", kernel_initializer=tf.random_normal_initializer(stddev=0.02)) + conv11)

#     ops = tf.concat([conv6,conv4_1, conv2_2,conv2_3], axis = -1)


#     # ops = tf.layers.dense(ops, 1, name = "d_f_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

#     # output = tf.layers.dense(ops, 1, name = "d_f_2", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

#     return ops




def GAN_generator(singer_label, phones, f0_notation, rand):

    singer_label = tf.reshape(tf.layers.dense(singer_label, config.wavenet_filters, name = "g_condi"), [config.batch_size,1,1,-1], name = "g_condi_reshape")

    phones = tf.layers.dense(phones, config.wavenet_filters, name = "G_phone", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    f0_notation = tf.layers.dense(f0_notation, config.wavenet_filters, name = "G_f0", kernel_initializer=tf.random_normal_initializer(stddev=0.02))
    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    # # conds = tf.concat([phones, f0_notation], axis = -1)

    # # conds = tf.layers.dense(conds, config.wavenet_filters, name = "G_conds")    

    inputs = tf.concat([phones, f0_notation, singer_label, rand], axis = -1)

    # inputs = tf.layers.dense(inputs, config.wavenet_filters, name = "G_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, 1, -1])

    # rand = tf.layers.dense(rand, config.wavenet_filters, name = "G_rand", kernel_initializer=tf.random_normal_initializer(stddev=0.02))

    conv1 =  tf.nn.relu(tf.layers.conv2d(inputs, 32, (3,1), strides=(2,1),  padding = 'same', name = "G_1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv5 =  tf.nn.relu(tf.layers.conv2d(conv1, 64, (3,1), strides=(2,1),  padding = 'same', name = "G_5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))
    # import pdb;pdb.set_trace()

    conv6 =  tf.nn.relu(tf.layers.conv2d(conv5, 128, (3,1), strides=(2,1),  padding = 'same', name = "G_6", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv7 = tf.nn.relu(tf.layers.conv2d(conv6, 256, (3,1), strides=(2,1),  padding = 'same', name = "G_7", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    conv8 = tf.nn.relu(tf.layers.conv2d(conv7, 512, (3,1), strides=(2,1),  padding = 'same', name = "G_8", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv1 = tf.image.resize_images(conv8, size=(8,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv1 = tf.nn.relu(tf.layers.conv2d(deconv1, 512, (3,1), strides=(1,1),  padding = 'same', name = "G_dec1", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv1 = tf.concat([deconv1, conv7], axis = -1)

    deconv2 = tf.image.resize_images(deconv1, size=(16,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv2 = tf.nn.relu(tf.layers.conv2d(deconv2, 256, (3,1), strides=(1,1),  padding = 'same', name = "G_dec2", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv2 = tf.concat([deconv2, conv6], axis = -1)


    deconv3 = tf.image.resize_images(deconv2, size=(32,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv3 = tf.nn.relu(tf.layers.conv2d(deconv3, 128, (3,1), strides=(1,1),  padding = 'same', name = "G_dec3", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv3 = tf.concat([deconv3, conv5], axis = -1)


    deconv4 = tf.image.resize_images(deconv3, size=(64,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv4 = tf.nn.relu(tf.layers.conv2d(deconv4, 64, (3,1), strides=(1,1),  padding = 'same', name = "G_dec4", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv4 = tf.concat([deconv4, conv1], axis = -1)


    deconv5 = tf.image.resize_images(deconv4, size=(128,1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    deconv5 = tf.nn.relu(tf.layers.conv2d(deconv5, 64, (3,1), strides=(1,1),  padding = 'same', name = "G_dec5", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    deconv5 = tf.concat([deconv5, inputs], axis = -1)


    # deconv1 = tf.concat([deconv2d(conv8, [config.batch_size, 8, 1, 512], name = "G_dec1"),  conv7], axis = -1)

    # deconv2 = tf.concat([deconv2d(deconv1, [config.batch_size, 16, 1, 256], name = "g_dec2") , conv6] , axis = -1)

    # deconv3 = tf.concat([deconv2d(deconv2, [config.batch_size, 32, 1, 128], name = "G_dec3"),  conv5] , axis = -1)

    # deconv4 = tf.concat([deconv2d(deconv3, [config.batch_size, 64, 1, 64], name = "G_dec4"),  conv1], axis  = -1)

    # deconv5 = tf.concat([deconv2d(deconv4, [config.batch_size, 128, 1, config.wavenet_filters], name = "G_dec5"), inputs] , axis = -1)

    # output = tf.nn.relu(tf.layers.conv2d(deconv5 , config.wavenet_filters, 1, strides=1,  padding = 'same', name = "G_o", kernel_initializer=tf.random_normal_initializer(stddev=0.02)))

    output = tf.layers.conv2d(deconv5, 64, 1, strides=1,  padding = 'same', name = "G_o_2", activation = tf.nn.tanh)

    # output = tf.layers.conv2d(deconv5, 64*128, (128,1), strides=1,  padding = 'valid', name = "G_o_2", activation = tf.nn.tanh)


    output = tf.reshape(output, [config.batch_size, config.max_phr_len, -1])

    return output


def singer_network(inputs, prob):



    embed_1 = tf.layers.dense(inputs, 32, name = "s1")


    conv1 = tf.layers.conv1d(inputs=embed_1, filters=128, kernel_size=2, padding='same', activation=tf.nn.relu, name = "s2")

    # maxpool1 = tf.layers.max_pooling1d(conv1, pool_size=2, strides=2, padding='same', name = "s3")

    conv2 = tf.layers.conv1d(inputs=conv1, filters=64, kernel_size=4, padding='same', activation=tf.nn.relu, name = "s4")

    # maxpool2 = tf.layers.max_pooling1d(conv2, pool_size=2, strides=2, padding='same', name = "s5")

    conv3 = tf.layers.conv1d(inputs=conv2, filters=32, kernel_size=4, padding='same', activation=tf.nn.relu, name = "s6")

    # encoded = tf.layers.max_pooling1d(conv3, pool_size=2, strides=2, padding='same', name = "s7")

    encoded = tf.reshape(conv3, [config.batch_size, -1], name = "s8")

    encoded_1= tf.layers.dense(encoded, 256, name = "s9")

    singer = tf.layers.dense(encoded_1, 12, name = "s10")

    return encoded_1, singer






    # return x

def wavenet_block(inputs, conditioning, dilation_rate = 2, scope = 'wavenet_block'):
    # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features])
    in_padded = tf.pad(inputs, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_forward = tf.pad(conditioning, [[0,0],[dilation_rate,0],[0,0]],"CONSTANT")
    con_pad_backward = tf.pad(conditioning, [[0,0],[0,dilation_rate],[0,0]],"CONSTANT")
    in_sig = tf.layers.conv1d(in_padded, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_sig_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    # con_sig = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    sig = tf.sigmoid(in_sig+con_sig_forward+con_sig_backward)

    in_tanh = tf.layers.conv1d(in_padded, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_tanh_forward = tf.layers.conv1d(con_pad_forward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')
    con_tanh_backward = tf.layers.conv1d(con_pad_backward, config.wavenet_filters, 2, dilation_rate = dilation_rate, padding = 'valid')    
    # con_tanh = tf.layers.conv1d(conditioning,config.wavenet_filters,1)

    tanh = tf.tanh(in_tanh+con_tanh_forward[:,:in_tanh.shape[1],:]+con_tanh_backward[:,:in_tanh.shape[1],:])


    outputs = tf.multiply(sig,tanh)

    skip = tf.layers.conv1d(outputs,1,1)

    residual = skip + inputs

    return skip, residual


def wavenet(inputs, conditioning, num_block = config.wavenet_layers):
    receptive_field = 2**num_block

    first_conv = tf.layers.conv1d(inputs, 66, 1)
    skips = []
    skip, residual = wavenet_block(first_conv, conditioning, dilation_rate=1)
    output = skip
    for i in range(num_block):
        skip, residual = wavenet_block(residual, conditioning, dilation_rate=2**(i+1))
        skips.append(skip)
    for skip in skips:
        output+=skip
    output = output+first_conv

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,66,1)

    output = tf.nn.relu(output)

    output = tf.layers.conv1d(output,66,1)

    output = tf.nn.relu(output)

    harm_1 = tf.layers.dense(output, 60, activation=tf.nn.relu)
    ap = tf.layers.dense(output, 4, activation=tf.nn.relu)
    f0 = tf.layers.dense(output, 64, activation=tf.nn.relu) 
    f0 = tf.layers.dense(f0, 1, activation=tf.nn.relu)
    vuv = tf.layers.dense(ap, 1, activation=tf.nn.sigmoid)
    return output[:,:,:-1],vuv


# def GAN_generator(inputs, num_block = config.wavenet_layers):
#     # inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len, config.input_features,1])
#     # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_1")
#     # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_2")
#     # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_3")
#     # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_4")
#     # # inputs = tf.layers.conv2d(inputs, config.wavenet_filters, 5,  padding = 'same', name = "G_5")

#     # inputs = tf.layers.conv2d(inputs, 1, 5,  padding = 'same', name = "G_6")

#     inputs = tf.layers.dense(inputs, config.lstm_size, name = "G_1")
#     inputs = tf.layers.dense(inputs, 60, name = "G_2")
#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 1)

#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 2, padding = 'same', name = "G_c1")
#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 4, padding = 'same', name = "G_c2")
#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 8, padding = 'same', name = "G_c3")
#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 16, padding = 'same', name = "G_c4")
#     inputs = tf.layers.conv1d(inputs, config.wavenet_filters, 32, padding = 'same', name = "G_c5")

#     harm = tf.nn.tanh(tf.layers.dense(inputs, 60, name = "G_3"))
#     # import pdb;pdb.set_trace()
#     # inputs = tf.reshape(inputs,[config.batch_size, config.max_phr_len, config.input_features] )
#     return harm
    # import pdb;pdb.set_trace()









def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, config.input_features])
    tec = np.random.rand(config.batch_size, config.max_phr_len, config.input_features) #  batch_size, time_steps, features
    seqlen = tf.placeholder("float", [config.batch_size, 256])
    outs = f0_network_2(seqlen, vec, vec)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    f0_1= sess.run(outs, feed_dict={vec: tec, seqlen: np.random.rand(config.batch_size, 256)})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()