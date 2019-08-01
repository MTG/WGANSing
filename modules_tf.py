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








def selu(x):
   alpha = 1.6732632423543772848170429916717
   scale = 1.0507009873554804934193349852946
   return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))



def encoder_conv_block_gan(inputs, layer_num, is_train, num_filters = config.filters):

    output = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(inputs, num_filters * 2**int(layer_num/2), (config.filter_len,1)
        , strides=(2,1),  padding = 'same', name = "G_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name = "GBN_"+str(layer_num))
    return output

def decoder_conv_block_gan(inputs, layer, layer_num, is_train, num_filters = config.filters):

    deconv = tf.image.resize_images(inputs, size=(int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # embedding = tf.tile(embedding,[1,int(config.max_phr_len/2**(config.encoder_layers - 1 - layer_num)),1,1])

    deconv = tf.layers.batch_normalization(tf.nn.relu(tf.layers.conv2d(deconv, layer.shape[-1]
        , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "D_"+str(layer_num), kernel_initializer=tf.random_normal_initializer(stddev=0.02))), training = is_train, name =  "DBN_"+str(layer_num))

    # embedding =tf.nn.relu(tf.layers.conv2d(embedding, layer.shape[-1]
    #     , (config.filter_len,1), strides=(1,1),  padding = 'same', name =  "DEnc_"+str(layer_num)))

    deconv =  tf.concat([deconv, layer], axis = -1)

    return deconv





def encoder_decoder_archi_gan(inputs, is_train):
    """
    Input is assumed to be a 4-D Tensor, with [batch_size, phrase_len, 1, features]
    """

    encoder_layers = []

    encoded = inputs

    encoder_layers.append(encoded)

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block_gan(encoded, i, is_train)
        encoder_layers.append(encoded)
    
    encoder_layers.reverse()



    decoded = encoder_layers[0]

    for i in range(config.encoder_layers):
        decoded = decoder_conv_block_gan(decoded, encoder_layers[i+1], i, is_train)

    return decoded



def full_network(f0, phos,  singer_label, is_train):

    f0 = tf.layers.batch_normalization(tf.layers.dense(f0, config.filters
        , name = "F0_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'F0_in_BN')

    phos = tf.layers.batch_normalization(tf.layers.dense(phos, config.filters
        , name = "Pho_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'Pho_in_BN')

    singer_label = tf.layers.batch_normalization(tf.layers.dense(singer_label, config.filters
        , name = "Singer_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'Singer_in_BN')

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([f0, phos,singer_label], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])

    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters
        , name = "S_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train,name = 'S_in_BN')


    output = encoder_decoder_archi_gan(inputs, is_train)


    output = tf.tanh(tf.layers.batch_normalization(tf.layers.dense(output, config.output_features, name = "Fu_F", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_out"))

    return tf.squeeze(output)

def discriminator(inputs, phos, f0, singer_label, is_train):

    f0 = tf.layers.batch_normalization(tf.layers.dense(f0, config.filters
        , name = "F0_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'F0_in_BN')

    phos = tf.layers.batch_normalization(tf.layers.dense(phos, config.filters
        , name = "Pho_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'Pho_in_BN')

    singer_label = tf.layers.batch_normalization(tf.layers.dense(singer_label, config.filters
        , name = "Singer_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = 'Singer_in_BN')

    singer_label = tf.tile(tf.reshape(singer_label,[config.batch_size,1,-1]),[1,config.max_phr_len,1])

    inputs = tf.concat([inputs, f0, phos,singer_label], axis = -1)

    inputs = tf.reshape(inputs, [config.batch_size, config.max_phr_len , 1, -1])



    inputs = tf.layers.batch_normalization(tf.layers.dense(inputs, config.filters *2 
        , name = "S_in", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_1")

    encoded = inputs

    for i in range(config.encoder_layers):
        encoded = encoder_conv_block_gan(encoded, i, is_train)
    encoded = tf.squeeze(encoded)

    output = tf.layers.batch_normalization(tf.layers.dense(encoded, 1, name = "Fu_F", kernel_initializer=tf.random_normal_initializer(stddev=0.02)), training = is_train, name = "bn_fu_out")

    return tf.squeeze(output)

def main():    
    vec = tf.placeholder("float", [config.batch_size, config.max_phr_len, config.input_features])
    tec = np.random.rand(config.batch_size, config.max_phr_len,config.input_features) #  batch_size, time_steps, features
    is_train = tf.placeholder(tf.bool, name="is_train")
    # seqlen = tf.placeholder("float", [config.batch_size, 256])
    # with tf.variable_scope('singer_Model') as scope:
    #     singer_emb, outs_sing = singer_network(vec, is_train)
    # with tf.variable_scope('f0_Model') as scope:
    #     outs_f0 = f0_network(vec, is_train)
    # with tf.variable_scope('phone_Model') as scope:
    #     outs_pho = phone_network(vec, is_train)
    with tf.variable_scope('full_Model') as scope:
        out_put = discriminator(vec,is_train)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    op= sess.run(out_put, feed_dict={vec: tec, is_train: True})
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.add_summary(summary, global_step=1)
    import pdb;pdb.set_trace()


if __name__ == '__main__':
  main()