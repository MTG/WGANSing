import tensorflow as tf
import modules_tf as modules
import config
from data_pipeline import data_gen
import time, os
import utils
import h5py
import numpy as np
import mir_eval
import pandas as pd
from random import randint
import librosa
# import sig_process

import soundfile as sf

import matplotlib.pyplot as plt
from scipy.ndimage import filters


def one_hotize(inp, max_index=config.num_phos):


    output = np.eye(max_index)[inp.astype(int)]

    return output

class Model(object):
    def __init__(self):
        self.get_placeholders()
        self.model()


    def test_file_all(self, file_name, sess):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        return scores

    def validate_file(self, file_name, sess):
        """
        Function to extract multi pitch from file, for validation. Currently supports only HDF5 files.
        """
        scores = self.extract_f0_file(file_name, sess)
        pre = scores['Precision']
        acc = scores['Accuracy']
        rec = scores['Recall']
        return pre, acc, rec




    def load_model(self, sess, log_dir):
        """
        Load model parameters, for synthesis or re-starting training. 
        """
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep= config.max_models_to_keep)


        sess.run(self.init_op)

        ckpt = tf.train.get_checkpoint_state(log_dir)

        if ckpt and ckpt.model_checkpoint_path:
            print("Using the model in %s"%ckpt.model_checkpoint_path)
            self.saver.restore(sess, ckpt.model_checkpoint_path)


    def save_model(self, sess, epoch, log_dir):
        """
        Save the model.
        """
        checkpoint_file = os.path.join(log_dir, 'model.ckpt')
        self.saver.save(sess, checkpoint_file, global_step=epoch)

    def print_summary(self, print_dict, epoch, duration):
        """
        Print training summary to console, every N epochs.
        Summary will depend on model_mode.
        """

        print('epoch %d took (%.3f sec)' % (epoch + 1, duration))
        for key, value in print_dict.items():
            print('{} : {}'.format(key, value))
            

class WGANSing(Model):

    def get_optimizers(self):
        """
        Returns the optimizers for the model, based on the loss functions and the mode. 
        """

        self.final_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Final_Model')
        self.d_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'Discriminator')

        self.final_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)
        self.dis_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5)


        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.global_step_dis = tf.Variable(0, name='dis_global_step', trainable=False)



        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.final_train_function = self.final_optimizer.minimize(self.final_loss, global_step = self.global_step, var_list = self.final_params)
            self.dis_train_function = self.dis_optimizer.minimize(self.D_loss, global_step = self.global_step_dis, var_list = self.d_params)
            self.clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, -0.01, 0.01)) for var in self.d_params]

    def loss_function(self):
        """
        returns the loss function for the model, based on the mode. 
        """



        self.final_loss = tf.reduce_sum(tf.abs(self.output_placeholder- (self.output/2+0.5)))/(config.batch_size*config.max_phr_len*64) * config.lambda + tf.reduce_mean(self.D_fake+1e-12)

        self.D_loss = tf.reduce_mean(self.D_real +1e-12) - tf.reduce_mean(self.D_fake+1e-12)



    def get_summary(self, sess, log_dir):
        """
        Gets the summaries and summary writers for the losses.
        """


        self.final_summary = tf.summary.scalar('final_loss', self.final_loss)

        self.dis_summary = tf.summary.scalar('dis_loss', self.D_loss)




        self.train_summary_writer = tf.summary.FileWriter(log_dir+'train/', sess.graph)
        self.val_summary_writer = tf.summary.FileWriter(log_dir+'val/', sess.graph)
        self.summary = tf.summary.merge_all()


    def get_placeholders(self):
        """
        Returns the placeholders for the model. 
        Depending on the mode, can return placeholders for either just the generator or both the generator and discriminator.
        """

        self.output_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, config.output_features),
                                           name='output_placeholder')       


        self.phoneme_labels = tf.placeholder(tf.int32, shape=(config.batch_size, config.max_phr_len),
                                        name='phoneme_placeholder')
        self.phone_onehot_labels = tf.one_hot(indices=tf.cast(self.phoneme_labels, tf.int32), depth = config.num_phos)
        
        self.f0_placeholder = tf.placeholder(tf.float32, shape=(config.batch_size, config.max_phr_len, 1),
                                        name='f0_placeholder')


        self.singer_labels = tf.placeholder(tf.float32, shape=(config.batch_size),name='singer_placeholder')
        self.singer_onehot_labels = tf.one_hot(indices=tf.cast(self.singer_labels, tf.int32), depth = config.num_singers)


        self.is_train = tf.placeholder(tf.bool, name="is_train")


    def train(self):
        """
        Function to train the model, and save Tensorboard summary, for N epochs. 
        """
        sess = tf.Session()


        self.loss_function()
        self.get_optimizers()
        self.load_model(sess, config.log_dir)
        self.get_summary(sess, config.log_dir)
        start_epoch = int(sess.run(tf.train.get_global_step()) / (config.batches_per_epoch_train))


        print("Start from: %d" % start_epoch)


        for epoch in range(start_epoch, config.num_epochs):

            data_generator = data_gen()
            val_generator = data_gen(mode = 'Val')


            val_final_loss = 0
            val_dis_loss = 0
            batch_num = 0
            epoch_final_loss = 0
            epoch_dis_loss = 0





            start_time = time.time()




            with tf.variable_scope('Training'):
                for feats_targs, f0_out, pho_targs,targets_singers in data_generator:


                    final_loss, dis_loss, summary_str = self.train_model(feats_targs, f0_out, pho_targs,targets_singers, epoch, sess)



                    epoch_final_loss+=final_loss
                    epoch_dis_loss+=dis_loss




                    self.train_summary_writer.add_summary(summary_str, epoch)
                    self.train_summary_writer.flush()

                    utils.progress(batch_num,config.batches_per_epoch_train, suffix = 'training done')

                    batch_num+=1

                epoch_final_loss = epoch_final_loss/batch_num
                epoch_dis_loss = epoch_dis_loss/batch_num


                print_dict = {"Final Loss": epoch_final_loss}

                print_dict["Dis Loss"] =  epoch_dis_loss



            if (epoch + 1) % config.validate_every == 0:
                batch_num = 0
                with tf.variable_scope('Validation'):
                    for feats_targs, f0_out, pho_targs,targets_singers in val_generator:

                        final_loss, dis_loss, summary_str = self.validate_model(feats_targs, f0_out, pho_targs,targets_singers, sess)

                        val_final_loss+=final_loss
                        val_dis_loss+=dis_loss

                        self.val_summary_writer.add_summary(summary_str, epoch)
                        self.val_summary_writer.flush()
                        batch_num+=1

                        utils.progress(batch_num, config.batches_per_epoch_val, suffix='validation done')

                    val_final_loss = val_final_loss/batch_num
                    val_dis_loss = val_dis_loss/batch_num


                    print_dict["Val Final Loss"] =  val_final_loss
                    print_dict["Val Dis Loss"] =  val_dis_loss



            end_time = time.time()
            if (epoch + 1) % config.print_every == 0:
                self.print_summary(print_dict, epoch, end_time-start_time)
            if (epoch + 1) % config.save_every == 0 or (epoch + 1) == config.num_epochs:
                self.save_model(sess, epoch+1, config.log_dir)


    def train_model(self,feats_targs, f0_out, pho_targs,targets_singers, epoch, sess):
        """
        Function to train the model for each epoch
        """


        if epoch<25 or epoch%100 == 0:
            n_critic = 15
        else:
            n_critic = 5
        feed_dict = {self.output_placeholder: feats_targs[:,:,:-2], self.f0_placeholder: f0_out,self.phoneme_labels: pho_targs, self.singer_labels:targets_singers,  self.is_train: True}
        for critic_itr in range(n_critic):
            sess.run(self.dis_train_function, feed_dict = feed_dict)
            sess.run(self.clip_discriminator_var_op, feed_dict = feed_dict)
            
        _, final_loss, dis_loss = sess.run([self.final_train_function,self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, dis_loss, summary_str
 

    def validate_model(self,feats_targs, f0_out, pho_targs,targets_singers, sess):
        """
        Function to train the model for each epoch
        """
        # assert (np.argmax(singer_targs, axis = -1)<4).all()
        feed_dict = {self.output_placeholder: feats_targs[:,:,:-2], self.f0_placeholder: f0_out,self.phoneme_labels: pho_targs, self.singer_labels:targets_singers,  self.is_train: False}
            
        final_loss, dis_loss = sess.run([self.final_loss, self.D_loss], feed_dict=feed_dict)

        summary_str = sess.run(self.summary, feed_dict=feed_dict)


        return final_loss, dis_loss, summary_str



    def read_hdf5_file(self, file_name):
        """
        Function to read and process input file, given name and the synth_mode.
        Returns features for the file based on mode (0 for hdf5 file, 1 for wav file).
        Currently, only the HDF5 version is implemented.
        """
        # if file_name.endswith('.hdf5'):
        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()

        with h5py.File(config.voice_dir + file_name) as feat_file:

            feats = np.array(feat_file['feats'])[()]

            pho_target = np.array(feat_file["phonemes"])[()]

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])


        return feats, f0_nor, pho_target



    def test_file_hdf5(self, file_name, singer_index):
        """
        Function to extract multi pitch from file. Currently supports only HDF5 files.
        """
        sess = tf.Session()
        self.load_model(sess, log_dir = config.log_dir)
        feats, f0_nor, pho_target = self.read_hdf5_file(file_name)

        out_feats = self.process_file(f0_nor, pho_target, singer_index,  sess)

        self.plot_features(feats, out_feats)

        synth = utils.query_yes_no("Synthesize output? ")

        if synth:

            out_featss = np.concatenate((out_feats[:feats.shape[0]], feats[:out_feats.shape[0],-2:]), axis = -1)

            utils.feats_to_audio(out_featss,file_name[:-4]+'output') 
        synth_ori = utils.query_yes_no("Synthesize gorund truth with vocoder? ")

        if synth_ori:
            utils.feats_to_audio(feats,file_name[:-4]+'ground_truth') 

    def plot_features(self, feats, out_feats):

        plt.figure(1)
        
        ax1 = plt.subplot(211)

        plt.imshow(feats[:,:-2].T,aspect='auto',origin='lower')

        ax1.set_title("Ground Truth STFT", fontsize=10)

        ax3 =plt.subplot(212, sharex = ax1, sharey = ax1)

        ax3.set_title("Output STFT", fontsize=10)

        plt.imshow(out_feats.T,aspect='auto',origin='lower')


        plt.show()


    def process_file(self,f0_nor, pho_target, singer_index,  sess):

        stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

        max_feat = np.array(stat_file["feats_maximus"])
        min_feat = np.array(stat_file["feats_minimus"])
        stat_file.close()



        in_batches_f0, nchunks_in = utils.generate_overlapadd(np.expand_dims(f0_nor, -1))

        in_batches_pho, nchunks_in_pho = utils.generate_overlapadd(np.expand_dims(pho_target, -1))

        in_batches_pho = in_batches_pho.reshape([in_batches_pho.shape[0], config.batch_size, config.max_phr_len])


        out_batches_feats = []


        for in_batch_f0, in_batch_pho in zip(in_batches_f0, in_batches_pho) :
            speaker = np.repeat(singer_index, config.batch_size)
            feed_dict = { self.f0_placeholder: in_batch_f0,self.phoneme_labels: in_batch_pho, self.singer_labels:speaker, self.is_train: False}
            out_feats = sess.run(self.output, feed_dict=feed_dict)
            out_batches_feats.append(out_feats)

        out_batches_feats = np.array(out_batches_feats)

        out_batches_feats = utils.overlapadd(out_batches_feats,nchunks_in)

        out_batches_feats = out_batches_feats/2+0.5

        out_batches_feats = out_batches_feats*(max_feat[:-2] - min_feat[:-2]) + min_feat[:-2]

        return out_batches_feats



    def model(self):
        """
        The main model function, takes and returns tensors.
        Defined in modules.

        """


        with tf.variable_scope('Final_Model') as scope:
            self.output = modules.full_network(self.phone_onehot_labels, self.f0_placeholder, self.singer_onehot_labels, self.is_train)
            # self.output_decoded = tf.nn.sigmoid(self.output)
            # self.output_wav_decoded = tf.nn.sigmoid(self.output_wav)
        with tf.variable_scope('Discriminator') as scope: 
            self.D_real = modules.discriminator((self.output_placeholder-0.5)*2, self.phone_onehot_labels, self.f0_placeholder, self.singer_onehot_labels, self.is_train)
            scope.reuse_variables()
            self.D_fake = modules.discriminator(self.output, self.phone_onehot_labels, self.f0_placeholder, self.singer_onehot_labels, self.is_train)



def test():
    # model = DeepSal()
    # # model.test_file('nino_4424.hdf5')
    # model.test_wav_folder('./helena_test_set/', './results/')

    model = MultiSynth()
    model.train()

if __name__ == '__main__':
    test()





