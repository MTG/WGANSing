import numpy as np
import tensorflow as tf


wav_dir_nus = '../datasets/nus-smc-corpus_48/'


voice_dir = '../ss_synthesis/voice/'


log_dir = './log/'






data_log = './log/data_log.log'



stat_dir = './stats/'

val_dir = './val_dir_synth/'

in_mode = 'mix'
norm_mode_out = "max_min"
norm_mode_in = "max_min"

voc_ext = '_voc_stft.npy'
feats_ext = '_synth_feats.npy'

f0_weight = 10
max_models_to_keep = 10
f0_threshold = 1


def get_teacher_prob(epoch):
    if epoch < 500:
        return 0.95
    elif epoch < 1000:
        return 0.75
    else:
        return 0.55

filter_len = 5
encoder_layers = 7
filters = 64


kernel_size = 2
num_filters = 100
skip_filters = 240
first_conv = 10
dilation_rates = [1,2,4,1,2]
wavenet_layers = 5


max_phr_len = 128

output_features = 64


phonemas_nus = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'Sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh', 'ax']

num_phos = len(phonemas_nus)

singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY']

num_singers = len(singers)


split = 0.9

augment = True
aug_prob = 0.5

noise_threshold = 0.4 #0.7 for the unnormalized features
pred_mode = 'all'

# Hyperparameters
num_epochs = 950

batches_per_epoch_train = 100
batches_per_epoch_val = 10

batch_size = 30
samples_per_file = 6
input_features = 513


lamda = 0.001



highway_layers = 4
highway_units = 128
init_lr = 0.0002
num_conv_layers = 8
conv_filters = 128
conv_activation = tf.nn.relu
dropout_rate = 0.0
projection_size = 3
fs = 44100
comp_mode = 'mfsc'
hoptime = 5.80498866

noise = 0.05


rec_field = 2**wavenet_layers
wavenet_filters = 64

print_every = 1
save_every = 50
validate_every = 20 

use_gan = False
gan_lr = 0.001

dtype = tf.float32
