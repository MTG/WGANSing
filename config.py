import numpy as np
import tensorflow as tf

ikala_gt_fo_dir = '../datasets/iKala/PitchLabel/'
wav_dir = '../datasets/iKala/Wavfile/'
wav_dir_nus = '../datasets/nus-smc-corpus_48/'
wav_dir_mus = '../datasets/musdb18/train/'
wav_dir_mir = '../datasets/MIR1k/'
wav_dir_med = '../datasets/medleydB/'
wav_dir_vctk = '../datasets/VCTK/VCTK_files/VCTK-Corpus/wav48/'
wav_dir_vctk_lab = '../datasets/VCTK/VCTK_files/VCTK-Corpus/forPritish/'


voice_dir = '../ss_synthesis/voice/'
backing_dir = './backing/'

log_dir = '../ss_synthesis/log_Wassertien1_L1_vs_imagegan_NN/'

# log_dir = './log_cross_GANandL1/'


# log_dir = './log/'
log_dir_m1 = './log_m1_old/'
# log_dir = './log_mfsc_6_best_so_far/'
data_log = './log/data_log.log'


dir_npy = './data_npy/'
stat_dir = './stats/'
h5py_file_train = './data_h5py/train.hdf5'
h5py_file_val = './data_h5py/val.hdf5'
val_dir = './val_dir_synth/'

in_mode = 'mix'
norm_mode_out = "max_min"
norm_mode_in = "max_min"

voc_ext = '_voc_stft.npy'
feats_ext = '_synth_feats.npy'

f0_weight = 10
max_models_to_keep = 100
f0_threshold = 1

def get_teacher_prob(epoch):
    if epoch < 500:
        return 0.95
    elif epoch < 1000:
        return 0.75
    else:
        return 0.55



phonemas = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh', 'ax']

# phonemas_weights = [1.91694048e-03, 3.13983774e-03, 2.37052131e-03, 3.88045684e-03,
#        1.41986299e-03, 1.12648565e-02, 3.30023014e-03, 5.00321922e-03,
#        5.87243483e-04, 4.37742526e-03, 1.97692391e-02, 9.70398460e-04,
#        3.21655616e-03, 1.35928733e-03, 5.93524695e-04, 5.65175305e-04,
#        6.80717094e-03, 1.10015365e-03, 4.38444037e-03, 1.70260315e-04,
#        8.75424154e-03, 1.16470447e-03, 8.02211731e-03, 1.75907101e-03,
#        8.74937266e-03, 1.27897334e-02, 1.20364751e-03, 8.12214268e-04,
#        3.27038554e-03, 2.33057364e-01, 1.74212315e-02, 2.22823967e-02,
#        2.25256804e-03, 8.29516836e-04, 6.36704322e-03, 1.80612767e-02,
#        2.42758721e-03, 1.96789743e-03, 5.61834716e-01, 2.38381211e-03,
#        8.39230304e-03]

phonemas_weights = np.ones(42)*0.9
phonemas_weights[19] = 0.5
phonemas_weights[15] = 0.75
phonemas_weights[8] = 0.75
phonemas_weights[14] = 0.75
phonemas_weights[27] = 0.8
phonemas_weights[33] = 0.8
phonemas_weights[11] = 0.8
phonemas_weights[17] = 0.85
phonemas_weights[21] = 0.85
phonemas_weights[13] = 0.85
phonemas_weights[4] = 0.85
phonemas_weights[34] = 0.95
phonemas_weights[16] = 0.95
phonemas_weights[22] = 0.95
phonemas_weights[40] = 0.95
phonemas_weights[24] = 0.95
phonemas_weights[20] = 0.95
phonemas_weights[5] = 0.95
phonemas_weights[25] = 0.95
phonemas_weights[30] = 0.95
phonemas_weights[35] = 0.95
phonemas_weights[10] = 0.95
phonemas_weights[31] = 0.95
phonemas_weights[29] = 1.0
phonemas_weights[38] = 1.0

val_files = 30

singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY']
# , 'p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']


vctk_speakers = ['p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']

split = 0.9

augment = True
aug_prob = 0.5

noise_threshold = 0.005 #0.7 for the unnormalized features
pred_mode = 'all'

# Hyperparameters
num_epochs = 4000
num_epochs_m1 = 2000
batches_per_epoch_train = 100
batches_per_epoch_val = 10
batches_per_epoch_val_m1 = 300
batch_size = 30
samples_per_file = 5
max_phr_len = 128
input_features = 513

first_embed = 256


lamda = 0.001

lstm_size = 64

output_features = 66

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

wavenet_layers = 7
rec_field = 2**wavenet_layers
wavenet_filters = 64

print_every = 1
save_every = 10

use_gan = False
gan_lr = 0.001

dtype = tf.float32
