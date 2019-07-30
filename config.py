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
backing_dir = '../ss_synthesis/backing/'


log_dir = './log/'


log_dir_phone = './log_nr_wavenet/'



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

num_f0 = 256
max_phr_len = 128
input_features = 513
output_features = 64


phonemas_nus = ['t', 'y', 'l', 'k', 'aa', 'jh', 'ae', 'ng', 'ah', 'hh', 'z', 'ey', 'f', 'uw', 'iy', 'ay', 'b', 's', 'd', 'Sil', 'p', 'n', 'sh', 'ao', 'g', 'ch', 'ih', 'eh', 'aw', 'sp', 'oy', 'th', 'w', 'ow', 'v', 'uh', 'm', 'er', 'zh', 'r', 'dh', 'ax']
phonemas_esp = ['B', 'U', 'g', 'k', 'm', 'tS', 'J', 'L', 'x', 'n', 'i', 'r', 'a', 'o', 'w', 'j', 's', 'f', 'I', 'rr', 't', 'd', 'e', 'l', 'b', 'Sil', 'u', 'D', 'p', 'G', 'T']
phonemas_cat = ['B', 'U', 'g', 'S', 'k', 'm', 'O', 'dZ', 'J', 'Z', 'tS', 'ts', 'n', 'r', 'i', 'a', 'z', 'w', 'o', 'ae', 'j', 's', 'f', 'I', 'rr', 't', 'd', 'e', 'l', 'N', 'b', 'Sil', 'u', 'L0', 'E', 'D', 'p', 'dz', 'G']
phonemas_full = ['E', 'ae', 'a', 'L', 'I', 'U', 'j', 'dZ', 'm', 'T', 'p', 'd', 'o', 't', 'f', 'Z', 'w', 'D', 'n', 'G', 'e', 'L0', 'x', 'u', 'g', 'z', 'tS', 'O', 'dz', 'J', 'i', 'ts', 'Sil', 'l', 'B', 'rr', 's', 'S', 'N', 'b', 'k', 'r']
phonemas_all = ['E', 'ae', 'hh', 'a', 'uh', 'L', 'I', 'U', 'iy', 'eh', 'j', 'm', 'dh', 'dZ', 'aa', 'T', 'th', 'ah', 'p', 'd', 'o', 'oy', 'sh', 't', 'f', 'ax', 'Z', 'w', 'D', 'uw', 'n', 'sp', 'er', 'ao', 'G', 'e', 'L0', 'y', 'ow', 'g', 'ch', 'x', 'u', 'z', 'tS', 'O', 'v', 'dz', 'zh', 'ng', 'J', 'jh', 'i', 'ts', 'Sil', 'ay', 'l', 'B', 'ey', 's', 'rr', 'S', 'N', 'b', 'aw', 'ih', 'k', 'r']

num_phos = len(phonemas_nus)

do_not_use = ['casascat_Miguel_a0057.hdf5', 'casasesp_JosepC_a0112.hdf5', 'casasesp_Miriam_a0077.hdf5', 'casasesp_Miguel_a0068.hdf5', 'casascat_Miguel_a0086.hdf5', 'casascat_Pol_a0021.hdf5', 'casascat_Miguel_a0015.hdf5', 'casasesp_Pau_a0002.hdf5', 'casascat_Miguel_a0054.hdf5', 'casascat_Miguel_a0046.hdf5', 'casascat_Miguel_a0017.hdf5', 'casascat_Miguel_a0065.hdf5', 'casascat_Miguel_a0036.hdf5', 'casascat_Pol_a0007.hdf5', 'casascat_Miguel_a0091.hdf5', 'casasesp_Miriam_a0038.hdf5', 'casascat_Miquel_a0034.hdf5', 'casascat_Miguel_a0078.hdf5', 'casascat_Miguel_a0082.hdf5', 'casascat_Sara_a0026.hdf5', 'casascat_Miquel_a0015.hdf5', 'casascat_Miguel_a0090.hdf5', 'casasesp_Miguel_a0108.hdf5', 'casascat_Miquel_a0046.hdf5', 'casasesp_Miriam_a0041.hdf5', 'casascat_Miguel_a0052.hdf5', 'casascat_Miguel_a0095.hdf5', 'casasesp_Miguel_a0032.hdf5', 'casasesp_Miriam_a0015.hdf5', 'casascat_Miguel_a0012.hdf5', 'casascat_Miguel_a0014.hdf5', 'casascat_Anna_a0086.hdf5', 'casascat_Miguel_a0058.hdf5', 'casasesp_Miguel_a0118.hdf5', 'casascat_Miguel_a0079.hdf5', 'casasesp_Pol_a0119.hdf5', 'casascat_Miquel_a0002.hdf5', 'casascat_Miquel_a0028.hdf5', 'casascat_Miguel_a0019.hdf5', 'casascat_Pol_a0049.hdf5', 'casasesp_Miriam_a0083.hdf5', 'casascat_Miquel_a0059.hdf5', 'casascat_Miquel_a0094.hdf5', 'casascat_Miquel_a0036.hdf5', 'casascat_JosepC_a0035.hdf5', 'casascat_Miguel_a0075.hdf5', 'casascat_Miguel_a0051.hdf5', 'casascat_Miquel_a0074.hdf5', 'casascat_Miguel_a0077.hdf5', 'casasesp_Miguel_a0058.hdf5', 'casascat_Miguel_a0033.hdf5', 'casascat_Miguel_a0050.hdf5', 'casasesp_JosepC_a0086.hdf5', 'casasesp_Pol_a0063.hdf5', 'casascat_Miquel_a0035.hdf5', 'casascat_Miguel_a0005.hdf5', 'casascat_Miguel_a0009.hdf5', 'casasesp_Miguel_a0024.hdf5', 'casasesp_Pol_a0116.hdf5', 'casascat_Pol_a0090.hdf5', 'casascat_JosepC_a0026.hdf5', 'casascat_Miriam_a0038.hdf5', 'casascat_Pol_a0032.hdf5', 'casascat_Miguel_a0034.hdf5', 'casascat_JosepT_a0073.hdf5', 'casascat_Miguel_a0071.hdf5', 'casascat_Miguel_a0028.hdf5', 'casascat_Miriam_a0054.hdf5', 'casascat_Miguel_a0006.hdf5', 'casasesp_Pol_a0093.hdf5', 'casascat_Miguel_a0018.hdf5', 'casascat_Miguel_a0096.hdf5', 'casascat_Miguel_a0023.hdf5', 'casasesp_Miriam_a0035.hdf5', 'casasesp_Pau_a0006.hdf5', 'casasesp_Miriam_a0080.hdf5', 'casascat_Miquel_a0001.hdf5', 'casascat_Pol_a0083.hdf5', 'casascat_Miriam_a0035.hdf5', 'casascat_Pau_a0082.hdf5', 'casascat_Miquel_a0085.hdf5', 'casascat_Miguel_a0059.hdf5', 'casascat_Miguel_a0061.hdf5', 'casascat_Miquel_a0007.hdf5', 'casasesp_Mar_a0115.hdf5', 'casascat_Miguel_a0094.hdf5', 'casasesp_Miriam_a0060.hdf5', 'casasesp_JosepT_a0102.hdf5', 'casasesp_Miguel_a0090.hdf5', 'casasesp_Miguel_a0071.hdf5', 'casascat_Miguel_a0016.hdf5', 'casascat_Pol_a0044.hdf5', 'casasesp_Pol_a0120.hdf5', 'casasesp_Miguel_a0093.hdf5', 'casascat_Miguel_a0072.hdf5', 'casasesp_JosepT_a0095.hdf5', 'casascat_Miguel_a0083.hdf5', 'casasesp_Pau_a0070.hdf5', 'casasesp_JosepT_a0106.hdf5', 'casascat_Miguel_a0041.hdf5', 'casascat_Miriam_a0057.hdf5', 'casasesp_Pol_a0051.hdf5', 'casascat_Miguel_a0047.hdf5', 'casascat_Miguel_a0081.hdf5', 'casascat_Miguel_a0037.hdf5', 'casasesp_Pau_a0029.hdf5', 'casascat_Miquel_a0081.hdf5', 'casascat_Pol_a0006.hdf5', 'casasesp_JosepT_a0019.hdf5', 'casascat_Miguel_a0084.hdf5', 'casascat_Miguel_a0004.hdf5']


singers = ['ADIZ', 'JLEE', 'JTAN', 'KENN', 'MCUR', 'MPOL', 'MPUR', 'NJAT', 'PMAR', 'SAMF', 'VKOW' ,'ZHIY']
# ,'Miriam', 'Anna', 'Pau', 'Mar', 'Pol', 'Irene', 'Sara', 'JosepC', 'JosepT', 'Clara', 'Miguel', 'Miquel']
singers_med = ['StrandOfOaks', 'MatthewEntwistle', 'AimeeNorwich', 'BrandonWebster', 'LizNelson', 'ClaraBerryAndWooldog', 'LizNelson', 'AlexanderRoss', 'Grants', 'BigTroubles', 'PortStWillow', 'PurlingHiss', 'ClaraBerryAndWooldog', 'ClaraBerryAndWooldog', 'MatthewEntwistle', 'TheSoSoGlos', 'DreamersOfTheGhetto', 'Meaxic', 'Auctioneer', 'TheDistricts', 'AlexanderRoss', 'BrandonWebster', 'HopAlong', 'FamilyBand', 'Snowmine', 'FacesOnFilm', 'ClaraBerryAndWooldog', 'TheScarletBrand', 'ClaraBerryAndWooldog', 'Meaxic', 'AClassicEducation', 'Creepoid', 'AvaLuna', 'Lushlife', 'InvisibleFamiliars', 'SweetLights', 'SecretMountains', 'HezekiahJones', 'StevenClark', 'Wolf',]
# , 'p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']
num_singers = len(singers)

vctk_speakers = ['p255', 'p285', 'p260', 'p247', 'p266', 'p364', 'p265', 'p233', 'p341', 'p347', 'p243', 'p300', 'p284', 'p283', 'p239', 'p269', 'p236', 'p281', 'p293', 'p241', 'p240', 'p259', 'p244', 'p271', 'p294', 'p287', 'p263', 'p261', 'p334', 'p323', 'p227', 'p282', 'p313', 'p248', 'p277', 'p297', 'p314', 'p250', 'p335', 'p374', 'p315', 'p304', 'p298', 'p288', 'p234', 'p310', 'p262', 'p329', 'p251', 'p330', 'p339', 'p312', 'p256', 'p258', 'p231', 'p249', 'p317', 'p301', 'p292', 'p306', 'p360', 'p272', 'p316', 'p311', 'p308', 'p318', 'p229', 'p245', 'p361', 'p232', 'p257', 'p264', 'p237', 'p226', 'p246', 'p351', 'p270', 'p228', 'p286', 'p267', 'p376', 'p333', 'p252', 'p253', 'p345', 'p254', 'p278', 'p336', 'p268', 'p363', 'p326', 'p303', 'p362', 'p295', 'p274', 'p273', 'p305', 'p343', 'p276', 'p275', 'p225', 'p238', 'p302', 'p279', 'p307', 'p299', 'p340', 'p280', 'p230']

split = 0.9

augment = True
aug_prob = 0.5

noise_threshold = 0.4 #0.7 for the unnormalized features
pred_mode = 'all'

# Hyperparameters
num_epochs = 2500

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
