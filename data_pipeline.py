import numpy as np
import os
import time
import h5py

import matplotlib.pyplot as plt
import collections
import config
import utils


def gen_train_val():
    mix_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and x.startswith('med') ]

    train_list = mix_list[:int(len(mix_list)*config.split)]

    val_list = mix_list[int(len(mix_list)*config.split):]

    utils.list_to_file(val_list,config.log_dir+'val_files.txt')

    utils.list_to_file(train_list,config.log_dir+'train_files.txt')




def data_gen(mode = 'Train', sec_mode = 0):

    # import pdb;pdb.set_trace()



    # voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('ikala') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    # voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus')  and not x.startswith('nus_KENN') and not x == 'nus_MCUR_sing_04.hdf5' and not x == 'nus_MCUR_read_04.hdf5']

    voc_list = [x for x in os.listdir(config.voice_dir) if
                x.endswith('.hdf5') and x.startswith('nus') and not x == 'nus_MCUR_sing_04.hdf5' and not x == 'nus_ADIZ_read_01.hdf5'
                and not x == 'nus_JLEE_sing_05.hdf5' and not x == 'nus_JTAN_read_07.hdf5' and not x.startswith('nus_KENN_read')]
                # and not x == 'nus_MPOL_read_11.hdf5' and not x == 'nus_MPUR_sing_16.hdf5'
                    # 'nus_KENN') and not x == 'nus_MCUR_sing_04.hdf5' and not x == 'nus_MCUR_read_04.hdf5']

    # voc_list = [x for x in voc_list if x.split('_')[2] == 'sing']

    mix_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('nus_KENN') ]

    # import pdb;pdb.set_trace()

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    # mix_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus')  and not x.startswith('nus_KENN_read')]

    # all_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('nus') and not x.startswith('vctk')]

    all_list = mix_list

    # val_list = mix_list[int(len(mix_list)*config.split):]

    # import pdb;pdb.set_trace()

    # train_list = mix_list

    val_list = ['nus_MCUR_sing_04.hdf5', 'nus_ADIZ_read_01.hdf5', 'nus_JLEE_sing_05.hdf5','nus_JTAN_read_07.hdf5' ]

    # import pdb;pdb.set_trace()

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])
    max_voc = np.array(stat_file["voc_stft_maximus"])
    min_voc = np.array(stat_file["voc_stft_minimus"])
    max_back = np.array(stat_file["back_stft_maximus"])
    min_back = np.array(stat_file["back_stft_minimus"])
    max_mix = np.array(max_voc)+np.array(max_back)
    stat_file.close()

    # import pdb;pdb.set_trace()
    # min_mix = 


    max_files_to_process = int(config.batch_size/config.samples_per_file)

    if mode == "Train":
        num_batches = config.batches_per_epoch_train
        if sec_mode == 0:
            file_list = voc_list

    else: 
        num_batches = config.batches_per_epoch_val
        file_list = val_list

    for k in range(num_batches):
        if sec_mode == 1:
            if np.random.rand(1)<config.aug_prob:
                file_list = voc_list
            else:
                file_list = voc_list
        

        feats_targs = []
        targets_f0_1 = []
        targets_singers = []
        pho_targs = []

        # start_time = time.time()
        if k == num_batches-1 and mode =="Train":
            file_list = voc_list

        for i in range(max_files_to_process):


            voc_index = np.random.randint(0,len(file_list))
            voc_to_open = file_list[voc_index]


            voc_file = h5py.File(config.voice_dir+voc_to_open, "r")


            # voc_stft = np.array(voc_file['voc_stft'])

            # voc_stft_phase = np.array(voc_file['voc_stft_phase'])

            # import pdb;pdb.set_trace()

            # plt.imshow(np.log(voc_stft.T), aspect = 'auto', origin = 'lower')
            # plt.show()

            feats = np.array(voc_file['feats'])

            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med



            # import pdb;pdb.set_trace()

            f0_midi = np.rint(f0) - 30

            f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])

            # feats[:,:15] = (feats[:,:15]-min(min_feat[:15]))/(max(max_feat[:15])-min(min_feat[:15]))

            # feats[:,15:60] = (feats[:,15:60]-min(min_feat[15:60]))/(max(max_feat[15:60])-min(min_feat[15:60]))

            # feats[:,60:64] = (feats[:,60:64]-min(min_feat[60:64]))/(max(max_feat[60:64])-min(min_feat[60:64]))

            # plt.imshow(feats[:,:60].T, aspect = 'auto', origin = 'lower')

            # plt.show()

            # import pdb;pdb.set_trace()

            

            feats = (feats-min_feat)/(max_feat-min_feat)

            feats[:,-2] = f0_nor



            # back_index = np.random.randint(0,len(back_list))

            # back_to_open = back_list[back_index]

            # back_file = h5py.File(config.backing_dir+back_to_open, "r")
            if voc_to_open.startswith('nus'):
                if not  "phonemes" in voc_file:
                    print(voc_file)
                    Flag = False
                else: 
                    Flag = True
                    pho_target = np.array(voc_file["phonemes"])
                    haha = np.diff(pho_target)
                    # baba = haha/41
                    # baba[baba==0]=np.nan
                    # baba = np.nan_to_num(baba)
                    # baba = np.pad(baba, (0,1), mode = 'constant')
                    # import pdb;pdb.set_trace()
                    singer_name = voc_to_open.split('_')[1]
                    singer_index = config.singers.index(singer_name)
            else:
                Flag = False


            # print("Backing file: %s" % back_file)

            # back_stft = back_file['back_stft']


            for j in range(config.samples_per_file):
                    voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)
                    # bac_idx = np.random.randint(0,len(back_stft)-config.max_phr_len)
                    # mix_stft = voc_stft[voc_idx:voc_idx+config.max_phr_len,:]
                    # *np.clip(np.random.rand(1),0.5,0.9) + back_stft[bac_idx:bac_idx+config.max_phr_len,:]*np.clip(np.random.rand(1),0.0,0.9)+ np.random.rand(config.max_phr_len,config.input_features)*np.clip(np.random.rand(1),0.0,config.noise_threshold)
                    targets_f0_1.append(f0_nor[voc_idx:voc_idx+config.max_phr_len])
                    if Flag:
                        pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])
                        # pho_targs_2.append(baba[voc_idx:voc_idx+config.max_phr_len])
                        targets_singers.append(singer_index)
                    # inputs.append(mix_stft)
                    feats_targs.append(feats[voc_idx:voc_idx+config.max_phr_len])

        targets_f0_1 = np.array(targets_f0_1)

        # targets_f0_2 = np.array(targets_f0_2)
        
        # inputs = np.array(inputs)

        feats_targs = np.array(feats_targs)



        if Flag:

            yield feats_targs, targets_f0_1, np.array(pho_targs), np.array(targets_singers)
        else:
            yield inputs_norm, feats_targs, targets_f0_1, targets_f0_2, None, None, Flag




def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus')  and not x.startswith('nus_KENN') ]

    back_list = [x for x in os.listdir(config.backing_dir) if x.endswith('.hdf5') and not x.startswith('._') and not x.startswith('mir') and not x.startswith('med')]

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        voc_stft = voc_file['voc_stft']

        feats = np.array(voc_file['feats'])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_voc[i]:
                max_voc[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_voc[i]:
                min_voc[i] = mini_voc_stft[i]

        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i]   

    for voc_to_open in back_list:

        voc_file = h5py.File(config.backing_dir+voc_to_open, "r")

        voc_stft = voc_file["back_stft"]

        maxi_voc_stft = np.array(voc_stft).max(axis=0)

        # if np.array(feats).min()<0:
        #     import pdb;pdb.set_trace()

        for i in range(len(maxi_voc_stft)):
            if maxi_voc_stft[i]>max_mix[i]:
                max_mix[i] = maxi_voc_stft[i]

        mini_voc_stft = np.array(voc_stft).min(axis=0)

        for i in range(len(mini_voc_stft)):
            if mini_voc_stft[i]<min_mix[i]:
                min_mix[i] = mini_voc_stft[i]

    hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   
    hdf5_file.create_dataset("voc_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("voc_stft_minimus", [513], np.float32)   
    hdf5_file.create_dataset("back_stft_maximus", [513], np.float32) 
    hdf5_file.create_dataset("back_stft_minimus", [513], np.float32)   

    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat
    hdf5_file["voc_stft_maximus"][:] = max_voc
    hdf5_file["voc_stft_minimus"][:] = min_voc
    hdf5_file["back_stft_maximus"][:] = max_mix
    hdf5_file["back_stft_minimus"][:] = min_mix

    # import pdb;pdb.set_trace()

    hdf5_file.close()


def get_stats_phonems():

    phon=collections.Counter([])

    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus') and not x.startswith('nus_KENN') and not x == 'nus_MCUR_read_17.hdf5']

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")
        pho_target = np.array(voc_file["phonemes"])
        phon += collections.Counter(pho_target)
    phonemas_weights = np.zeros(41)
    for pho in phon:
        phonemas_weights[pho] = phon[pho]

    phonemas_above_threshold = [config.phonemas[x[0]] for x in np.argwhere(phonemas_weights>70000)]

    pho_order = phonemas_weights.argsort()

    # phonemas_weights = 1.0/phonemas_weights
    # phonemas_weights = phonemas_weights/sum(phonemas_weights)
    import pdb;pdb.set_trace()


def main():
    # gen_train_val()
    # get_stats()
    gen = data_gen('Train', sec_mode = 0)
    while True :
        start_time = time.time()
        inputs, feats_targs, targets_f0_1, targets_f0_2, pho_targs, targets_singers, Flag = next(gen)
        print(time.time()-start_time)

    #     plt.subplot(411)
    #     plt.imshow(np.log(1+inputs.reshape(-1,513).T),aspect='auto',origin='lower')
    #     plt.subplot(412)
    #     plt.imshow(targets.reshape(-1,66)[:,:64].T,aspect='auto',origin='lower')
    #     plt.subplot(413)
    #     plt.plot(targets.reshape(-1,66)[:,-2])
    #     plt.subplot(414)
    #     plt.plot(targets.reshape(-1,66)[:,-1])

    #     plt.show()
    #     # vg = val_generator()
    #     # gen = get_batches()


        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()