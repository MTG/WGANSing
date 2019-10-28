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


    voc_list = [x for x in os.listdir(config.voice_dir) if
                x.endswith('.hdf5') and x.startswith('nus') and not x == 'nus_MCUR_sing_04.hdf5' and not x == 'nus_ADIZ_read_01.hdf5'
                and not x == 'nus_JLEE_sing_05.hdf5' and not x == 'nus_JTAN_read_07.hdf5']




    val_list = ['nus_MCUR_sing_04.hdf5', 'nus_ADIZ_read_01.hdf5', 'nus_JLEE_sing_05.hdf5','nus_JTAN_read_07.hdf5' ]

    # import pdb;pdb.set_trace()

    stat_file = h5py.File(config.stat_dir+'stats.hdf5', mode='r')

    max_feat = np.array(stat_file["feats_maximus"])
    min_feat = np.array(stat_file["feats_minimus"])

    stat_file.close()


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



            feats = np.array(voc_file['feats'])

            f0 = feats[:,-2]

            med = np.median(f0[f0 > 0])

            f0[f0==0] = med

            f0_nor = (f0 - min_feat[-2])/(max_feat[-2]-min_feat[-2])
            

            feats = (feats-min_feat)/(max_feat-min_feat)

            feats[:,-2] = f0_nor


            if voc_to_open.startswith('nus'):
                if not  "phonemes" in voc_file:
                    print(voc_file)
                    Flag = False
                else: 
                    Flag = True
                    pho_target = np.array(voc_file["phonemes"])
                    singer_name = voc_to_open.split('_')[1]
                    singer_index = config.singers.index(singer_name)
            else:
                Flag = False


            for j in range(config.samples_per_file):
                    voc_idx = np.random.randint(0,len(feats)-config.max_phr_len)
                    targets_f0_1.append(f0_nor[voc_idx:voc_idx+config.max_phr_len])
                    if Flag:
                        pho_targs.append(pho_target[voc_idx:voc_idx+config.max_phr_len])
                        targets_singers.append(singer_index)

                    feats_targs.append(feats[voc_idx:voc_idx+config.max_phr_len])

        targets_f0_1 = np.expand_dims(np.array(targets_f0_1), -1)


        feats_targs = np.array(feats_targs)

        assert feats_targs.max()<=1.0 and feats_targs.min()>=0.0

        yield feats_targs, targets_f0_1, np.array(pho_targs), np.array(targets_singers)





def get_stats():
    voc_list = [x for x in os.listdir(config.voice_dir) if x.endswith('.hdf5') and x.startswith('nus')  and not x.startswith('nus_KENN') ]

    max_feat = np.zeros(66)
    min_feat = np.ones(66)*1000

    max_voc = np.zeros(513)
    min_voc = np.ones(513)*1000

    max_mix = np.zeros(513)
    min_mix = np.ones(513)*1000    

    for voc_to_open in voc_list:

        voc_file = h5py.File(config.voice_dir+voc_to_open, "r")

        feats = np.array(voc_file['feats'])

        f0 = feats[:,-2]

        med = np.median(f0[f0 > 0])

        f0[f0==0] = med

        feats[:,-2] = f0

        maxi_voc_feat = np.array(feats).max(axis=0)

        for i in range(len(maxi_voc_feat)):
            if maxi_voc_feat[i]>max_feat[i]:
                max_feat[i] = maxi_voc_feat[i]

        mini_voc_feat = np.array(feats).min(axis=0)

        for i in range(len(mini_voc_feat)):
            if mini_voc_feat[i]<min_feat[i]:
                min_feat[i] = mini_voc_feat[i]   



    hdf5_file = h5py.File(config.stat_dir+'stats.hdf5', mode='w')

    hdf5_file.create_dataset("feats_maximus", [66], np.float32) 
    hdf5_file.create_dataset("feats_minimus", [66], np.float32)   
  

    hdf5_file["feats_maximus"][:] = max_feat
    hdf5_file["feats_minimus"][:] = min_feat

    hdf5_file.close()



def main():
    # gen_train_val()
    get_stats()
    gen = data_gen('Train', sec_mode = 0)
    while True :
        start_time = time.time()
        feats_targs, targets_f0_1, pho_targs, targets_singers = next(gen)
        print(time.time()-start_time)



        import pdb;pdb.set_trace()


if __name__ == '__main__':
    main()