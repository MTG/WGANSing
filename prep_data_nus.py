# from __future__ import division
import os,re
import collections
import soundfile as sf
import numpy as np
from scipy.stats import norm
import pyworld as pw
import matplotlib.pyplot as plt
import sys
import h5py

import config
import utils


def main():

    # maximus=np.zeros(66)
    # minimus=np.ones(66)*1000
    singers = next(os.walk(config.wav_dir_nus))[1]
    # singers = [x for x in singers if x not in["VKOW","SAMF","MPUR","JLEE","KENN"]]
    # import pdb;pdb.set_trace()

    # phonemas = set([])
    

    for singer in singers:
        sing_dir = config.wav_dir_nus+singer+'/sing/'
        read_dir = config.wav_dir_nus+singer+'/read/'
        sing_wav_files=[x for x in os.listdir(sing_dir) if x.endswith('.wav') and not x.startswith('.')]

        count = 0

        print ("Processing singer %s" % singer)
        for lf in sing_wav_files:
        # print(lf)
            # if not os.path.exists(config.voice_dir+'nus_'+singer+'_sing_'+lf[:-4]+'.hdf5'):

            audio,fs = sf.read(os.path.join(sing_dir,lf))
            if fs !=config.fs:
                command = "ffmpeg -y -i "+os.path.join(sing_dir,lf)+" -ar "+str(config.fs)+" "+os.path.join(sing_dir,lf)
                os.system(command)
            audio,fs = sf.read(os.path.join(sing_dir,lf))

            if len(audio.shape) == 2:

                vocals = np.array((audio[:,1]+audio[:,0])/2)

            else: 
                vocals = np.array(audio)

            voc_stft = abs(utils.stft(vocals))



            lab_f = open(os.path.join(sing_dir,lf[:-4]+'.txt'))
            # note_f=open(in_dir+lf[:-4]+'.notes')
            phos = lab_f.readlines()
            lab_f.close()

            phonemes=[]

            for pho in phos:
                st,end,phonote=pho.split()
                # import pdb;pdb.set_trace()
                st = int(np.round(float(st)/0.005804576860324892))
                en = int(np.round(float(end)/0.005804576860324892))
                if phonote=='pau' or phonote=='br':
                    phonote='sil'
                phonemes.append([st,en,phonote])
                # phonemas.add(phonote)

            # div_fac = float(end)/len(voc_stft)

            # for i in range(len(phonemes)):
            #     phonemes[i][0] = int(float(phonemes[i][0])/div_fac)
            #     phonemes[i][1] = int(float(phonemes[i][1])/div_fac)

            # import pdb;pdb.set_trace()

            phonemes[-1][1] = len(voc_stft)




            strings_p = np.zeros(phonemes[-1][1])

            for i in range(len(phonemes)):
                pho=phonemes[i]
                value = config.phonemas.index(pho[2])
                strings_p[pho[0]:pho[1]+1] = value

            # import pdb;pdb.set_trace()

            if not len(strings_p) == len(voc_stft):
                import pdb;pdb.set_trace()


            # out_feats = utils.stft_to_feats(vocals,fs)


            # if not out_feats.shape[0]==voc_stft.shape[0] :
            #     if out_feats.shape[0]<voc_stft.shape[0]:
            #         while out_feats.shape[0]<voc_stft.shape[0]:
            #             out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))
            #     elif out_feats.shape[0]<voc_stft.shape[0]:
            #         print("You are an idiot")

            # assert out_feats.shape[0]==voc_stft.shape[0]

            hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_sing_'+lf[:-4]+'.hdf5', mode='a')
            # import pdb;pdb.set_trace()

            if not  "phonemes" in hdf5_file:

                hdf5_file.create_dataset("phonemes", [voc_stft.shape[0]], int)

            hdf5_file["phonemes"][:,] = strings_p

            # hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            # hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            # hdf5_file["voc_stft"][:,:] = voc_stft

            # hdf5_file["feats"][:,:] = out_feats


            hdf5_file.create_dataset("voc_stft_phase", voc_stft_phase.shape, np.float32)

            # hdf5_file["phonemes"][:,] = strings_p

            # hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            # hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            # hdf5_file["voc_stft"][:,:] = voc_stft
            hdf5_file["voc_stft_phase"][:,:] = voc_stft_phase

            hdf5_file.close()






            count+=1

            utils.progress(count,len(sing_wav_files))


        read_wav_files=[x for x in os.listdir(read_dir) if x.endswith('.wav') and not x.startswith('.')]
        print ("Processing reader %s" % singer)
        count = 0
        if not singer == 'KENN':
            for lf in sing_wav_files:

                # if not os.path.exists(config.voice_dir+'nus_'+singer+'_read_'+lf[:-4]+'.hdf5'):
            # print(lf)
                audio,fs = sf.read(os.path.join(read_dir,lf))
                if fs !=config.fs:
                    command = "ffmpeg -y -i "+os.path.join(read_dir,lf)+" -ar "+str(config.fs)+" "+os.path.join(read_dir,lf)
                    os.system(command)
                audio,fs = sf.read(os.path.join(read_dir,lf))

                if len(audio.shape) == 2:

                    vocals = np.array((audio[:,1]+audio[:,0])/2)

                else: 
                    vocals = np.array(audio)

                voc_stft = abs(utils.stft(vocals))

                lab_f = open(os.path.join(read_dir,lf[:-4]+'.txt'))
            # note_f=open(in_dir+lf[:-4]+'.notes')
                phos = lab_f.readlines()
                lab_f.close()

                phonemes=[]

                for pho in phos:
                    st,end,phonote=pho.split()
                    # import pdb;pdb.set_trace()
                    st = int(np.round(float(st)/0.005804576860324892))
                    en = int(np.round(float(end)/0.005804576860324892))
                    if phonote=='pau' or phonote=='br':
                        phonote='sil'
                    phonemes.append([st,en,phonote])
                    # phonemas.add(phonote)

                phonemes[-1][1] = len(voc_stft)

                # div_fac = float(end)/len(voc_stft)

                # for i in range(len(phonemes)):
                #     phonemes[i][0] = int(float(phonemes[i][0])/div_fac)
                #     phonemes[i][1] = int(float(phonemes[i][1])/div_fac)


                strings_p = np.zeros(phonemes[-1][1])
                for i in range(len(phonemes)):
                    pho=phonemes[i]
                    # if singer == 'KENN':
                        # import pdb;pdb.set_trace()
                    value = config.phonemas.index(pho[2])
                    strings_p[pho[0]:pho[1]+1] = value

                if not len(strings_p) == len(voc_stft):
                    import pdb;pdb.set_trace()

                    # out_feats = utils.stft_to_feats(vocals,fs)

                    

                    # if not out_feats.shape[0]==voc_stft.shape[0] :
                    #     if out_feats.shape[0]<voc_stft.shape[0]:
                    #         while out_feats.shape[0]<voc_stft.shape[0]:
                    #             out_feats = np.concatenate(((out_feats,np.zeros((1,out_feats.shape[1])))))
                    #     elif out_feats.shape[0]<voc_stft.shape[0]:
                    #         print("You are an idiot")

                    # assert out_feats.shape[0]==voc_stft.shape[0] 

                hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_read_'+lf[:-4]+'.hdf5', mode='a')

                if not  "phonemes" in hdf5_file:

                    hdf5_file.create_dataset("phonemes", [voc_stft.shape[0]], int)

                hdf5_file["phonemes"][:,] = strings_p

                # hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

                # hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

                # hdf5_file["voc_stft"][:,:] = voc_stft

                # hdf5_file["feats"][:,:] = out_feats

                hdf5_file.close()

                count+=1

                utils.progress(count,len(read_wav_files))
    import pdb;pdb.set_trace()        


if __name__ == '__main__':
    main()