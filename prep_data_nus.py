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
import librosa

import config
import utils


def process_lab_file(filename, stft_len):

    lab_f = open(filename)

    phos = lab_f.readlines()
    lab_f.close()

    phonemes=[]

    for pho in phos:
        st,end,phonote=pho.split()
        st = int(np.round(float(st)/0.005804576860324892))
        en = int(np.round(float(end)/0.005804576860324892))
        if phonote=='pau' or phonote=='br' or phonote == 'sil':
            phonote='Sil'
        phonemes.append([st,en,phonote])


    strings_p = np.zeros((phonemes[-1][1],1))


    for i in range(len(phonemes)):
        pho=phonemes[i]
        value = config.phonemas_nus.index(pho[2])
        strings_p[pho[0]:pho[1]+1] = value
    return strings_p

def main():


    singers = next(os.walk(config.wav_dir_nus))[1]

    

    for singer in singers:
        sing_dir = config.wav_dir_nus+singer+'/sing/'
        read_dir = config.wav_dir_nus+singer+'/read/'
        sing_wav_files=[x for x in os.listdir(sing_dir) if x.endswith('.wav') and not x.startswith('.')]

        count = 0

        print ("Processing singer %s" % singer)
        for lf in sing_wav_files:

            audio, fs = librosa.core.load(os.path.join(sing_dir,lf), sr=config.fs)

            audio = np.float64(audio)

            if len(audio.shape) == 2:

                vocals = np.array((audio[:,1]+audio[:,0])/2)

            else: 
                vocals = np.array(audio)

            voc_stft = abs(utils.stft(vocals))

            out_feats = utils.stft_to_feats(vocals,fs)

            strings_p = process_lab_file(os.path.join(sing_dir,lf[:-4]+'.txt'), len(voc_stft))

            voc_stft, out_feats, strings_p = utils.match_time([voc_stft, out_feats, strings_p])


            hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_sing_'+lf[:-4]+'.hdf5', mode='a')

            if not  "phonemes" in hdf5_file:

                hdf5_file.create_dataset("phonemes", [voc_stft.shape[0]], int)

            hdf5_file["phonemes"][:,] = strings_p[:,0]

            hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            hdf5_file["voc_stft"][:,:] = voc_stft

            hdf5_file["feats"][:,:] = out_feats


            hdf5_file.close()

            count+=1

            utils.progress(count,len(sing_wav_files))

        read_wav_files=[x for x in os.listdir(read_dir) if x.endswith('.wav') and not x.startswith('.')]
        print ("Processing reader %s" % singer)
        count = 0

        for lf in read_wav_files:
            audio, fs = librosa.core.load(os.path.join(read_dir,lf), sr=config.fs)

            audio = np.float64(audio)

            if len(audio.shape) == 2:

                vocals = np.array((audio[:,1]+audio[:,0])/2)

            else: 
                vocals = np.array(audio)

            voc_stft = abs(utils.stft(vocals))

            out_feats = utils.stft_to_feats(vocals,fs)

            strings_p = process_lab_file(os.path.join(read_dir,lf[:-4]+'.txt'), len(voc_stft))

            voc_stft, out_feats, strings_p = utils.match_time([voc_stft, out_feats, strings_p])


            hdf5_file = h5py.File(config.voice_dir+'nus_'+singer+'_read_'+lf[:-4]+'.hdf5', mode='a')

            if not  "phonemes" in hdf5_file:

                hdf5_file.create_dataset("phonemes", [voc_stft.shape[0]], int)

            hdf5_file["phonemes"][:,] = strings_p[:,0]

            hdf5_file.create_dataset("voc_stft", voc_stft.shape, np.float32)

            hdf5_file.create_dataset("feats", out_feats.shape, np.float32)

            hdf5_file["voc_stft"][:,:] = voc_stft

            hdf5_file["feats"][:,:] = out_feats


            hdf5_file.close()

            count+=1

            utils.progress(count,len(read_wav_files))            

if __name__ == '__main__':
    main()