import os
import librosa.core
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


this_file_path = os.path.dirname(os.path.realpath(__file__))
file_names = os.listdir('transformed_audio/audio_mix/')


### function to generate MFCC features


def log_mfcc(file_name):
    name = file_name.split('.')[-2]
    y, sr = librosa.core.load('transformed_audio/audio_mix/' + file_name)
    ps = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20)
    plt.imsave(f'features/mfcc/audio_mfcc/{name}.png', ps, origin='lower', format='png', cmap=plt.get_cmap('magma'))

p = multiprocessing.Pool()
p.map(log_mfcc,file_names)



#list(map(log_mfcc, file_names))