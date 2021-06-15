import os
import librosa.core
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing



### function to generate log-scaled mel spectrograms features

this_file_path = os.path.dirname(os.path.realpath(__file__))
file_names = os.listdir('transformed_audio/audiomix22/')

def log_mfcc(file_name):
    name = file_name.split('.')[-2]
    y, sr = librosa.core.load('transformed_audio/audiomix22/' + file_name)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    ps_db = librosa.amplitude_to_db(ps, ref=np.max)
    plt.imsave(f'features/log_mel/log_mel spectogram22/{name}.png', ps_db, origin='lower', format='png', cmap=plt.get_cmap('magma'))

p = multiprocessing.Pool()
p.map(log_mfcc,file_names)

#list(map(log_mfcc, file_names))