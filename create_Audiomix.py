
import os
import pydub
import random
import shutil
import numpy as np
import time as timer
import datetime as dt
import multiprocessing
from pydub import AudioSegment
import matplotlib.pyplot as plt

##This function mixes the AED audio length with the background sounds. No volume reduction is implemented here

this_file_path = os.path.dirname(os.path.realpath(__file__))

sample_path = this_file_path + fr'/raw_data/class//'
audio_path = this_file_path + fr'/transformed_audio/audiomix22//'
bg_path = this_file_path + fr'/raw_data/backgrounds//'
samples = os.listdir(sample_path)
bgs = os.listdir(bg_path)

########################## create lists of audio segments #####################################

samples = dict(
    enumerate(list(map(lambda i: AudioSegment.from_file(sample_path + samples[i], format="wav"),[22]))))
def mix(bg):
    numdict = dict(zip([0], [22]))
    for sample in samples:
        seg = AudioSegment.from_file(bg_path + bg, format="wav")
        seg = seg.set_channels(1)
        diff = samples[sample].dBFS - seg.dBFS
        new_bg = seg.apply_gain(diff)
        end = len(seg) - len(samples[sample])
        pos = random.randint(0, end - 1)
        out = new_bg.overlay(samples[sample], position=pos)
        out.export(audio_path + str(numdict[sample]) + '_' + str(bg), format="wav")



p = multiprocessing.Pool()
p.map(mix, bgs)

# plt.hist(np.abs(df))
# print(np.mean(np.abs(df)))
#list(map(mix, bgs))