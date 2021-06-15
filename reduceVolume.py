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

sample_path = this_file_path + fr'/raw_data/class//'
audio_path = this_file_path + fr'/transformed_audio/audiomix22//'
background_path = this_file_path + fr'/raw_data/backgrounds//'
samples = os.listdir(sample_path)
backgrounds = os.listdir(background_path)

# CREATE LISTS OF AUDIO SEGMENTS
samples = dict(
    enumerate(list(map(lambda i: AudioSegment.from_file(sample_path + samples[i], format="wav"), [22]))))

# transform (gain) get summary statistsics of transformed dBFS difference
def mix(background):
    numdict = dict(zip([0], [22]))
    for sample in samples:
        seg = AudioSegment.from_file(background_path + background, format="wav")
        seg = seg.set_channels(1)
        diff = samples[sample].dBFS - seg.dBFS
        new_background = seg.apply_gain(diff)
        sample1 = samples[sample].apply_gain(-10) #reduce to 15 to generate lower loundness
        end = len(seg) - len(sample1)
        pos = random.randint(0, end - 1)
        out = new_background.overlay(sample1, position=pos)
        out.export(audio_path + str(numdict[sample]) + '__' + str(background), format="wav")

p = multiprocessing.Pool()
p.map(mix, backgrounds)