import os
import pydub
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import numpy as np
import multiprocessing


##### function for dividing audio length into 5 second chunks

input_path = 'downloads/'

def getandexport(start):
    slice = audio[start:start + 5000]
    slice.export(f'talks/talk_{start/5000}.wav', format='wav')

def divide(input_path):
    audio = AudioSegment.from_wav()
    rng = np.arange(0,len(audio),5000)
    p = multiprocessing.Pool()
    p.map(getandexport, rng)