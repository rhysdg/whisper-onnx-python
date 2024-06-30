import os
from whisper import silero_model
from whisper.silero_vad.utils_vad import *

SAMPLING_RATE = 16000

#print('current')
print(os.getcwd())





wav = read_audio('data/bee.wav', sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, silero_model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)