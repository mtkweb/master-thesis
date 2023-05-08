import os
from src.Splitter import Splitter

RECORDING_FILE = os.path.join('../recordings', 'page_7_16k.wav')
SAMPLE_RATE = 16000


if __name__ == '__main__':
    splitter = Splitter(600, 300, -55, 100, 1)
    splitter.split(RECORDING_FILE, './../recordings/segments/')


