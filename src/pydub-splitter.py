import os
from src.Splitter import Splitter

RECORDING_FILE = os.path.join('../recordings', 'page_13_16k.wav')
SAMPLE_RATE = 16000


if __name__ == '__main__':
    splitter = Splitter(1200, 300, -55, 100, 1)
    splitter.split(RECORDING_FILE, './../recordings/segments/')


