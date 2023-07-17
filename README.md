# Readme
This repository contains the Python code that was used to perform the experiments.
The code is organized in the following way:
- The `words` folder contains the dataset of words
- The `src` folder contains the all the Python code and the data files that with the results of our experiments

Python 3.10 was used to run all the code files. Please make sure the following packages are installed:
`pandas`, `librosa`, `seaborn`, `torch`, `pydub`, `transformers`.

# Source files
- `src/word-analysis.py` contains the primary code that performs the experiments. This script outputs a datafile for a
specific speaker
- `src/pydub-splitter.py` contains the code that split an audio recording into individual files for each word.
- `src/analysis.py` contains the code that imports the data files and calculates the correlation coefficients
- `src/table-generation.py` code to generate the tables in the report, based on the data files
- `src/validation.py` code to validate the similarity measure between two recordings of the same word type

NB, other files in the `src` folder contain helper functions and classes that are used by the files mentioned above.


