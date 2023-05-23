# Verifiable Federated Learning

This repository implements the experiments using Verification Via Commitments (VVC)
in the Verifiable Federated Learning paper: https://openreview.net/pdf?id=0HIa3HIyIHN

To run the code: 

`python main.py`

The configuration for the experiments is specified in the configuration dictionary found in main.py

To obtain the femnist dataset, clone this repo: https://github.com/TalwalkarLab/leaf , by default the paths in the code 
point to one directory higher eg (../leaf/).

Then go to `leaf/data/femnist/` and run `./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --smplseed 1549786595 --spltseed 1549786796`
following the original femnist paper setup.
