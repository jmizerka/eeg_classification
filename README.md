# EEG Classification Repository
This is a part of my master's research project. The goal of the project is to compare the classification performance of various algorithms on EEG signals classification task. The data was collected by Tsinghua University BCI Lab during an experiment with steady-state visual evoked potentials (SSVEPs) focusing on 40 characters flickering at different frequencies. 


## Repository Structure
The repository has the following structure:

main.py: The main script for running the EEG classification pipeline.
hyper_tuning/: This folder contains optimizers for tuning the model parameters.
preproces/: This folder contains scripts for preprocessing the EEG data, including filtering, standardization/normalization (min-max scaling), data import, and merging into a single dataset.
models/: This folder contains pre-trained models for EEG classification. The available models include dense neural networks, convolutional neural networks, LSTM networks, EEGNet, and SVMs.

## Getting Started
To use this repository, follow the steps below:

Clone the repository: git clone https://github.com/your_username/eeg-classification.git
Install the required dependencies: pip install -r requirements.txt
Feel free to modify the code and adapt it to your specific requirements.

## Dependencies

## Resources
We would like to thank the contributors and researchers who have made their work available for EEG classification. Their contributions have helped advance the field and make this repository possible.




