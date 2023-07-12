#EEG Classification Repository
This repository contains code and models for EEG classification. It serves as a framework for classifying EEG signals using various models, including dense neural networks, convolutional neural networks (CNNs), long short-term memory (LSTM) networks, EEGNet, and support vector machines (SVMs).

##Repository Structure
The repository has the following structure:

main.py: The main script for running the EEG classification pipeline.
hyper_tuning/: This folder contains optimizers for tuning the model parameters.
preproces/: This folder contains scripts for preprocessing the EEG data, including filtering, standardization/normalization (min-max scaling), data import, and merging into a single dataset.
models/: This folder contains pre-trained models for EEG classification. The available models include dense neural networks, convolutional neural networks, LSTM networks, EEGNet, and SVMs.
Getting Started
To use this repository, follow the steps below:

Clone the repository: git clone https://github.com/your_username/eeg-classification.git
Install the required dependencies: pip install -r requirements.txt
Preprocess the EEG data by running the scripts in the preproces/ folder.
Train and evaluate the models using the main.py script.
(Optional) Fine-tune the model parameters using the optimizers in the hyper_tuning/ folder.
Use the pre-trained models in the models/ folder for classification tasks.
Feel free to modify the code and adapt it to your specific requirements.

## Dependencies

##Resources
We would like to thank the contributors and researchers who have made their work available for EEG classification. Their contributions have helped advance the field and make this repository possible.




