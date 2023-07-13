# EEG Classification Repository
This is a part of my master's research project. The goal of the project is to compare the classification performance of various algorithms on EEG signals classification task. The data was collected by Tsinghua University BCI Lab during an experiment with steady-state visual evoked potentials (SSVEPs) focusing on 40 characters flickering at different frequencies [1]. You can check the details of the dataset at the link: http://bci.med.tsinghua.edu.cn/

Featured algorithms:
- Support Vector Machine
- Dense Neural Network
- Convolutional Neural Network
- LSTM Neural Network
- EEGNet implemented from Waytowitch et. al (2018)

## Repository Structure
The repository has the following structure:

main.py: The main script for running the EEG classification pipeline.
hyper_tuning/: This folder contains optimizers for tuning the model parameters.
preproces/: This folder contains scripts for preprocessing the EEG data, including filtering, standardization/normalization (min-max scaling), data import, and merging into a single dataset.
models/: This folder contains pre-trained models for EEG classification. The available models include dense neural networks, convolutional neural networks, LSTM networks, EEGNet, and SVMs.

## Dependencies
- tensorflow 2.4.1
- scikit-learn 1.2.2
- scipy 1.7.3
- numpy 1.20.3
- keras-tuner 1.3.5



## Resources

[1] Wang, Y., Chen, X., Gao, X., & Gao, S. (2016). A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.

[2] Waytowich, N., Lawhern, V. J., Garcia, J. O., Cummings, J., Faller, J., Sajda, P., & Vettel, J. M. (2018). Compact convolutional neural networks for classification of asynchronous steady-state visual evoked potentials. Journal of neural engineering, 15(6), 066031.



