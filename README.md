# EEG_wavenet
This repository contains the pipeline for creating a wavenet CNN to classify EEG data


# WaveNet EEG Classifier

## Abstract
WaveNet's Precision in EEG Classification


This study introduces a WaveNet-based deep learning model designed to automate the classification of Intracranial Electroencephalography (iEEG) signals into physiological activity, pathological/epileptic activity, power-line noise, and other non-cerebral artifacts categories. Traditional methods for iEEG signal classification, which rely on expert visual review, are becoming increasingly impractical due to the growing complexity and volume of iEEG recordings. Leveraging a publicly available annotated dataset from Mayo Clinic and St. Anne’s University Hospital, the WaveNet model was trained, validated, and tested on 209,231 samples with a 70/20/10 % split. The model achieved a classification accuracy exceeding previous non specialized CNN and LSTM-based approaches, and was benchmarked against a Temporal Convolutional Network (TCN) baseline. Notably, the model achieves high discrimination of noise and artifact classes (precision = 0.98 and 
, respectively). Although, classification between physiological and pathological signals exhibits a modest but clinically interpretable overlap, with F1-scores of 0.96 and 0.90 and 175 and 272 cross-class false positives, respectively. Reflecting inherent clinical overlap. WaveNet’s architecture, originally developed for raw audio synthesis, is well-suited for iEEG data due to its use of dilated causal convolutions and residual connections, enabling it to capture both fine-grained and long-range temporal dependencies. The research also details the preprocessing pipeline, including dynamic dataset partitioning, the use of focal loss to combat class imbalances and normalization steps that support model high performance. While results demonstrate strong in-distribution performance, generalizability across datasets and clinical settings has yet to be established.


Information:
See "04-09-2025TCN" for the up to date code.
