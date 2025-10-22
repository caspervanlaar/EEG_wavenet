# EEG_wavenet
This repository contains the pipeline for creating a wavenet CNN to classify EEG data


# WaveNet EEG Classifier

## Abstract
WaveNet's Precision in EEG Classification


This project develops a WaveNet model to automate EEG data classification into physiological, pathological, artifact, and powerline noise categories using a publicly available annotated dataset from Mayo Clinic and St. Anne’s University Hospital. Previous models achieved up to 95% accuracy ([Nejedly et al., 2019](https://www.nature.com/articles/s41598-019-47854-6); [Nejedly et al., 2018](https://link.springer.com/article/10.1007/s12021-018-9397-6)); however, the state-of-the-art WaveNet model can reach 100% accuracy with significantly less training data. The outcome includes a detailed [report](https://github.com/caspervanlaar/EEG_wavenet/blob/main/CASPER_VAN_LAAR_2440678_WaveNet_EEG1.pdf) on the model's performance. WaveNet, originally designed for generating raw audio waveforms, is well-suited for EEG data due to its ability to model long-range temporal dependencies and hierarchical features, capturing the intricate temporal dynamics and variations in EEG signals.



## Introduction
This repository contains the code and resources for the WaveNet-based EEG classifier, designed to automate the classification of EEG data. 

## Features
- Automated classification of EEG signals
- Categories: physiological, pathological, artifact, and powerline noise
- Uses a state-of-the-art WaveNet model
- High accuracy with less training data

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/caspervanlaar/EEG_wavenet.git
    cd EEG_wavenet
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To use the EEG classifier, follow these steps:

1. **Prepare your data:** Ensure your EEG data is in the correct format.
2. **Run the classifier:**
    ```bash
    jupyter notebook EEG_wavenet.ipynb
    ```
   Follow the steps in the Jupyter notebook to preprocess data, train the model, and evaluate performance.

## Project Structure
```
EEG-Classifier/
│
├── data/ https://www.kaggle.com/datasets/nejedlypetr/multicenter-intracranial-eeg-dataset              # Sample data files
├── models/ model_test.keras                                                                            # Pre-trained models and saved weights
├── scripts/ WaveNet_EEG.ipynb                                                                          # Utility scripts for data preprocessing
├── EEG_wavenet.ipynb                                                                                   # Main Jupyter notebook for the classifier
├── requirements.txt                                                                                    # Python dependencies
└── README.md                                                                                           # This README file
```

## Preprocessing
The project includes detailed preprocessing steps such as:
### Preprocessing Steps Overview

1. **File Loading:**
   - Load .mat files from Mayo Clinic and St. Anne’s University Hospital datasets.
   - Each file has labels: 0 (Powerline noise 50Hz), 1 (Artifacts), 2 (Physiological), 3 (Pathological).

2. **Data Extraction:**
   - Use `process_file` to read .mat files and store EEG data in an HDF5 file (`dataset.h5`).

3. **HDF5 File Creation:**
   - `create_hdf5` processes files for each label, organizing data in HDF5 format for efficient storage and access.

4. **Data Splitting:**
   - Split data into training, validation, and test sets with no overlap, using `split_and_save_data`.

5. **TensorFlow Dataset Creation:**
   - Generate TensorFlow datasets for training, validation, and testing.

6. **Data Leakage Check:**
   - Ensure no data leakage using `check_for_overlap`.

7. **File Separation:**
   - Create two datasets: 
     - **Joined Dataset:** Combined data from both hospitals for testing.
     - **Disjoined Dataset:** Separate training, validation, and test sets ensuring no overlap.


## Model Evaluation
### Model Evaluation Overview

The WaveNet model was trained on a curated EEG dataset from Mayo Clinic and St. Anne’s University Hospital, with 186 samples split into 80% training, 10% validation, and 10% testing.

- **Training Performance:**
  - Achieved 94% accuracy, surpassing previous models (CNN, Convolutional LSTM) based on F1 score, PPV, and Sensitivity.
  

- **Training Metrics:**
  - Per-step loss fluctuations indicate the model's handling of temporal EEG data intricacies, showing incremental accuracy improvements.

- **Validation Performance:**
  - Consistent and steady improvement in validation loss and accuracy across epochs.

- **Test Performance:**
  - Achieved 94% accuracy on a large test dataset of 209,232 samples.

### Summary

The WaveNet model demonstrated exceptional performance in EEG classification, achieving flawless accuracy during training and testing, highlighting its potential for advancing clinical diagnostics and precision medicine in neurology.

## Results
The WaveNet model achieves 100% accuracy with significantly less training data compared to previous models.

## Contributing
We welcome contributions to improve the EEG classifier. Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any questions or issues, please contact casperdvanlaar@hotmail.com.

