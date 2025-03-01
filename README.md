# Empowering Brain Tumor Diagnosis Through Explainable Deep Learning

## Project Overview

This project implements a Convolutional Neural Network (CNN) to classify brain MRI scans into different tumor categories. The model is designed to assist medical professionals in the early detection and classification of brain tumors, enhancing diagnostic precision and reliability.

Brain tumors are among the most lethal diseases, and early detection is crucial for improving patient outcomes. Currently, magnetic resonance imaging (MRI) is the most effective method for early brain tumor detection due to its superior imaging quality for soft tissues. However, manual analysis of brain MRI scans is prone to errors, largely influenced by the radiologists' experience and fatigue.

## Problem Statement

Brain tumor is the accumulation or mass growth of abnormal cells in the brain. There are basically two types of tumors, malignant and benign. Malignant tumors can be life-threatening based on the location and rate of growth. Hence timely intervention and accurate detection is of paramount importance when it comes to brain tumors. This project focuses on classifying 3 types of brain tumors based on their location from normal cases (no tumor) using Convolutional Neural Networks.

## Proposed Framework

The methodology includes several key steps:

1. **Dataset Acquisition**: The Brain Tumor MRI Dataset, which includes images of meningioma, glioma, and pituitary tumors, was sourced from publicly available repositories.

2. **Image Preprocessing**: To improve data quality, comprehensive image preprocessing techniques were applied:
   - Cropping to remove unwanted background noise
   - Noise removal using bilateral filtering
   - Applying colormap to enhance contrast
   - Resizing for standardized input

3. **Data Partitioning and Augmentation**: The dataset was randomly divided into training, validation, and testing sets. Image augmentation techniques were exclusively applied to the training set to increase its diversity and robustness.

4. **Model Training**: The CNN architecture was designed to effectively classify brain MRI scans into different tumor categories.

5. **Performance Evaluation**: The model's effectiveness was evaluated using various metrics, including accuracy, specificity, sensitivity, F1-score, and confusion matrix.

6. **Interpretability**: To improve transparency, visualization techniques were employed to understand the decision pathways of the model.

## Dataset

The dataset consists of brain MRI scans categorized into four classes:

- Glioma
- Meningioma
- Pituitary
- No tumor

### Dataset Structure

```
Brain_Tumor_MRI_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### Dataset Distribution

| **Category**       | **Training** | **Validation** | **Testing** |
|--------------------|--------------|----------------|-------------|
| Glioma tumor       | 1,060        | 261            | 300         |
| Meningioma tumor   | 1,072        | 267            | 306         |
| Pituitary tumor    | 1,158        | 299            | 300         |
| No tumor           | 1,279        | 316            | 405         |
| **Total**          | **4,569**    | **1,143**      | **1,311**   |

## Project Structure

```
Brain-Tumor-Classification/
├── Brain_Tumor_MRI_dataset/  # Dataset directory
├── CNN_model.py              # CNN model implementation
├── main.ipynb                # Main notebook for training and evaluation
├── Image_preprocessing.ipynb # Image preprocessing notebook
├── Results/                  # Directory containing results and visualizations
└── README.md                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- OpenCV
- NumPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

The `Image_preprocessing.ipynb` notebook contains code for:

- Loading and resizing MRI images
- Cropping to remove background noise
- Noise removal using bilateral filtering
- Applying colormap for contrast enhancement
- Data augmentation techniques

### Model Training

The `main.ipynb` notebook demonstrates:

- Dataset loading and preparation
- Model initialization and architecture
- Training process with hyperparameter configuration
- Evaluation metrics and performance analysis

### Model Architecture

The CNN architecture consists of:

- Convolutional layers for feature extraction
- Pooling layers for dimensionality reduction
- Fully connected layers for classification
- ReLU activation functions
- Softmax output layer for multi-class classification

## Results

The model achieves promising results in classifying brain tumors from MRI scans:

- Accuracy: (To be updated)
- Precision: (To be updated)
- Recall: (To be updated)
- F1 Score: (To be updated)

Confusion matrix and visualization of model predictions will be included to provide a comprehensive understanding of the model's performance.

## Interpretability

To enhance the transparency of the model's decision-making process, visualization techniques are employed to highlight the regions of interest in the MRI scans that contribute most significantly to the classification decision.

## Future Improvements

Despite the current advancements, several areas for improvement have been identified:

- Exploring federated learning techniques to enhance model accuracy while preserving patient privacy
- Integrating blockchain technology for improved transparency and traceability
- Expanding the dataset with more diverse cases to improve model robustness
- Implementing ensemble methods to combine predictions from multiple models
- Developing a more user-friendly interface for clinical deployment

## License

(To be updated with license information)

## Acknowledgements

- Dataset source: (To be updated)
- Research papers referenced: (To be updated)
- Contributors: (To be updated)
