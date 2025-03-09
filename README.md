# Empowering Brain Tumor Diagnosis Through Explainable Deep Learning

## Project Overview

This project implements a Deep Convolutional Neural Network (CNN) to classify brain MRI scans into different tumor categories. The model is designed to assist medical professionals in the early detection and classification of brain tumors, enhancing diagnostic precision and reliability.

Brain tumors are among the most lethal diseases, and early detection is crucial for improving patient outcomes. Currently, magnetic resonance imaging (MRI) is the most effective method for early brain tumor detection due to its superior imaging quality for soft tissues. However, manual analysis of brain MRI scans is prone to errors, largely influenced by the radiologists' experience and fatigue.

## Problem Statement

Brain tumor is the accumulation or mass growth of abnormal cells in the brain. There are basically two types of tumors, malignant and benign. Malignant tumors can be life-threatening based on the location and rate of growth. Hence timely intervention and accurate detection is of paramount importance when it comes to brain tumors. This project focuses on classifying 3 types of brain tumors based on their location from normal cases (no tumor) using Convolutional Neural Networks. The three types of tumors are glioma, meningioma, pituitary.

## Proposed Framework

The methodology includes several key steps:

1. **Dataset Acquisition**: The Brain Tumor MRI Dataset, which includes images of meningioma, glioma, and pituitary tumorswith a notumor class, was sourced from publicly available repositories.
2. **Image Preprocessing**: To improve data quality, comprehensive image preprocessing techniques were applied:

   - Cropping to remove unwanted background noise
   - Noise removal using bilateral filtering
   - Applying colormap to enhance contrast
   - Resizing for standardized input
3. **Data Partitioning and Augmentation**: The dataset was randomly divided into training, validation, and testing sets. Image augmentation techniques were exclusively applied to the training set to increase its diversity and robustness.
   Training data : 4569
   Validation data : 1143
   Test data : 1311
4. **Model Training**: I have fine-tune Resnet-50 model on the train data set  to effectively classify brain MRI scans into different tumor categories.
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

| **Category** | **Training** | **Validation** | **Testing** |
| ------------------ | ------------------ | -------------------- | ----------------- |
| Glioma tumor       | 1,060              | 261                  | 300               |
| Meningioma tumor   | 1,072              | 267                  | 306               |
| Pituitary tumor    | 1,158              | 299                  | 300               |
| No tumor           | 1,279              | 316                  | 405               |
| **Total**    | **4,569**    | **1,143**      | **1,311**   |

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
git https://github.com/tapan0p/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification

# Install dependencies
pip install -r requirements.txt
```

### Data Preprocessing

The `Image_preprocessing.ipynb` notebook contains code for:

- Loading and resizing MRI images
- Cropping to remove background noise
- Noise removal using bilateral filtering
- Applying colormap for contrast enhancement
- Data augmentation techniques

### Model Training

The `brain-tumor-classification-tl.ipynb` notebook demonstrates:

- Dataset loading and preparation
- Model initialization and architecture
- Training process with hyperparameter configuration
- Evaluation metrics and performance analysis

### Model Architecture of Resnet-50

1. **Deep CNN with 50 Layers**

   - ResNet-50 (Residual Network 50) consists of **50 layers** with convolutional, pooling, and fully connected layers. It is designed for deep learning tasks like image classification.
2. **Residual Connections (Skip Connections)**

   - The key feature of ResNet-50 is **residual learning**, which introduces **skip connections** (shortcut paths) to help mitigate the **vanishing gradient problem**, enabling deep networks to train effectively.
3. **Bottleneck Blocks for Efficiency**

   - Unlike shallower ResNet versions, ResNet-50 uses **bottleneck blocks**, each containing **three convolutional layers (1x1, 3x3, 1x1)** instead of two, reducing computational cost while maintaining accuracy.
4. **Layer Structure**

   - The architecture follows a **conv1 → conv2_x → conv3_x → conv4_x → conv5_x → fully connected** pattern, where:
     - **Conv1**: 7×7 convolution + max pooling
     - **Conv2_x to Conv5_x**: Series of bottleneck residual blocks
     - **Fully connected layer** outputs class probabilities
5. **Pretrained on ImageNet**

   - ResNet-50 is commonly **pretrained on ImageNet**, allowing transfer learning for various computer vision tasks. It achieves high accuracy on benchmark datasets while being computationally efficient.

## Results

The model achieves promising results in classifying brain tumors from MRI scans:

- Accuracy: 97.56%
- Precision: 97.68%
- Recall: 97.56
- F1 Score: 97.58
- Avg Inference Time per Image: 0.000209 sec
- Confusion matrix:
![Alt Text](https://github.com/tapan0p/Brain-Tumor-Classification/blob/main/Plots/confusion_matrix.png)
- Loss and accuracy plot during training:
![Alt Text](Plots\accuracy_plot.png)

## Interpretability

To enhance the transparency of the model's decision-making process, Grad-CAM is  employed to highlight the regions of interest in the MRI scans that contribute most significantly to the classification decision.

## License

## Acknowledgements

- Dataset source: [Link Text](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Research papers referenced: [Link Text](https://www.mdpi.com/2504-4990/6/4/111)
