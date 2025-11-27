# Breast Density Classification Model

**Course:** Machine Learning for the Life Sciences (BSCI238I)  
**Author:** Catherine Smith  
**Semester:** Fall 2025  

---

## Overview

This project develops a **deep learning model** that predicts **breast density** from mammogram images using a **ResNet-50 convolutional neural network**.

The model demonstrates:

* Image preprocessing and dataset management  
* Stratified train/validation/test split  
* Transfer learning with ResNet-50  
* Model training and hyperparameter tuning  
* Evaluation using accuracy, confusion matrices, and Grad-CAM saliency maps  
* Interpretation for early breast cancer risk assessment

---

## Repository Structure

```text
breast-density-classification/
│
├── README.md
├── breast_density_classification_mlproject.ipynb
│   
├── results/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── loss_accuracy_learning-rate-graphs
│   ├── saliency_map-A
│   ├── saliency_map-B
│   ├── saliency_map-C
│   └── saliency_map-D
```
To run the project, the following Python libraries are required:
* ```torch```
* ```torchvision```
* ```pandas```
* ```numpy```
* ```scikit-learn```
* ```matplotlib```
* ```seaborn```
* ```opencv-python```
* ```Pillow```
* ```tqdm```

## Methodology
Data Preparation
- Loaded mammogram images and metadata from the RSNA dataset
  - https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview
- Stratified train/validation/test split (70/15/15) to maintain class balance
- Applied image transformations: resizing, normalization

Model
- Architecture: ResNet-50 (pretrained on ImageNet)
- Fine-tuning: Last layers adapted for 4-class classification
- Loss function: Cross-entropy
- Optimizer: Adam with weight decay
- Learning rate scheduler: CosineAnnealingLR
- Hyperparameters: batch size = 32, learning rate = 1e-5, weight decay = 3e-3, epochs = 15

Evaluation Metrics
- Overall accuracy = 79.17%
- Note: The normalized confusion matrix displays percentages, rather than raw counts. For example, 0.93 in the top-left corner means that the model correctly classified 93% of class A images as density A.
- Confusion matrix analysis:
  - Best performance on class A
  - Worst performance on class B
  - Errors are not equally distributed; minority classes or classes with overlapping features are more likely to be misclassified
  - For class C, prioritizing higher specificity is recommended to reduce false positives
- Grad-CAM saliency maps for interpretability

## Key Results

Validation Performance
- Strong classification accuracy across all four breast density classes
- Stratified splitting maintained balanced class distribution

Test Performance
- Overall accuracy = 79.17%
- Confusion matrix highlights class-wise strengths and weaknesses
- Model focuses on relevant regions in breast tissue, confirmed via Grad-CAM

Interpretation
- Class C prioritizes higher specificity to minimize false positives
- Helps identify women with class C and D densities for potential additional screening

## License

This project is for educational purposes as part of the UMD Machine Learning for the Life Sciences course (BSCI238I).
