# TAM SERS Spectral Classification Based on CNN

This repository provides a complete deep learning pipeline for SERS classification, including spectral preprocessing, CNN model construction, training, evaluation, and visualization. 
The framework is designed for biological Raman spectroscopy applications and can be extended to cell phenotype recognition and drug sensitivity analysis.

Project Structure

├── Model.py
│   CNN model definition for Raman spectrum classification.
│
├── train.py
│   Training pipeline including data loading, preprocessing, model optimization and checkpoint saving.
│
├── Test.py
│   Model evaluation with quantitative metrics and visualization (confusion matrix, ROC, t-SNE, etc.).
│
├── pretreatment.py
│   Raman spectrum preprocessing, including smoothing, normalization and standardization.
│
└── picture.py
    Figure configuration utilities for generating publication-quality plots.
