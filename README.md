#  CognitiveEye — End-to-End Deep Metric Learning Pipeline

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-blue.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Computer Vision pipeline implementing **Deep Metric Learning** with **Hard Triplet Mining** and a fine-tuned **ResNet-50 backbone**. This engine maps high-dimensional images into a unified, discriminative 128-D vector space where visual similarity correlates directly with Euclidean distance.

---

##  Project Overview

Unlike traditional image classifiers that predict fixed categorical labels, **CognitiveEye** learns a continuous embedding space. By pulling images of the same class together and pushing different classes apart, the network optimizes an explicit distance metric. 

### Key Features
* **Feature Extraction Backbone:** ResNet-50 pre-trained weights, with early structural layers frozen to preserve generic feature detectors and deep blocks unfrozen for fine-tuning.
* **Projection Head:** Dense linear mapping to a 128-dimensional hypersphere enforced by custom L2 Normalization.
* **Data Ingestion Pipeline:** Automated downloading, cross-validation extraction, and filtering of a 10-class subset from the Stanford Tiny-ImageNet dataset.
* **Hard Triplet Mining Engine:** A custom balanced batch sampler paired with an online batch-hard mining strategy, isolating the most problematic positive and negative pairs per step.
* **Loss Function:** Optimized via Triplet Margin Loss with a dynamically scheduled `AdamW` optimizer.

---
##  Repository Structure
```text
CognitiveEye-Metric-Learning/
│
├── cognitive_eye_pipeline.ipynb   # Main Jupyter Notebook containing full pipeline execution
├── requirements.txt               # Required Python environments and framework versions
├── .gitignore                     # Prevents multi-gigabyte datasets and model weights from being tracked
└── README.md                      # Detailed technical documentation and project guide
```

#⚙️ Setup and Installation
1. Clone the Repository
Bash
git clone https://github.com/javaria-gul/CognitiveEye-Metric-Learning.git
cd CognitiveEye-Metric-Learning
2. Install Dependencies
Ensure you have Python 3.8+ installed along with a CUDA-capable GPU runtime environment.

Bash
pip install -r requirements.txt
 
  Pipeline Workflow
The complete pipeline is structured linearly inside the cognitive_eye_pipeline.ipynb notebook across automated code blocks:

Environment Initialization: Configures global seeds, multi-processing workers, and verifies cuda device state.

Dataset Restructuring: Scripts downsample the Tiny-ImageNet dataset to 10 active classes, organizing balanced training splits (500 images/class).

Model Construction: Instantiates the custom dual-component CognitiveEyeNet module.

Metric Training: Executes the training loops across 30 epochs utilizing online triplet selection.

Evaluation & Artifact Generation: Extracts model checkpoints (cognitive_eye_backbone.pth) and outputs structural 2D mappings via t-SNE algorithms.

  Expected Performance Metrics & Outputs
Upon successful execution of the full notebook pipeline, the following local artifacts are generated:

cognitive_eye_backbone.pth: The production-ready weights matrix containing state parameters.

training_curves.png: Epoch-by-epoch loss reduction profiles and distance tracking vectors.

embedding_tsne.png: A 2-D cluster projection illustrating high-margin separating boundaries between validation classes.

  License
Distributed under the MIT License. See LICENSE for more information.





