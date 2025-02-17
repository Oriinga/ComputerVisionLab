# Computer Vision Labs: Puzzle Assembly Automation

## Project Goal
The goal of this project is to develop a system that helps Tino efficiently solve puzzles by automating the puzzle assembly process. The system addresses challenges such as segmentation, shape matching, and texture matching using computational, statistical, and mathematical techniques.

## Repository
- **GitHub**: [ComputerVisionLab](https://github.com/Oriinga/ComputerVisionLab)
- **Languages**: Python
- **Libraries**: PyTorch, OpenCV, Weights & Biases (wandb)

---

## Labs Overview

### Lab 1: Segmentation
**Objective**: Separate puzzle piece pixels from the background.  
**Methods**: Implement segmentation algorithms and evaluate their accuracy on a held-out test set.  

**Key Steps**:
1. Read and display images using libraries like Matplotlib and OpenCV.
2. Calculate descriptive statistics of the images and masks.
3. Implement a background classifier using features like RGB, HSV, and filter outputs.
4. Evaluate performance using metrics like accuracy, precision, recall, F1 score, IoU, and ROC curves.
5. Perform feature selection to optimize the model.

---

### Lab 2: Per-pixel Feature Extraction
**Objective**: Enhance the background classifiers using more sophisticated feature extraction techniques.  
**Methods**: Implement and apply various filters, including:
- Gaussian
- Laplacian of Gaussian (LoG)
- Difference of Gaussian (DoG)
- RFS/MR8 filter banks
- Local Binary Patterns (LBPs)
- Haar filters
- Textons

---

### Lab 3: Segmentation with Deep Learning
**Objective**: Investigate deep learning-based segmentation models.  
**Methods**:
- Implement the UNet model from scratch using PyTorch.
- Experiment with advanced architectures from libraries.

---

### Lab 4: Contours, Shape & Texture Models
**Objective**: Extract shape information from puzzle piece masks to facilitate matching.  
**Methods**:
- Extract contours.
- Build shape models for each side.
- Match sides using k-nearest neighbours.

---

### Lab 5: Affine Transformations
**Objective**: Extract puzzle pieces from individual images and insert them into the correct position on the puzzle canvas.  
**Methods**: Use affine transformations to warp and insert puzzle pieces into the canvas.

---
