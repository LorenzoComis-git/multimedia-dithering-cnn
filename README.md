# Analysis of the Impact of Dithering on Image Classifiability with CNNs

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project analyzes how **quantization** and different **dithering** techniques affect the ability of a **Convolutional Neural Network (CNN)** to correctly classify realistic images.

## 📋 Abstract

The reduction in color depth (e.g., from 8 bits to 2 bits per channel) significantly degrades the visual quality of images, making classification tasks more challenging for neural networks. While dithering techniques are commonly used to improve human visual perception of quantized images, their impact on automatic image classification remains less explored.

This study investigates whether dithering techniques can maintain or improve classification accuracy when images undergo severe color quantization. We train a CNN on original 8-bit images and evaluate its performance on quantized 2-bit images with and without various dithering techniques.

## 🎯 Objectives

- Train a CNN on original images to establish a baseline performance
- Apply uniform quantization (2 bits per channel) to reduce color depth
- Implement and compare three dithering techniques: Random, Ordered (Bayer), and Error Diffusion (Floyd-Steinberg)
- Evaluate and compare model performance across all degradation methods
- Analyze which dithering technique best preserves classification accuracy

## 📊 Dataset

The project uses a subset of the **Caltech 256** dataset consisting of:
- **5 object classes**: butterfly, dolphin, laptop, screwdriver, spaghetti
- **100 images per class** (500 total images)
- **90/10 train-test split** (450 training, 50 test images)
- Images resized to **128×128 pixels**
- **RGB color space** (3 channels)

## 🔧 Degradation Techniques

### 1. Uniform Quantization (Baseline)
Direct reduction from 8 bits to 2 bits per channel (256 → 4 levels per channel), resulting in significant color banding and loss of detail.

### 2. Random Dithering
Adds uniform random noise before quantization to break up color banding patterns, trading banding artifacts for grain texture.

### 3. Ordered Dithering (Bayer Matrix)
Uses a 4×4 Bayer matrix to create a regular pattern of noise, producing a characteristic halftone appearance.

### 4. Error Diffusion (Floyd-Steinberg)
Propagates quantization errors to neighboring pixels using the Floyd-Steinberg kernel, preserving edge details and producing the most visually pleasing results.

## 🏗️ Model Architecture

The CNN architecture was designed to balance performance and training time:

```
Input (128×128×3)
    ↓
Conv2D(32 filters, 3×3) + ReLU + MaxPooling(2×2)
    ↓
Conv2D(64 filters, 3×3) + ReLU + MaxPooling(2×2)
    ↓
Conv2D(128 filters, 3×3) + ReLU + MaxPooling(2×2)
    ↓
Flatten
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(5) + Softmax
```

**Training Configuration:**
- Optimizer: Adam (learning rate = 0.001)
- Loss: Sparse Categorical Crossentropy
- Early Stopping: patience = 5 epochs
- Batch size: 8
- Max epochs: 50

## 📦 Requirements

- Python 3.8 or higher
- See `requirements.txt` for the full list of dependencies

### Installation

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install numpy matplotlib scikit-learn tensorflow seaborn jupyter
```

## 🚀 Usage

1. Place your dataset in the `Dataset/` folder with the following structure:
```
Dataset/
├── butterfly/
├── dolphin/
├── laptop/
├── screwdriver/
└── spaghetti/
```

2. Open and run `Multimedia.ipynb` in Jupyter Notebook or JupyterLab

## 📈 Results

### Classification Accuracy

The model was trained exclusively on original (8-bit) images and then tested on both original and degraded versions:

| Image Type | Accuracy | Relative Performance |
|-----------|----------|---------------------|
| **Original** | **70.0%** | Baseline (100%) |
| **Error Diffusion** | **68.0%** | 97.1% of baseline |
| **Random Dithering** | **66.0%** | 94.3% of baseline |
| **Quantized** | **62.0%** | 88.6% of baseline |
| **Ordered Dithering** | **62.0%** | 88.6% of baseline |

### Visual Comparison

![Accuracy Comparison](images/Accuracy%20barplot.png)
*Figure 1: Classification accuracy across different image degradation techniques*

### Key Findings

1. **Quantization Impact**: Simple 2-bit quantization causes an **11.4% drop** in accuracy (70% → 62%), demonstrating the significant impact of color depth reduction on CNN classification.

2. **Error Diffusion Performs Best**: Among dithering techniques, **Error Diffusion** best preserves classification accuracy (68%), losing only **2.9%** compared to the 11.4% loss from quantization alone.

3. **Dithering Helps**: All dithering techniques outperform or match simple quantization:
   - Error Diffusion: **+6.0%** vs quantization
   - Random Dithering: **+4.0%** vs quantization
   - Ordered Dithering: **0%** (same as quantization)

4. **Ordered Dithering Limitation**: Despite producing visually acceptable results for humans, Ordered Dithering (Bayer matrix) provides **no improvement** over simple quantization for CNN classification.

### Confusion Matrices

![Confusion Matrices](images/Confusion%20Matrix.png)
*Figure 2: Normalized confusion matrices showing prediction patterns across all test conditions*

### Dataset Sample

![Dataset Classes](images/Classes%20img.png)
*Figure 3: Sample images from each of the 5 object classes in the dataset*

### Visual Examples of Degradation

![Dithering Comparison](images/Modified%20img.png)
*Figure 4: Visual comparison of original and degraded images showing the effect of each quantization and dithering technique*

## 🔍 Discussion

### Why Error Diffusion Works Best

Error Diffusion (Floyd-Steinberg) outperforms other dithering methods because it:
- **Preserves edges**: The error propagation mechanism maintains sharp boundaries between objects
- **Maintains spatial coherence**: Errors are distributed to neighboring pixels, preserving local patterns that CNNs rely on
- **Adapts to content**: Unlike fixed-pattern dithering (Bayer), error diffusion adapts to image content

### Why Ordered Dithering Fails

The regular Bayer matrix pattern:
- Introduces **high-frequency artifacts** that may confuse convolutional filters
- Creates a **consistent halftone pattern** unrelated to image content
- May trigger CNN filters trained on natural image statistics in unexpected ways

### Implications

These results suggest that:
1. **Dithering can partially compensate** for quantization losses in CNN classification tasks
2. **Content-adaptive techniques** (Error Diffusion) are superior to fixed-pattern approaches (Bayer)
3. **Visual quality ≠ Classification accuracy**: Techniques that improve human perception may not equally benefit CNNs
4. CNNs trained on high-quality images can maintain reasonable performance on heavily degraded inputs if appropriate preprocessing is applied

## 💡 Conclusions

This study demonstrates that severe color quantization significantly degrades CNN classification performance, but appropriate dithering techniques can mitigate this loss. Error Diffusion proves most effective, recovering approximately **50% of the accuracy lost** to quantization. This has practical implications for deploying CNNs in bandwidth-constrained or low-storage scenarios where image quality must be sacrificed.

## 👨‍💻 Author

Lorenzo Comis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
