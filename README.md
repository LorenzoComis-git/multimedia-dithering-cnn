# Analysis of the Impact of Dithering on Image Classifiability with CNNs

This project analyzes how **quantization** and different **dithering** techniques affect the ability of a **Convolutional Neural Network (CNN)** to correctly classify realistic images.

## 📋 Overview

The reduction in color depth (e.g., from 8 bits to 2 bits per channel) degrades the visual quality of images, making classification tasks more difficult. This project investigates whether dithering techniques can improve not only human visual perception but also automatic classification by neural networks.

## 🎯 Objectives

- Train a CNN on original images to establish a baseline
- Apply quantization and various dithering techniques to test images
- Evaluate model performance on degraded images
- Compare classification accuracy across different degradation methods

## 📊 Dataset

The project uses a subset of the **Caltech 256** dataset consisting of:
- **5 object classes**: butterfly, dolphin, laptop, screwdriver, spaghetti
- **100 images per class**
- Images resized to **128×128 pixels**

## 🔧 Techniques Evaluated

1. **Uniform Quantization** (2 bits per channel)
2. **Random Dithering**
3. **Ordered Dithering** (Bayer matrix)
4. **Error Diffusion** (Floyd-Steinberg algorithm)

## 🏗️ Model Architecture

The CNN consists of:
- 3 convolutional layers with ReLU activation and max pooling
- 1 fully-connected layer with ReLU activation
- Dropout layer (0.5) for regularization
- Output layer with softmax activation

## 📦 Requirements

```
numpy
matplotlib
scikit-learn
tensorflow
seaborn
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

The notebook includes:
- Visual comparison of original vs. degraded images
- Accuracy metrics for each degradation technique
- Confusion matrices for error analysis
- Performance comparison charts

## 👨‍💻 Author

Lorenzo Comis

## 📄 License

This project is available for educational purposes.
