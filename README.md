# Skin Lesion Classification with Deep Learning  
📁 Project: HAM10000 – Melanoma Detection

## 🔍 Project Overview
This project applies Convolutional Neural Networks (CNNs) and transfer learning (ResNet50) to classify dermatoscopic images from the **HAM10000 dataset**. The aim is to support early detection of skin cancer by building models that can distinguish between 7 skin lesion types.

## 🧠 Models Compared

We implemented and evaluated three models:

1. **Simple CNN**  
   A custom-built convolutional neural network with 3 convolutional blocks and dropout.

2. **ResNet50 (Frozen)**  
   A pre-trained ResNet50 model with frozen layers, used as a feature extractor.

3. **ResNet50 Fine-Tuned + Class Weights**  
   The top-performing model. It unfreezes deeper layers of ResNet50 and includes class weighting to address class imbalance.

## 📊 Results Summary

| Model                          | Accuracy | F1 Macro | F1 Weighted |
|-------------------------------|----------|----------|-------------|
| Simple CNN                    | 0.48     | 0.30     | 0.54        |
| ResNet50 (Frozen)             | 0.67     | 0.12     | 0.54        |
| ResNet50 Fine-Tuned + Weights | 0.66     | 0.29     | 0.66        |

> ✅ The third model showed the best generalization, especially across minority classes.

## 🧪 Dataset

- **Name**: HAM10000 – Human Against Machine with 10000 training images
- **Classes**: akiec, bcc, bkl, df, mel, nv, vasc
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## 🧼 Preprocessing

- Resized images to 128x128
- Normalized pixel values to [0,1]
- Encoded labels to integers
- Handled class imbalance using `compute_class_weight` from scikit-learn

## 📈 Key Insights

- Fine-tuning ResNet50 with class weights achieved the best overall performance.
- The model showed a steady learning curve over 25 epochs, indicating potential for further improvement with more computational resources.
- Minority classes such as “df” and “vasc” remain difficult to classify, suggesting a need for synthetic augmentation or focal loss.

## 🚧 Limitations & Next Steps

- Class imbalance strongly affected underrepresented classes.
- Future work could explore:
  - Focal loss or weighted focal loss
  - Synthetic sample generation using GANs
  - Using EfficientNet or Vision Transformers
  - Hyperparameter tuning with Bayesian Optimization

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn
- NumPy / Pandas


