# **Histopathology Image Classification Using Transfer Learning and Ensemble Learning**

This project explores transfer learning and ensemble learning approaches for histopathology image classification using publicly available datasets. The goal was to use pretrained networks and ensemble strategies to improve classification performance.


## **Datasets**

The following public datasets were used:

* [GasHisSDB](https://gitee.com/neuhwm/GasHisSDB)

* [HICL](http://medisp.bme.teiath.gr/hicl/)


## **Libraries and Tools**

* Tensorflow/Keras
* sklearn
* matplotlib


## **Workflow**

### **1. Data preprocessing**


* Applied data augmentation techniques such as rotation and flipping.


## **2. Models training and evaluation**


* Utilized pretrained networks (base models) with ImageNet weights, including DenseNet, ResNet, EfficientNet, InceptionV3, and Xception


* Trained and evaluated each base model on the datasets.

* Developed ensemble models using the top-performing base models based on validation set performance using following ensemble strategies:
  * Majority voting
  * Weighted averaging
  * Unweighted averaging


## **3. Model performance analysis**


* Organized samples into folders based on correct and incorrect predictions by individual models and ensemble models.


* Created performance visualizations, including:
  * Training/testing accuracies against epochs
  * ROC
  * Precision-Recall curve


* Utilized GradCAM heatmap to identify the ROI in the images


## **Examples**

### **GradCAM heatmap example**

<img src="https://github.com/MPYong/gashissdb_transfer_ensemble_learning/blob/main/figures/gradcam.jpg" width="800" />


## **ROC graph example**


<img src="https://github.com/MPYong/gashissdb_transfer_ensemble_learning/blob/main/figures/ROC.jpg" width="600" />


Through this project, I:

* Gained hands-on experience with transfer learning and pretrained models for histopathology image classification.
* Improved understanding of ensemble learning techniques to enhance model performance.
* Learned to interpret model predictions using Grad-CAM heatmaps.
* Enhanced my skills in evaluating model performance with scikit-learn and Matplotlib.
* Strengthened my knowledge of deep learning by using TensorFlow/Keras effectively.
