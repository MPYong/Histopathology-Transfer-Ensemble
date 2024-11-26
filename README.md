This project explored the models based on transfer learning and ensemble learning for histopathology image analysis


Deep learning library used: Tensorflow/Keras, sklearn, matplotlib




**1. Data preprocessing**


* Performed augmentation on the datasets such as rotation and flipping


* Loaded the augmented datasets


**2. Models training and evaluation**


* Deployed pretrained networks (base models) with ImageNet weights such as DenseNet, ResNet, EfficientNet, InceptionV3, and Xception


* Trained and evaluated the base models on the datasets


* Developed ensemble models combining the base models with best validation set performance using following ensemble strategies:
  * Majority voting
  * Weighted averaging
  * Unweighted averaging


**3. Model performance analysis**


* Generated folders to identify and arrange samples predicted correctly and wrongly by individual models and all models


* Generated performance visualization graphs such as training/testing accuracies against epochs, ROC and Precision-Recall curve


* Utilized GradCAM heatmap to identify the ROI in the images



![GradCAM heatmap example](https://github.com/MPYong/gashissdb_transfer_ensemble_learning/blob/main/figures/gradcam.jpg?raw=true)
