
************Age Prediction Model (UTKFace Dataset)**************

Overview
This project trains a convolutional neural network (CNN) to predict age from grayscale facial images using the UTKFace dataset.

Model Architecture
Input Shape: (128, 128, 1)
Convolutional Layers:
3 Conv2D layers with ReLU activation
Batch Normalization
MaxPooling layers for downsampling
Fully Connected Layers:
Flatten layer
Dense layer with 256 neurons (ReLU)

Dropout (40%)

Output:
A single neuron (Linear activation) predicting age
Preprocessing
Convert images to grayscale
Resize to 128x128 pixels
Normalize pixel values to [0,1]
Scale age labels to [0,1] (divide by 100)

Training
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam
Metric: Mean Absolute Error (MAE)
Early Stopping: Stops training if validation loss doesn’t improve

Testing & Prediction
Load trained model (model_age.keras)
Preprocess input image (grayscale, resize, normalize)
Predict age and scale back to original range


link dataset: https://www.kaggle.com/datasets/jangedoo/utkface-new