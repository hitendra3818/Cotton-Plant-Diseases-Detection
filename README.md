# Cotton-Plant-Diseases-Detection
Introduction
Cotton is a crucial crop in the agricultural sector. Diseases affecting cotton plants can cause significant yield losses. Early detection of these diseases is vital for effective management and control. This project uses deep learning techniques to classify diseases in cotton plants based on leaf images.

Dataset
The dataset consists of images of cotton leaves classified into different disease categories. The images are divided into training and validation sets, which are used to train and evaluate the model.

Data Preprocessing
Data augmentation is applied to increase the diversity of the training set and improve the model's generalization capabilities. The following augmentations are applied:

Rescaling: Pixel values are scaled to the range [0, 1].
Width and Height Shift: Random shifts in the width and height of images.
Shear and Zoom Transformations: To simulate variations in image capture conditions.
Horizontal Flip: Randomly flipping images to account for leaf orientation.
Model Architecture
The CNN model is constructed using TensorFlow and Keras, with the following architecture:

Convolutional Layers: Four layers with increasing filter sizes (32, 64, 128, 256) and kernel size of 3x3.
Pooling Layers: MaxPooling layers with a pool size of 2x2 after each convolutional layer.
Dropout Layers: Dropout layers to prevent overfitting, with dropout rates of 0.5, 0.1, and 0.25.
Fully Connected Layers: Dense layers with 128 and 256 units, using ReLU activation.
Output Layer: A final Dense layer with 4 units (corresponding to 4 classes) using softmax activation.
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as the evaluation metric.

Training
The model is trained on the augmented training set using a batch size of 32 and is validated on a separate validation set. The training process involves monitoring accuracy and loss to ensure proper learning.

Results
The results of the training process, including accuracy and loss curves, are used to evaluate the model's performance. A confusion matrix is also generated to assess the model's classification accuracy across different disease categories.

Conclusion
The CNN model developed in this project successfully classifies cotton plant diseases with a high degree of accuracy. This model can be further improved by experimenting with more complex architectures or larger datasets.
