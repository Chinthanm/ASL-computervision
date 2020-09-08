# ASL Image CLassification

End-to-end Machine Learning problem from ingestion and analysis of image dataset to development of a deep learning neural network to accurately classify images of American Sign Language hand gestures.

## Table of Contents

* [Project Outline](#project-outline)
* [Data Wrangling](#data-cleaning-and-wrangling)
* [Data Analysis](#data-analysis)
* [Machine Learning Predictions](#machine-learning-analysis)
* [Results](#conclusion-and-results)

## Project Outline

#### Problem Definition
Study the images for each letter of the American Sign Language and use the data to perform classifications and real-world applications to understand what someone is saying. 
#### Client Base and Motivation
The society of people that use American Sign Language can be better understood with the help of deep learning for computer vision to interpret their gestures in real-time.  
   
#### Data Overview
* Acquired from Kaggle:  https://www.kaggle.com/grassknoted/asl-alphabet 
* Two separate folders files: train and test
  * Train: The data contains 87,000 images of 200x200 pixels for the 29 classes. The classes include the 26 letters of the  alphabet, space, delete, and nothing. Each of the classes has 3,000 images for training.
  * Test: The folder contains one image for each class
   
## Data Cleaning and Wrangling
* We extract this data and store the images, characters they represent, and their label (0-28) in numpy arrays and write to   file for easier access.
* For classification methods, we flatten each image (originally 200x200 pixels) into a 1x40000 feature vector for input into our neural network.
* Then shuffled the data to introduce more randomness

## Data Analysis

#### Sample image for each class
A random sample of each of the 29 classes was plotted for visual analysis.

<img width="658" alt="Screen Shot 2019-10-31 at 5 56 48 PM" src="https://user-images.githubusercontent.com/40244616/68058455-62012380-fcb6-11e9-8b38-993218b61adf.png">

#### Histogram analysis for each class
Histograms were plotted for each class to analyze the pixel intensity of the colors red, blue, and green. An example of class ‘A’ is plotted below.

<img width="401" alt="Screen Shot 2019-11-01 at 2 47 59 PM" src="https://user-images.githubusercontent.com/40244616/68058536-9f65b100-fcb6-11e9-8cd0-6cc7db76b960.png">

#### Average image for each class
We display the averages as black and white (grayscale) images because if we use the color version of the images, the averages are all white (sum of all colors = white). These show that the background is almost constant in all images and that this could lead to possible classification errors when testing the model on custom data.


<img width="493" alt="Screen Shot 2019-11-01 at 2 48 19 PM" src="https://user-images.githubusercontent.com/40244616/68058562-af7d9080-fcb6-11e9-875c-2bf14d51f685.png">

#### Average image of all classes
   <img width="165" alt="Screen Shot 2019-11-01 at 2 48 27 PM" src="https://user-images.githubusercontent.com/40244616/68058566-b1475400-fcb6-11e9-9bdc-c51d4929c844.png">


## Machine Learning Analysis


### Shallow Neural Networks - Multi-Layer Perceptron
 * We first build two simple MLPs (with one and two hidden layers) to test predictive power of these models before we perform further analysis of Deep Neural Networks.
**Multi-Layer Perceptron Model:**
* A series of dense, fully-connected layers with an input layer (of size equal to the length of our input feature vectors - in our case, 40000)
* Series of hidden layers with each neuron being connected to all neurons in the preceding layer
Output layer - with the same number of neurons as the number of classes (for a classification model)

### Deep Neural Networks - Convolution Neural Network (CNN)

We now explore more robust neural network architectures which take advantage of more complex layers than the trivial fully-connected layers.
* Convolutional Layers - Performs convolutions with the input to these layers to detect lower level features such as edges and builds up higher level features such as distinguishing between different letters of the ASL alphabet. Convolutional layers provide most of the predictive power in deep neural networks.
* Pooling Layers - We use max-pooling which looks at n x n pixels (we use 2x2)  in the image at a time and returns the maximum of these pixels as a single value. Its function is to progressively reduce the spatial size the representation to reduce the number of parameters and the amount of computation in the network.
* Flattening Layers - To reshape our data into a 1xn layer to prepare it for the final densely connected layer used for classification.

#### Architectures used
<img width="830" alt="Screen Shot 2019-11-01 at 2 56 16 PM" src="https://user-images.githubusercontent.com/40244616/68058908-cbcdfd00-fcb7-11e9-859b-4b61125f18a6.png">

### ResNet 50
I decided to implement a 50 layer Residual Network (ResNet-50) which has been proven to perform well for image classification tasks. We modify the pre-defined base model defined by Keras to include a global average pooling later and a dropout later before our final dense prediction layer. The main purpose of these two layers is to reduce the number of trainable parameters in the model and hence reduce the overall variance in the model as a method of reducing the possibility of overfitting.

This model's architecture is straight forward - it has a very deep stack of simple residual units in the middle with each residual unit composed of two convolutional layers without pooling layers. Since these CNN algorithms have deep layers, the key is to use residual learning. This type of learning is when the input (x) to the layers is also added to the output of layers higher in the stack of layers, which is referred to as skipped connections. A networks goals is to model the output h(x) but the skipped connection forces the network to model f(x) = h(x) - x. Each section of network layers that has a skip connection is considered a residual unit.

<img width="1004" alt="Screen Shot 2019-11-06 at 10 28 07 AM" src="https://user-images.githubusercontent.com/40244616/68326372-297b9400-0080-11ea-866a-8d8e2584237d.png">


### Other models
A model was by Dan Becker, found on Kaggle, trained on a deeper model with dropout layers (to reduce variance). This model performed with 97% accuracy. https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu

<img width="412" alt="Screen Shot 2019-11-01 at 2 57 52 PM" src="https://user-images.githubusercontent.com/40244616/68058975-033ca980-fcb8-11e9-809b-e59cd520779e.png">

## Conclusion and Results
As we can see, our results align with our expectations. The first two shallow neural network models are both trained much faster than the convolutional neural network models but they do not have the predictive power to make any meaningful predictions about our data set. Both of these models result in a validation-set accuracy of around 0.03-0.04 which is actually no better than a Naive Bayes classifier (i.e. if you randomly assigned an image to a class, you would be correct with a probability of approximately 1/29 ~ accuracy of 0.034).

Our convolutional neural network models, even though their architectures were chosen arbitrarily without too much thought or experimentation, clearly show much more predictive power. Even our single layer CNN model with only one convolutional layer and one pooling layer results in an accuracy of 0.5252

When we use a deep neural network that is specifically designed for image classification with multiple convolutional layers as well as the introduction of dropout layers, we see a huge jump in performance. This really showcases the power of CNNs as well as the importance of network architecture in Computer Vision.

<img width="622" alt="Screen Shot 2019-11-06 at 10 27 17 AM" src="https://user-images.githubusercontent.com/40244616/68326328-110b7980-0080-11ea-954b-706f539e504c.png">

