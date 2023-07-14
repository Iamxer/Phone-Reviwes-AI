# Neural Network for Text-Based Rating Prediction
This GitHub repository hosts a neural network model designed for predicting ratings based on textual inputs. The model is trained using a five-point rating system, enabling it to analyze the content of a given text and estimate an appropriate rating.

# Usage
To use the neural network model, follow these steps:

Navigate to the desired language version directory within the repository.
- Run the main.py file.
  Upon running the main.py file, you will be presented with a menu that provides the following options:

  0 - Run the pre-trained neural network.
  
  1 - Generate and save a new neural network.
  
  It is recommended to use the pre-trained neural network for efficient rating prediction. However, if you wish to generate and save a new neural network, you can choose the second option.

Please note that this repository does not incorporate any advanced techniques or make use of other neural network models. It focuses on the implementation of a basic TensorFlow-based neural network for rating prediction.

# Requirements
To successfully run the code in this repository, ensure that you have the following libarys installed:

Python(3.xx) with:
- TensorFlow
- Pandas
- Keras_preprocessing
- Numpy
Make sure to meet these requirements to avoid any compatibility issues.

# Versions

## v0.1

  - This is the first and earliest version of the neural network, it is very simple and works best on very short reviews consisting of a ~100 symbols, more or less than make it is unpredictable. 
  - Available in English and Russian language

## v0.2

  - Unlike the previous version, it has longer training time and slower learning speed, which gives more accurate prediction results for the neural network.
  - Only available in Russian language

## v0.3

  - This version consists of three different models: for short texts (0 - 150 characters), for medium (150 - 350 characters) and for large (350 - 600 characters), which are able to switch automatically when the neural network works, despite expectations the model was a complete failure and is almost unable to work, presumably because of the large variation in text size and a small database for each model (only 10,000 examples per model as opposed to 15,000 in previous versions).
  - Only available in Russian language

# Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.
