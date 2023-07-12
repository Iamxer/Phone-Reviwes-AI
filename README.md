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
  
### _Types:_
 - ### a
      A standard model that achieves an optimal ratio of time to accuracy.
 - ### b
      This model has greater accuracy due to longer training time and lower learning speed.

## v0.1

  - This is the first and earliest version of the neural network, it is very simple and works best on very short reviews consisting of a ~100 symbols, more or less than make it is unpredictable. 
  - Only available in Russian language

# Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.
