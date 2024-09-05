# Handwritten Digit Recognition using MNIST Dataset


This project demonstrates a neural network model built to recognize handwritten digits from the MNIST dataset. The model is trained to classify images of digits (0-9) and can make predictions on custom images written by the user. Any contributions are welcome, submit a pull request with a description of the changes and stick to the code structure.


## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Testing with Custom Images](#testing-with-custom-images)



## Requirements

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow numpy opencv-python matplotlib
```
## Training the Model

The training script `model.py` includes data preprocessing, model building, training, and saving the model. I may in the future make this actually user friendly and split the model files into easier to understand and runnable componenets.

## Testing with Custom Images

Place a grayscale image in the `tests/` directory. Ensure the image is 28x28 pixels you can do this in microsoft paint. Replace the img url with your image.


