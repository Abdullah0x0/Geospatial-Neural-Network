# Geospatial Neural Network

This project features a Geospatial Perceptron Neural Network designed to classify geographical coordinates into continents or oceans. The network employs perceptron learning, a fundamental neural network algorithm, to train individual neurons representing both continents and major oceans to recognize and classify input coordinates based on their geospatial properties.

## Features

- **Broad Classification Scope**: Capable of classifying coordinates into continents and major oceans, enhancing the network's utility for geospatial analysis.
- **Perceptron Learning**: Utilizes the perceptron learning algorithm for training neurons, laying the groundwork for understanding more complex neural network structures.
- **Adaptive Learning**: Neurons adjust their weights through learning, improving their ability to accurately classify geospatial data over time.
- **Normalized Geospatial Inputs**: Latitude and longitude inputs are normalized to optimize the network's learning and classification processes.
- **Interactive User Experience**: An interactive menu facilitates the training process, weight management, network testing, and other functionalities, making the system user-friendly.

## Requirements

- Python 3.x
- NumPy

## Setup Instructions

1. Verify the installation of Python 3.x on your system. If it's not installed, you can download it from the [Python website](https://www.python.org/).
2. Use pip to install the NumPy package, which is essential for numerical computations:
   ```
   pip install numpy
   ```

## How to Use

1. Prepare your training and testing datasets with geographical coordinates. Each data point should be labeled with the corresponding continent or ocean.
2. Run the `neural-network.py` script:
   ```
   python neural-network.py
   ```
3. Navigate through the options in the interactive menu to train the network, manage weights, test the classifier, or exit the application.

## Data Format

Your data should be structured as follows, where each line in your dataset represents one data point:
```
latitude longitude label
```
For instance:
```
-33.92487 18.424055 Africa
0 0 Atlantic Ocean
```
