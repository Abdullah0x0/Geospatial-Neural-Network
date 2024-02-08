import numpy as np
import sys

# Global variables
LEARNING_RATE = 0.0001
NUM_EPOCHS = 500
GLOBAL_THRESHOLD = 0.8
CONTINENTS = ["Africa", "America", "Antarctica", "Asia", "Australia", "Europe", "Arctic", "Atlantic", "Indian", "Pacific"]
BEST_WEIGHTS = None

class Neuron:
  # Constructor
  def __init__(self, continent, lat_weight=0.0, long_weight=0.0, threshold=GLOBAL_THRESHOLD):
    self.continent = continent
    self.lat_weight = lat_weight
    self.long_weight = long_weight
    self.threshold = threshold
    self.correct_pred = 0.0
    
  # Activation function that determines if the neuron fires
  def activation(self, lat_input, long_input):
    weighted_sum = lat_input * self.lat_weight + long_input * self.long_weight
    if weighted_sum >= self.threshold:
      return 1
    else:
      return 0

  # Update weights based on the error in prediction
  def update_weights(self, correct_output, lat_input, long_input):
    self.lat_weight += LEARNING_RATE * (correct_output - self.activation(lat_input, long_input)) * lat_input
    self.long_weight += LEARNING_RATE * (correct_output - self.activation(lat_input, long_input)) * long_input

  def __str__(self):
    return f"Neuron(continent={self.continent}, lat_weight={self.lat_weight}, long_weight={self.long_weight}, threshold={self.threshold})"


class InputData:
  # Constructor
  def __init__(self, file_name):
    self.data_list = []
    self.read_data(file_name)

  # Read data from a file and normalize latitude and longitude
  def read_data(self, file_name):
    with open(file_name) as f:
      for line in f:
        lat, long, continent = map(str.strip, line.split())
        # Normalize
        lat = (float(lat) - (-90)) / 180
        long = (float(long) - (-180)) / 360
        self.data_list.append((lat, long, continent))


# Train neurons using training data for multiple epochs
def train_neurons(neurons, training_data):
  for _ in range(NUM_EPOCHS):
    for lat, long, continent in training_data:
      for neuron in neurons:
        correct_output = 1 if continent == neuron.continent else 0
        neuron.update_weights(correct_output, lat, long)


# Test neurons on testing data and calculate performance metrics
def test_neurons(neurons, testing_data):
  global BEST_WEIGHTS 
  # Initialize counters for performance metrics
  correct_classified = 0
  multiple_neurons_firing = 0
  zero_neurons_firing = 0
  # Iterate through each testing data point
  for lat, long, continent in testing_data:
    fired_neurons = [neuron for neuron in neurons if neuron.activation(lat, long) == 1]
    # Check for different cases and update counters
    if len(fired_neurons) == 1 and fired_neurons[0].continent == continent:
      # Single neuron firing and correct classification
      correct_classified += 1
    elif len(fired_neurons) > 1:
      # Multiple neurons firing
      multiple_neurons_firing += 1
    elif len(fired_neurons) == 0:
      # No neurons firing
      zero_neurons_firing += 1
      # Update correct_pred for neurons that should have fired but didn't
      for neuron in neurons:
        if neuron.continent == continent:
          neuron.correct_pred += neuron.activation(lat, long)
  # Calculate total samples and accuracy
  total_samples = len(testing_data)
  accuracy = correct_classified / total_samples

  # Update BEST_WEIGHTS if the current accuracy is better than the stored one
  if BEST_WEIGHTS is None or accuracy > BEST_WEIGHTS['accuracy']:
    BEST_WEIGHTS = {
      'accuracy': accuracy,
      'weights': [(neuron.lat_weight, neuron.long_weight) for neuron in neurons],
      'continents': [neuron.continent for neuron in neurons]
    }
      # Save the best weights, continents to a file
    with open("BestWeights.txt", 'w') as f:
      for (lat_weight, long_weight), continent in zip(BEST_WEIGHTS['weights'], BEST_WEIGHTS['continents']):
        f.write(f"{lat_weight} {long_weight} {continent}\n")
    print("\nBest learned weights, continents stored into file: BestWeights.txt")

  # Print performance metrics
  print("\nResults:")
  print(f"Percentage of correctly classified examples: {accuracy * 100}%")
  print(f"Percentage of examples with multiple neurons firing: {multiple_neurons_firing / total_samples * 100}%")
  print(f"Percentage of examples with zero neurons firing: {zero_neurons_firing / total_samples * 100}%")
  
  return accuracy


# Print performance metrics for each neuron
def print_performance_metrics(neurons, testing_data):
  for neuron in neurons:
    true_positives = sum(neuron.activation(lat, long) == 1 and continent == neuron.continent for lat, long, continent in testing_data)
    true_negatives = sum(neuron.activation(lat, long) == 0 and continent != neuron.continent for lat, long, continent in testing_data)
    false_positives = sum(neuron.activation(lat, long) == 1 and continent != neuron.continent for lat, long, continent in testing_data)
    false_negatives = sum(neuron.activation(lat, long) == 0 and continent == neuron.continent for lat, long, continent in testing_data)

    total_samples = len(testing_data)
    accuracy = (true_positives + true_negatives) / total_samples * 100
    
    print("\n")
    print(f"Neuron: {neuron.continent}")
    print(f"   Correct: {accuracy:.2f}%")
    print(f"   True Positives: {(true_positives / total_samples) * 100:.2f}%")
    print(f"   True Negatives: {(true_negatives / total_samples) * 100:.2f}%")
    print(f"   False Positives: {(false_positives / total_samples) * 100:.2f}%")
    print(f"   False Negatives: {(false_negatives / total_samples) * 100:.2f}%")


# Store learned weights into a file
def store_weights(neurons, file_name):
  with open(file_name, 'w') as f:
    for neuron in neurons:
      f.write(f"{neuron.lat_weight} {neuron.long_weight} {neuron.continent} \n")
  print("Learned weights stored into a file.")


# Load initial weights from a file
def load_weights(neurons, file_name):
  with open(file_name) as f:
    # Iterate through each line of the file and each neuron in the list
    for line, neuron in zip(f, neurons):
      # Split the line into stripped components: latitude weight, longitude weight, continent.
      lat_weight, long_weight, continent = map(str.strip, line.split())
      neuron.lat_weight = float(lat_weight)
      neuron.long_weight = float(long_weight)
      # Assuming continent is a string, no need to convert it and update the continent.
      neuron.continent = continent


# Function to iniatialize weights randomly.
def initialize_random_weights(neurons):
  for neuron in neurons:
    # Initialize neurons with random weights
    neuron.lat_weight = np.random.rand()
    neuron.long_weight = np.random.rand()
  print("Neurons initialized with random weights.")

  
def main():
  print("Artificial Neuron Network\n")
  # Create Neuron objects for each continent in the CONTINENTS list
  neurons = [Neuron(continent) for continent in CONTINENTS]
  for neuron in neurons:
    print(neuron)
  print("\n")

  while True:
    print("\nMenu:")
    print("1. Train the neural network")
    print("2. Store learned weights into a file")
    print("3. Load initial weights from a file or initialize randomly")
    print("4. Test the neural network")
    print("5. Exit")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
      training_data = InputData(input("Enter the filename for training data: "))
      train_neurons(neurons, training_data.data_list)
      print("Neural network trained successfully!")
    elif choice == "2":
      # Store the learned weights into a file (you can choose file name).
      file_name = input("Enter the filename to store learned weights: ")
      store_weights(neurons, file_name)
    elif choice == "3":
      # Load weights from a file or randomly generate weights.
      option = input("Enter 'file' to load from a file, 'random' to initialize randomly: ").lower()
      if option == 'file':
        file_name = input("Enter the filename with initial weights: ")
        load_weights(neurons, file_name)
        print("Initial weights loaded from ", file_name)
      elif option == 'random':
        initialize_random_weights(neurons)
        print("Weights initialized randomly.")
      else:
        print("Invalid option. Please enter 'file' or 'random'.")
    elif choice == "4":
      testing_data = InputData(input("Enter the filename for testing data: "))
      print_performance_metrics(neurons, testing_data.data_list)
      test_neurons(neurons, testing_data.data_list)
    elif choice == "5": 
      print("Exiting the program.")
      break
    else:
      print("Invalid choice. Please enter a number between 1 and 5.")
        
# Execute the main function
main()