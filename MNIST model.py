import os
import numpy as np
from PIL import Image

## Paths
train_path = r".\mnist_png\training"
test_path = r".\mnist_png\testing"
actual_test_path = r".\mnist_png\actual_testing"

result_file_path = f"411185030.txt"

## Mode change
mode = "actualtest"  

## Loading data
def load_mnist_data(base_path):
    data = []
    labels = []

    # Folder 0-9
    for label in range(10):
        folder_path = os.path.join(base_path, str(label))
        if not os.path.exists(folder_path):
            continue
        
        # Image
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                file_path = os.path.join(folder_path, file_name)
                
                # to numPy arr
                with Image.open(file_path) as img:
                    img = img.convert('L')  #  Grayscale 0-255
                    img_array = np.array(img).flatten() / 255.0  # Flatten and normalize
                
                # Append to data and labels list
                data.append(img_array)
                labels.append(label)

    # Convert to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)

    return data, labels

## One-hot encoding (number to vector) for labels
def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

## Activation Functions
# Sigmoid for hidden layer
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid Derivative for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Softmax for output layer
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-9)

## Forward pass
def forward_pass(X):
    # dot products and activations
    z_hidden = np.dot(X, weights_input_hidden) + bias_hidden
    a_hidden = sigmoid(z_hidden)

    z_output = np.dot(a_hidden, weights_hidden_output) + bias_output
    a_output = softmax(z_output)

    return a_hidden, a_output

## Loss
def cross_entropy_loss(y_true, y_pred):
    
    y_true = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true

    n_samples = y_true.shape[0]
    epsilon = 1e-9  
    
    # -log(true prob + epsilon)
    log_preds = -np.log(y_pred[range(n_samples), y_true] + epsilon)
    loss = np.sum(log_preds) / n_samples
    return loss

## Backpropagation
def backward_pass(X, y_true, a_hidden, a_output):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    
    n_samples = X.shape[0]
    
    # Ensure y_true contains integer labels
    y_true = np.argmax(y_true, axis=1) if len(y_true.shape) > 1 else y_true

    # one-hot encoded version of y_true
    y_true_one_hot = np.zeros((n_samples, output_size))
    y_true_one_hot[np.arange(n_samples), y_true] = 1

    # output layer error
    output_error = a_output - y_true_one_hot

    # Gradients for weights from hidden to output
    d_weights_hidden_output = np.dot(a_hidden.T, output_error) / n_samples
    d_bias_output = np.sum(output_error, axis=0, keepdims=True) / n_samples

    # hidden layer error
    hidden_error = np.dot(output_error, weights_hidden_output.T) * sigmoid_derivative(a_hidden)

    # Gradients for weights from input to hidden
    d_weights_input_hidden = np.dot(X.T, hidden_error) / n_samples
    d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True) / n_samples

    # Update weights and biases
    weights_input_hidden -= learning_rate * d_weights_input_hidden
    bias_hidden -= learning_rate * d_bias_hidden
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    bias_output -= learning_rate * d_bias_output

## Saving and Loading
def save_model(file_path):
    np.savez(file_path,
             weights_input_hidden=weights_input_hidden,
             bias_hidden=bias_hidden,
             weights_hidden_output=weights_hidden_output,
             bias_output=bias_output)

def load_model(file_path):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    data = np.load(file_path)
    weights_input_hidden = data['weights_input_hidden']
    bias_hidden = data['bias_hidden']
    weights_hidden_output = data['weights_hidden_output']
    bias_output = data['bias_output']
    print("Model is loaded!")
    
# Create file for prediction results
def predictions_to_file(predictions, filename):
    with open(filename, 'w') as file:
        for img_name, prediction in predictions:
            file.write(f"{img_name} {prediction}\n")


## Train and test
if mode == "train":
    # Load
    train_data, train_labels = load_mnist_data(train_path)
    train_labels_encoded = one_hot_encode(train_labels)

    # Initialize Parameters
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.01

    np.random.seed(42)
    weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
    bias_output = np.zeros((1, output_size))

    # Training loop
    epochs = 500
    batch_size = 64
    for epoch in range(epochs):
        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels_encoded = train_labels_encoded[indices]
        train_labels = train_labels[indices]

        for i in range(0, train_data.shape[0], batch_size):
            X_batch = train_data[i:i+batch_size]
            y_batch = train_labels_encoded[i:i+batch_size]
            y_batch_labels = train_labels[i:i+batch_size]  # Original labels for printing

            # Forward pass
            a_hidden, a_output = forward_pass(X_batch)

            # Compute loss
            loss = cross_entropy_loss(y_batch, a_output)

            # Backward pass
            backward_pass(X_batch, y_batch, a_hidden, a_output)

        # Print Acc
        if epoch % 10 == 0: 
            a_hidden, a_output = forward_pass(train_data)
            train_predictions = np.argmax(a_output, axis=1)
            accuracy = np.sum(train_predictions == train_labels) / train_labels.size
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Training Accuracy: {accuracy * 100:.2f}%')

    save_model(r".\mnist_model.npz")

elif mode == "test":
    load_model(r".\mnist_model.npz")

    # Load testing data
    test_data, test_labels = load_mnist_data(test_path)
    
    # Shuffle 
    indices = np.arange(test_data.shape[0])
    np.random.shuffle(indices)
    test_data = test_data[indices]
    test_labels = test_labels[indices]
    
    # Test
    batch_size = 64
    for i in range(0, test_data.shape[0], batch_size):
        X_batch = test_data[i:i+batch_size]
        y_batch_labels = test_labels[i:i+batch_size]

        # Forward pass
        _, test_output = forward_pass(X_batch)

        # Get predictions and print
        test_predictions = np.argmax(test_output, axis=1)

        if i // batch_size % 10 == 0:
            print(f"Testing Batch {i // batch_size}: Predictions: {test_predictions}, Targets: {y_batch_labels}")

    # Calculate accuracy for test set
    _, test_output = forward_pass(test_data)
    test_predictions = np.argmax(test_output, axis=1)
    accuracy = np.sum(test_predictions == test_labels) / test_labels.size
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

elif mode == "actualtest":
    load_model(r".\mnist_model.npz")

    # Prepare to store predictions
    predictions = []
    correct = 0
    total = 0

    # Iterate through all files in the test folder
    for file_name in os.listdir(actual_test_path):
        if file_name.endswith(".png"):
            file_path = os.path.join(actual_test_path, file_name)

            # Load the image
            with Image.open(file_path) as img:
                img = img.convert('L')  # Convert to grayscale
                img_array = np.array(img).flatten() / 255.0  # Flatten to 1D and normalize
                img_array = img_array.reshape(1, -1)  # Reshape to match input size (1, 784)

            # Forward pass to get prediction
            _, test_output = forward_pass(img_array)
            test_prediction = np.argmax(test_output, axis=1)[0]

            # Extract the true label from the filename if available
            # true_label = int(file_name.split(".")[0][-1])  # Assuming last character before ".png" is the label

            # Collect predictions and filenames
            predictions.append((file_name.split(".")[0], test_prediction))
                
            # Update accuracy calculations if true label is available
            # if true_label == test_prediction:
            #     correct += 1
            # total += 1

    # Calculate and print accuracy for test set
    # accuracy = correct / total * 100
    # print(f'Actual Test Accuracy: {accuracy:.2f}%')

    predictions_to_file(predictions, result_file_path)
    
    print(f"Predictions saved to {result_file_path}")


