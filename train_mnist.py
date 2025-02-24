import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from mlp import Softmax, CrossEntropy, Layer, MultilayerPerceptron, Tanh, Relu



class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        # Read label files
        with open(labels_filepath, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f'Label file magic number mismatch, expected 2049, got {magic}')
            labels = array("B", f.read())

        # Read images file
        with open(images_filepath, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f'Image file magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", f.read())

        images = []
        for i in range(size):
            # Extract the i-th image data and reshape it into a 28x28 array.
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            images.append(img.reshape(rows, cols))

        return images, labels
    
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def main():
    training_images_filepath ='train-images-idx3-ubyte'
    training_labels_filepath = 'train-labels-idx1-ubyte'
    test_images_filepath = 't10k-images-idx3-ubyte'
    test_labels_filepath = 't10k-labels-idx1-ubyte'

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath,
                                    test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #basic information about the dataset
    print("Number of training images:", len(x_train))
    print("Number of test images:", len(x_test))
    print("Shape of the first training image:", x_train[0].shape)
    print("First 10 training labels:", list(y_train[:10]))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    #flattening the image to the vector length of 784
    x_train = x_train.reshape(x_train.shape[0], -1) 
    x_test = x_test.reshape(x_test.shape[0], -1)    
   
    # normalizing for pixel stability
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    y_train = one_hot_encode(y_train, num_classes=10)
    y_val   = one_hot_encode(y_val, num_classes=10)
    y_test  = one_hot_encode(y_test, num_classes=10)
    # Print out the shapes after splitting.
    print("After splitting:")
    print("Training set shape:", x_train.shape)   # Expected ~ (48000, 784)
    print("Validation set shape:", x_val.shape)     # Expected ~ (12000, 784)
    print("Test set shape:", x_test.shape)          # Expected (10000, 784)
    print("One-hot encoded training labels shape:", y_train.shape)  # Expected: (48000, 10)

    Layer1 = Layer(fan_in=784, fan_out=512, activation_function=Relu())
    Layer2 = Layer(fan_in=512, fan_out=256, activation_function=Relu())
    Layer3 = Layer(fan_in=256, fan_out=128, activation_function=Relu())
    Layer4 = Layer(fan_in=128, fan_out=64, activation_function=Relu())
    Layer5 = Layer(fan_in=64, fan_out=10, activation_function=Softmax())

    model = MultilayerPerceptron(layers=(Layer1, Layer2, Layer3, Layer4, Layer5))
    loss_function = CrossEntropy()
    learning_rate = 0.001
    batch_size = 32
    epochs = 100

    print("\nStarting the training...\n")
    training_losses, validation_losses = model.train(
        train_x=x_train,
        train_y=y_train,
        val_x=x_val,
        val_y=y_val,
        loss_func=loss_function,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )

    y_pred, _ = model.forward(x_test)
    predictions = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == true_labels)
    print(f"\nTest set accuracy: {accuracy * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
