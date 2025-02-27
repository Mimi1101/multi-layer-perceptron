from ucimlrepo import fetch_ucirepo
from mlp import Linear, Relu, SquaredError, Layer, MultilayerPerceptron, Tanh
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting and cleaning the data
auto_mpg = fetch_ucirepo(id=9)

x = auto_mpg.data.features #only has numerical features, 7 columns
y=auto_mpg.data.targets #mpg

print("Features sample:")
print(x.head()) 
print("\nTarget sample:")
print(y.head())

data = pd.concat([x, y], axis=1)
cleaned_data = data.dropna()
x = cleaned_data.iloc[:, :-1]
y = cleaned_data.iloc[:, -1]
print("Shape of x:", x.shape) 

x_train, x_leftover, y_train, y_leftover = train_test_split(
   x, y,
    test_size=0.3,
    random_state=42,
    shuffle = True
)
x_val, x_test, y_val, y_test = train_test_split(
    x_leftover, y_leftover,
    test_size = 0.5,
    random_state = 42,
    shuffle = True
)

# Compute statistics for X (features)
x_mean = x_train.mean(axis=0)  # Mean of each feature
x_std = x_train.std(axis=0)    # Standard deviation of each feature

# Standardize X
x_train = (x_train - x_mean) / x_std
x_val = (x_val - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# Compute statistics for y (targets)
y_mean = y_train.mean()  # Mean of target
y_std = y_train.std()    # Standard deviation of target

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

x_train.shape

print(f"Samples in Training:   {len(x_train)}")
print(f"Samples in Validation: {len(x_val)}")
print(f"Samples in Testing:    {len(x_test)}")

x_train_np = x_train.to_numpy()
x_val_np = x_val.to_numpy()
x_test_np = x_test.to_numpy()
y_train_np = y_train.to_numpy().reshape(-1, 1)
y_val_np = y_val.to_numpy().reshape(-1, 1)
y_test_np = y_test.to_numpy().reshape(-1, 1)

#Initializing the layers
layer_one = Layer(fan_in=7, fan_out=32, activation_function=Relu())
layer_two = Layer(fan_in=32, fan_out=16, activation_function=Relu())
layer_three = Layer(fan_in = 16, fan_out=8, activation_function=Relu() )
layer_four = Layer(fan_in =8, fan_out=1, activation_function=Linear() )

model = MultilayerPerceptron(layers=(layer_one, layer_two, layer_three, layer_four))
loss_function = SquaredError()
learning_rate = 0.001
batch_size = 16
epochs = 200

print("\nStarting the training...\n")
training_losses, validation_losses = model.train(
    train_x=x_train_np,
    train_y=y_train_np,
    val_x=x_val_np,
    val_y=y_val_np,
    loss_func=loss_function,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs
)

#evaluating on full test set
test_predictions, _ = model.forward(x_test_np, training=False)
test_loss = np.mean(loss_function.loss(y_test_np, test_predictions))
print(f"\nFinal Test Loss: {test_loss:.4f}")

#table
np.random.seed(42)
indices = np.random.choice(range(x_test_np.shape[0]), size=10, replace=False)
selected_true = y_test_np[indices].flatten()
selected_pred = test_predictions[indices].flatten()
# de standardizing the selected true and predicted values
selected_true_orig = selected_true * y_std + y_mean
selected_pred_orig = selected_pred * y_std + y_mean

comparison_df_orig = pd.DataFrame({
    'True MPG': selected_true_orig,
    'Predicted MPG': selected_pred_orig
})
print("\nComparison Table (Original MPG Scale) for 10 Test Samples:")
print(comparison_df_orig)


#Plotting the loss curves
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs_range, training_losses, label='Training Loss', linewidth=2, color='blue')
plt.plot(epochs_range, validation_losses, label='Validation Loss', linewidth=2, color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.show()