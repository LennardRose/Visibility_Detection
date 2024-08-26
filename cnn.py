import torch.nn as nn
import torch.optim as optim
import utils
import training

def init_cnn(device, input_channel, out_channels, output_size):
    """
    Initialize the CNN model for regression.

    :param device: PyTorch device (CPU or CUDA)
    :param input_channel: Number of input channels (1 for greyscale images)
    :param out_channels: List of integers specifying the number of channels for each Conv layer
    :param output_size: Number of outputs
    :return: The initialized CNN model
    """
    layers = []

    # input layer
    previous_channels = input_channel

    # hidden layers
    for out_channel in out_channels:
        layer = nn.Sequential(
            nn.Conv2d(previous_channels, out_channel, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        layers.append(layer)
        previous_channels = out_channel

    # Prepare for linear layers: calculate the flattened size after convolutions and pooling
    layers.append(nn.Flatten())

    # Image size is reduced by a factor of 2 per MaxPool layer.
    # Initial image size is 100x100; we reduce it by (1/2)^(len(out_channels)) for each layer.
    scale_factor = 100 // (2 ** len(out_channels))  # Factor to reduce the input size

    # Final linear layers
    layers.append(nn.Linear(in_features=scale_factor * scale_factor * previous_channels, out_features=100))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=100, out_features=output_size))

    # Build the model
    cnn = nn.Sequential(*layers)
    cnn.to(device)

    return cnn


def cnn_train(out_channels, epochs, batch_size, learning_rate, cuda, plot, verbose):
    """
    Trains a CNN model for regression on two target values and greyscale 100x100 image data

    :param out_channels: List of integers specifying the number of channels for each Conv layer
    :param learning_rate: Learning rate for the optimizer
    :param cuda: Boolean indicating whether to use GPU (True) or CPU (False)
    :param epochs: Number of epochs for training
    :param batch_size: Batch size for training
    :param plot: Boolean to indicate whether to plot the training/validation metrics
    :param verbose: Boolean to indicate whether to print training progress
    :return: The trained model, training/validation losses and accuracies
    """
    # Set the device
    device = utils.get_device(cuda)

    # Define the input and output sizes
    input_channel = 1  # Greyscale input
    output_size = 2  # Two continuous output values

    # Initialize the CNN model
    cnn = init_cnn(device=device,
                   input_channel=input_channel,
                   output_size=output_size,
                   out_channels=out_channels)

    # Train the model
    train_loss, train_acc, val_loss, val_acc = training.train_model(
        model=cnn,
        criterion=nn.MSELoss(),  # Loss function for regression
        optimizer=optim.Adam(params=cnn.parameters(), lr=learning_rate),
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose
    )

    # plot training/validation results
    if plot:
        utils.plot_training(
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dims=out_channels,
            batch_size=batch_size,
            training_losses=train_loss,
            training_accuracies=train_acc,
            validation_losses=val_loss,
            validation_accuracies=val_acc
        )

    # Return the trained model and training results
    return cnn, (train_loss, val_loss), (train_acc, val_acc)
