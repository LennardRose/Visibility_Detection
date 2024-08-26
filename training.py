import torch
from tqdm import trange
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import utils


def train_model(model, criterion, optimizer, device, batch_size, epochs, verbose):
    """
    Executes the training for a given model, initializes the data, Trains the model and validates the training,
    records and outputs the progress.
    :param model: The ANN Model to train.
    :param criterion: The Loss/error function (for regression).
    :param optimizer: The method to update the weights.
    :param device: The device to execute the training/validation on (cpu/gpu).
    :param batch_size: Batch size for the dataloader to yield.
    :param epochs: Number of epochs to train the model.
    :param verbose: Set True to output every epoch's loss/metrics.
    :return: Training and Validation losses and additional metrics.
    """

    # Get training and validation dataloaders
    training_dataloader, val_dataloader = utils.get_train_test_loaders(csv_file=utils.config["csv_path"],
                                                                       img_dir=utils.config["img_path"],
                                                                       batch_size=batch_size)

    training_losses = []
    validation_losses = []
    train_mse_scores = []
    val_mse_scores = []

    for epoch in trange(epochs, desc="Epochs"):
        # Training
        model.train()
        epoch_train_loss = 0
        y_pred_train = []
        y_true_train = []

        for batch, labels in training_dataloader:
            torch.cuda.empty_cache()
            batch = batch.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(batch)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            # Track loss
            epoch_train_loss += loss.item()

            # Store predictions and true values for metrics
            y_pred_train.extend(output.detach().cpu().numpy())
            y_true_train.extend(labels.detach().cpu().numpy())

        # Calculate metrics for training
        avg_train_loss = epoch_train_loss / len(training_dataloader)
        train_mse = mean_squared_error(y_true_train, y_pred_train)
        training_losses.append(avg_train_loss)
        train_mse_scores.append(train_mse)

        # Validation
        model.eval()
        epoch_val_loss = 0
        y_pred_val = []
        y_true_val = []

        with torch.no_grad():
            for batch, labels in val_dataloader:
                torch.cuda.empty_cache()
                batch = batch.to(device)
                labels = labels.to(device)

                # Forward pass
                output = model(batch)
                loss = criterion(output, labels)

                # Track validation loss
                epoch_val_loss += loss.item()

                # Store predictions and true values for metrics
                y_pred_val.extend(output.detach().cpu().numpy())
                y_true_val.extend(labels.detach().cpu().numpy())

        # Calculate metrics for validation
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_mse = mean_squared_error(y_true_val, y_pred_val)
        validation_losses.append(avg_val_loss)
        val_mse_scores.append(val_mse)

        # Output progress
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Training Loss: {round(avg_train_loss, 3)} - MSE: {round(train_mse, 3)}")
            print(f"Validation Loss: {round(avg_val_loss, 3)} - MSE: {round(val_mse, 3)}")

    return training_losses, train_mse_scores, validation_losses, val_mse_scores
