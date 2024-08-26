import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
import json
import os
import random

plt.rcParams.update({'font.size': 14})

with open("config.json") as json_file:
    config = json.load(json_file)

# CSV file has the header : filename, perspective_score_hood, perspective_score_backdoor_left
# Example row: eb13d082-9bb7-44e9-8e2e-1b39c338ea73.jpg, 0.0, 0.0
# Overall 4000 rows

# images in the folder are named corresponding to filename
# colored images with 674 x 506 pixels


class CustomImageDataset(Dataset):
    def __init__(self, annotations, img_dir, transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotations)


    def __getitem__(self, idx):
        # Get the filename from the CSV and generate the file path
        filename = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{filename}")

        # Load the image and convert to greyscale
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get labels y1 and y2
        y1 = self.annotations.iloc[idx, 1]
        y2 = self.annotations.iloc[idx, 2]

        # Convert labels to tensor
        labels = torch.tensor([y1, y2], dtype=torch.float32)

        return image, labels


def get_train_test_loaders(csv_file, img_dir, test_size=0.2, batch_size=32, shuffle=True):
    # Load the CSV file into a pandas DataFrame
    annotations = pd.read_csv(csv_file)

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(annotations, test_size=test_size, random_state=42, shuffle=True)

    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Resize images to 100x100 pixels, TODO this could be a parameter
        transforms.Grayscale(num_output_channels=1),  # Convert to greyscale explicitly (though already done)
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize greyscale image - This centers the pixel values (which range from 0 to 1 after ToTensor()) around 0 with a range of -1 to 1.
    ])

    # Create datasets for training and testing
    train_dataset = CustomImageDataset(annotations=train_df, img_dir=img_dir, transform=transform)
    test_dataset = CustomImageDataset(annotations=test_df, img_dir=img_dir, transform=transform)

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_device(cuda):
    """
    get the device to train on, only if available!
    :param cuda: True if GPU, False if CPU
    :return:
    """
    if cuda and torch.cuda.is_available():
        # Clear cache if non-empty
        torch.cuda.empty_cache()
        # See which GPU has been allotted
        print(f"Using cuda device: {torch.cuda.get_device_name(torch.cuda.current_device())} for training")
        return "cuda"
    else:
        print("Using cpu for training")
        return "cpu"


def plot_training(epochs, learning_rate, hidden_dims, batch_size,
         training_losses, validation_losses, training_accuracies, validation_accuracies):
    """
    Plot 4 Subplots consisting of the Training and Validation losses and accuracies
    :param epochs:  The learning rate used during the training
    :param learning_rate: The learning rate used during the training
    :param hidden_dims: the hidden dimensions used for training, in case of cnn the output channels
    :param batch_size:  The learning rate used during the training
    :param training_losses: The training losses to display as a list
    :param validation_losses:  The Validation losses to display as a list
    :param training_accuracies:  The training accuracies to display as a list
    :param validation_accuracies:  The Validation accuracies to display as a list
    """

    fig, axs = plt.subplots(2, 2, figsize=(20, 10), constrained_layout=True)
    fig.suptitle(f'Epochs: {epochs} LR: {learning_rate} Hidden Layers: {hidden_dims} Batch Size: {batch_size}')

    # losses
    axs[0, 0].plot(range(epochs), training_losses)
    axs[0, 0].set_title("Training Losses")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].set_xlabel("Epochs")
    axs[1, 0].plot(range(epochs), validation_losses)
    axs[1, 0].set_title("Validation Losses")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].set_xlabel("Epochs")

    # accuracies
    axs[0, 1].plot(range(epochs), training_accuracies)
    axs[0, 1].set_title("Training Accuracies")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].set_xlabel("Epochs")
    axs[1, 1].plot(range(epochs), validation_accuracies)
    axs[1, 1].set_title("Validation Accuracies")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].set_xlabel("Epochs")

    plt.show()



def save_model(model, model_dir=config["model_dir"], model_name='cnn_model.pth'):
    # Ensure the directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Define the full path for the model
    model_path = os.path.join(model_dir, model_name)

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")


def load_model(model, model_dir=config["model_dir"], model_name='cnn_model.pth', device='cpu'):
    # Define the full path for the model
    model_path = os.path.join(model_dir, model_name)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"Model loaded from {model_path}")

    return model


def get_random_filenames(folder_path, num_files=1):
    """
    Get a list of random filenames from the specified folder.

    :param folder_path: Path to the folder containing files.
    :param num_files: Number of random filenames to return.
    :return: List of random filenames.
    """
    # List all files in the folder
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Check if the number of files requested is more than available
    if num_files > len(all_files):
        raise ValueError("Number of requested files exceeds the number of available files in the folder.")

    # Get random filenames
    random_files = random.sample(all_files, num_files)

    return random_files

