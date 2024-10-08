import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = './data'

# Define transformation for data normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_and_save_datasets():
    if not os.path.exists(os.path.join(DATA_DIR, 'train.pt')) or not os.path.exists(os.path.join(DATA_DIR, 'test.pt')):
        # Load MNIST, Fashion-MNIST, EMNIST datasets
        mnist = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        fashion_mnist = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        #emnist = datasets.EMNIST(root=DATA_DIR, split='balanced', train=True, download=True, transform=transform)

        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset([mnist, fashion_mnist])

        # Split dataset into train and test
        train_size = int(0.8 * len(combined_dataset))
        test_size = len(combined_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])

        # Save the datasets
        torch.save(train_dataset, os.path.join(DATA_DIR, 'train.pt'))
        torch.save(test_dataset, os.path.join(DATA_DIR, 'test.pt'))
    else:
        train_dataset = torch.load(os.path.join(DATA_DIR, 'train.pt'))
        test_dataset = torch.load(os.path.join(DATA_DIR, 'test.pt'))

    return train_dataset, test_dataset

def get_client_datasets(train_dataset, num_clients):
    """Divides the train dataset into unique subsets for each client."""
    client_datasets = []
    indices = np.random.permutation(len(train_dataset))
    split_size = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        client_indices = indices[i * split_size: (i + 1) * split_size]
        client_datasets.append(Subset(train_dataset, client_indices))
    
    return client_datasets

def get_dataloaders(client_datasets, test_dataset, batch_size=64):
    """Creates DataLoaders for clients and test dataset."""
    client_loaders = [DataLoader(client_data, batch_size=batch_size, shuffle=True) for client_data in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader