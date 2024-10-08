import os

import torch

from client import ClientGAN
from cloud import CloudServer
from dataset import (get_client_datasets, get_dataloaders,
                     load_and_save_datasets)
from edge_server import EdgeServer
from pretrained_gan import load_pretrained_gan


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    noise_multiplier = 1.0  # Set the noise multiplier for differential privacy
    max_grad_norm = 1.0  # Set the maximum gradient norm for clipping

    num_edges = 3    # Number of edge servers
    clients_per_edge = 3    # Number of clients per edge server
    num_clients = num_edges * clients_per_edge  # Total clients
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset = load_and_save_datasets()
    client_datasets = get_client_datasets(train_dataset, num_clients)
    client_loaders, test_loader = get_dataloaders(client_datasets, test_dataset)
    
    # Load pretrained GAN
    print("Loading pretrained GAN model...")
    gan_model = load_pretrained_gan()
    
    # Train each client
    print("Training clients...")
    client_params = []
    for i, data_loader in enumerate(client_loaders):
        print(f"Training client {i + 1}...")
        client_gan = ClientGAN(gan_model, noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm)
        client_params.append(client_gan.train(data_loader, device))
        torch.save(client_params[-1], f'client_{i}_params.pt')  # Save client parameters

    # Aggregate parameters at the edge server
    print("Aggregating parameters at the edge server...")
    edge_server = EdgeServer(num_clients)
    aggregated_params = edge_server.aggregate_params()
    torch.save(aggregated_params, 'edge_server_params.pt')  # Save aggregated parameters
    
    # Aggregate parameters at the cloud server
    print("Aggregating parameters at the cloud server...")
    cloud_server = CloudServer(num_edges)
    final_params = cloud_server.aggregate_edge_params()
    torch.save(final_params, 'cloud_gan_params.pt')  # Save final cloud parameters
    
    print("All processes completed successfully.")

if __name__ == "__main__":
    main()