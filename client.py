import torch
from torch import nn

from dataset import (get_client_datasets, get_dataloaders,
                     load_and_save_datasets)
from pretrained_gan import load_pretrained_gan


class ClientGAN:
    def __init__(self, gan_model, noise_multiplier=1.0, max_grad_norm=1.0):
        self.generator = gan_model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    # Inside the ClientGAN class
    def train(self, data_loader, device):
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002)

        for epoch in range(10):  # Example number of epochs
            for images, _ in data_loader:
                images = images.to(device)
                optimizer.zero_grad()

                # Generate noise and samples
                z = self.generator.sample_latent(batch_size=images.size(0), device=device)
                y = self.generator.sample_class(batch_size=images.size(0), device=device)
                generated_images = self.generator(z=z, y=y)  # Should be 256x256

                # Example loss function (MSE)
                loss = ((generated_images - images) ** 2).mean()
                loss.backward()

                # Clip gradients
                nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)

                # Add Gaussian noise to gradients
                for param in self.generator.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, self.noise_multiplier * self.max_grad_norm, size=param.grad.shape).to(device)
                        param.grad += noise

                optimizer.step()

        return self.generator.state_dict()  # Return the trained parameters

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_clients = 3  # Adjust as needed

    # Load datasets
    train_dataset, test_dataset = load_and_save_datasets()
    client_datasets = get_client_datasets(train_dataset, num_clients)
    client_loaders, test_loader = get_dataloaders(client_datasets, test_dataset)

    # Load pretrained GAN
    gan_model = load_pretrained_gan()

    # Train each client
    for i, data_loader in enumerate(client_loaders):
        client_gan = ClientGAN(gan_model)
        client_params = client_gan.train(data_loader, device)
        torch.save(client_params, f'client_{i}_params.pt')  # Save client parameters