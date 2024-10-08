import torch
from pytorch_pretrained_gans import make_gan


def load_pretrained_gan(gan_type='biggan', model_name='biggan-deep-256'):
    """Loads a pretrained GAN model."""
    G = make_gan(gan_type=gan_type, model_name=model_name)
    return G

if __name__ == "__main__":
    gan_model = load_pretrained_gan()
    print("Pretrained GAN model loaded successfully")