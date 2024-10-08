import torch
from torchmetrics import FID, InceptionScore


def test_gan(global_gan, test_loader, device):
    inception_score = InceptionScore().to(device)
    fid = FID().to(device)
    
    global_gan.generator.eval()
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            z = global_gan.generator.sample_latent(batch_size=images.size(0), device=device)
            y = global_gan.generator.sample_class(batch_size=images.size(0), device=device)
            generated_images = global_gan.generator(z, y)
            
            inception_score.update(generated_images)
            fid.update(generated_images, real=False)
            fid.update(images, real=True)
    
    return {
        'Inception Score': inception_score.compute(),
        'FID Score': fid.compute()
    }