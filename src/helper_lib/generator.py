import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def generate_samples(model, device, num_samples=10):
    model.eval()
    model.to(device)
    
    # TODO: generate num_samples points in the latent space
    with torch.no_grad():
        z = torch.randn(num_samples, 100).to(device)
        
        # run the generator to construct the image
        fake_images = model.generator(z)
        
    # plot the samples on a grid
    grid = vutils.make_grid(fake_images, nrow=5, normalize=True)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis("off")
    plt.title("Generated MNIST Samples")
    plt.show()