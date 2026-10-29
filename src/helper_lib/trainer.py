import torch

def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    model.to(device)
    generator = model.generator
    discriminator = model.discriminator
    optimizer_G, optimizer_D = optimizer # Unpack optimizers for G and D
    
    # TODO: run several iterations of the training loop
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(data_loader):
            real_imgs = imgs.to(device)
            batch_size = real_imgs.size(0)
            
            # Truth labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            outputs_real = discriminator(real_imgs)
            d_loss_real = criterion(outputs_real, real_labels)
            
            z = torch.randn(batch_size, 100).to(device)
            fake_imgs = generator(z)
            outputs_fake = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(outputs_fake, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            outputs_fake_for_g = discriminator(fake_imgs)
            g_loss = criterion(outputs_fake_for_g, real_labels)
            
            g_loss.backward()
            optimizer_G.step()
            
    return model