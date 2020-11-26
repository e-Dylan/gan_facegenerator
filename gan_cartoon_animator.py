from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
from matplotlib import pyplot as plt
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

DATA_PATH = f"data/celeba/"

LR = 0.0002
BS = 64
IMAGE_SIZE = 64
IMG_CHANNELS = 3
EPOCHS = 5

# my_transforms = transforms.Compose([
# 	transforms.Resize(IMAGE_SIZE),
# 	transforms.CenterCrop(IMAGE_SIZE),
# 	transforms.ToTensor(),
# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])
print(f"Preparing data, b=[{BS}].")
dataset = dset.ImageFolder(root=DATA_PATH, transform=transforms.Compose([
								transforms.Resize(IMAGE_SIZE),
								transforms.CenterCrop(IMAGE_SIZE),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=True)
num_batches = len(dataloader)
print(f"Finished preparing {num_batches} batches, b=[{BS}]")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

nz = 100
ngf = 64
ndf = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		
		self.net = nn.Sequential(
			# input z = noise, latent vector
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            # ngf*8 x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            # ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # ngf*2 x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
            # nc x 64 x 64
		)

	def forward(self, x):
		return self.net(x)

netG = Generator().to(device)
netG.apply(weights_init)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.net = nn.Sequential(
			# IMAGES_CHANNELS x 64 x 64
            nn.Conv2d(IMG_CHANNELS, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*2 x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*4 x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf*8 x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
		)
	
	def forward(self, x):
		return self.net(x).view(-1, 1).squeeze(1)

netD = Discriminator().to(device)
netD.apply(weights_init)

loss_function = nn.BCELoss()

fixed_noise = torch.randn(BS, nz, 1, 1, device=device) # used as input to generator, makes images.
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))

# training_data = np.load(PIC_DIR+"image_data-2000.npy")

print("Running on GPU" if torch.cuda.is_available() else print("Running on CPU Slow."))

# for i, data in enumerate(dataloader):
# 	print(data[0].shape)
# 	break

writer_real = SummaryWriter(f"logs/FACEGEN/test_real")
writer_fake = SummaryWriter(f"logs/FACEGEN/test_fake")

print("starting training...")
print(f"Batches: {num_batches}. Batch Size: {BS}. EPOCHS: {EPOCHS}. LR: {LR}.")

def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# image_file: tensor of images.
# saves training image progression as gif.
import imageio
import shutil 
images_to_gif = []
def save_training_images(image_file):
	# convert image_file tensor -> np array
	# image_file = image_file.numpy()
	iters = 0
	for i in image_file:
		iters += 1
		if iters % 5 == 0:
			print('images appended: ', iters)
			images_to_gif.append(i.permute(1,2,0).numpy())
		else:
			pass
	imageio.mimsave('./training_visual.gif', images_to_gif)

save_training_images(torch.load('training_img_grid.pt'))

# param: image_file -> torch.Tensor file (.pt) containing images.
# param: delay -> delay (ms) between each image.
def play_training_images(image_file, delay):
	image_file = image_file.numpy()
	# image_file1 = image_file[:len(image_file)//6] # first quarter
	# image_file2 = image_file[-len(image_file)//4:] # fourth half
	# image_file = np.concatenate((image_file1, image_file2))
	print(len(image_file))
	iters = 0
	for i in image_file:
		iters += 1
		print(f"tensor shape: {i.shape}, training progress index: {iters}")
		img = i.transpose(1,2,0)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (750, 750))
		cv2.imshow('training images', img)
		k = cv2.waitKey(delay)
		if k == ord('q'):
			break

# play_training_images(torch.load('training_img_grid.pt'), 30)

def generate_image(model_file):
	netG = Generator().to(device)
	netG.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
	noise = torch.randn(64, nz, 1, 1, device=device)
	fake = netG(noise)
	print("genning")
	img = vutils.make_grid(fake, padding=2, normalize=True)
	plt.figure(figsize=(12,12))
	plt.axis('off')
	plt.imshow(img.detach().numpy().transpose(1,2,0))
	plt.show()

# generate_image(model_file='models/netG_EPOCHS=10_IMGSIZE=64.pth')

def train():
	start_time = time.time()
	G_losses = []
	D_losses = []
	for epoch in range(EPOCHS):
		for batch_idx, data in enumerate(dataloader, 0):

			# [1] train D with real data
			netD.zero_grad()
			real_batch = data[0].to(device)
			batch_size = real_batch.size(0)
			labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
			output = netD(real_batch) # D's guess on real image.
			lossD_real = loss_function(output, labels) # loss on real images -> teach real = 1.
			lossD_real.backward() # CALCULATE WEIGHT ADJUSTMENTS WITH REAL'S GRADIENTS.
			D_x = output.mean().item() # D(x) mean guess on real data.

			# [1] train D with fake's data
			noise = torch.randn(batch_size, nz, 1, 1, device=device)
			fake = netG(noise) # generated image
			labels.fill_(fake_label)
			output = netD(fake.detach()) # just want to show D what a fake image is, don't want to learn from G's gradients
			lossD_fake = loss_function(output, labels)
			lossD_fake.backward() # CALCULATE WEIGHT ADJUSTMENTS FOR FAKE'S GRADIENTS.
			D_Gz_1 = output.mean().item()
			# combine (real + fake) loss:
			lossD = lossD_real + lossD_fake
			optimizerD.step()

			# [2] train generator with new updated Discriminator
			netG.zero_grad()
			labels.fill_(real_label)
			output = netD(fake) # D(G(z)), move weights to increase discriminator's output, make think real.
			lossG = loss_function(output, labels)
			lossG.backward() # CALCULATE WEIGHT ADJUSTMENTS BASED ON WHAT DISCRIMINATOR COULD TELL.
			D_Gz_2 = output.mean().item() # discriminator's prediction AFTER G has improved.
			optimizerG.step()

			if batch_idx % 1 == 0:
				print(f"[e{epoch}/{EPOCHS}] [b{batch_idx}/{num_batches}] lossD: {lossD.item():.4f}, lossG: {lossG.item():.4f}, D(x): {D_x:.4f}, D(G(z)): {D_Gz_1:.4f}/{D_Gz_2:.4f}")

			G_losses.append(lossG)
			D_losses.append(lossD)

			# TENSORBOARD WRITER
			if batch_idx % 10 == 0:
				vutils.save_image(data[0], './real_samples.png', normalize=True)
				fake = netG(fixed_noise)
				vutils.save_image(fake.detach(), f'./fake_samples_epoch_{epoch}.png', normalize=True)

				with torch.no_grad():
					fake = netG(fixed_noise)
					img_grid_real = vutils.make_grid(data[0][:32], normalize=True)
					img_grid_fake = vutils.make_grid(fake[:32], normalize=True)
					writer_real.add_image("Real Images", img_grid_real)
					writer_fake.add_image("Fake Generator Images", img_grid_fake)

	torch.save(netG.state_dict(), f'./netG_EPOCHS={EPOCHS}.pth')
	torch.save(netD.state_dict(), f'./netD_EPOCHS={EPOCHS}.pth')

	train_time = time.time() - start_time
	print(f"Finished training on {num_batches} batches. Training time: {train_time}")

	plt.figure(figsize=(10,5))
	plt.title("FACEGEN G/D Training Loss")
	plt.plot(G_losses, label="G")
	plt.plot(D_losses)
	plt.xlabel("steps")
	plt.ylabel("loss")
	plt.legend()
	plt.show()

if __name__ == "__main__":
	# train()
	# image_tensor = torch.load('training_img_grid.pt')
	# play_training_images(image_tensor)
	pass
