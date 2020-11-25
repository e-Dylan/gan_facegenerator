import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import torchvision.utils as vutils
import copy

data_path = 'data/celeba/img_align_celeba'
IMG_SIZE = 128
IMG_CHANNELS = 3
DATA_SIZE = 10000

# data_size: None defaults to all, any amount specifies how many files to take.
def save_images_from_folder(folder, data_size=None):
	images_data = []
	save_idx = 1

	img_idx = 0
	for filename in os.listdir(folder):
		if (data_size is not None) and (img_idx > data_size):
			np.save(f"image_training_data-{save_idx}_IMGSIZE={IMG_SIZE}-SIZE={data_size}", images_data)
			images_data = []
			save_idx += 1
			img_idx = 0

		img = cv2.imread(os.path.join(folder, filename))
		img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		if (img is not None):
			images_data.append(img)
			print(f"appended image {filename} at idx: {img_idx}")

		img_idx += 1
	# return images_data

# @param training_data: array of 64 x 64 images in the form of numpy arrays.
# @param shuffle: whether or not images should be shuffled before putting into batches.
def make_batches(training_data, batch_size, shuffle):
	batch_data = []
	training_data = np.load(f"image_training_data-1_IMGSIZE={IMG_SIZE}-SIZE={DATA_SIZE}.npy")

	if shuffle:
		random.shuffle(training_data)

	# define an empty batch.
	image_batch = np.empty((batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), dtype="float64")
	batch_idx = 0
	for i in range(len(training_data)):
		if batch_idx % batch_size == 0 and batch_idx != 0:
			# finished a batch
			image_batch = np.reshape(image_batch, (batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE))
			batch = copy.deepcopy(torch.from_numpy(image_batch))
			batch_data.append(batch)
			print(len(batch_data))
			batch_idx = 0

		# append after we've checked if the batch is full
		image = np.transpose(training_data[i], (2,0,1))
		# print(image.shape)
		image_batch[batch_idx] = image
		batch_idx += 1

	return batch_data

image_data = save_images_from_folder(data_path, data_size=DATA_SIZE)
# img = np.load("image_training_data-1_IMGSIZE={IMG_SIZE}-SIZE=50000.npy", allow_pickle=True)
# print(img.shape)
# plt.imshow(img[0])
# plt.show()

# image_data = np.load("image_data-2000.npy", allow_pickle=True)
# print(len(image_data))

# batch_data = make_batches(training_data=[], batch_size=128, shuffle=True)
# print(len(batch_data))

# np.save(f"image_data-{DATA_SIZE}.npy", image_data)



