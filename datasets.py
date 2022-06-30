import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

import numpy as np
from PIL import Image

'''
This code takes in object from data.py classes and structure them into few-shot learning format
'''

'''
==== Pseudocode / High level Idea ====
Train loading:
1. Random a 0/1 siamese label
2. __getitem__() receives a randomed index as training data1 index from dataloader
3. Choose data2 index wrt to siamese label: if siamese label=1, choose from same class as data1, else, choose from other class

Test loading:
- Alternates the loaded data into +ve and -ve pairs: [idx1, idx2, binary]


==== FUTURE UPDATE(?) ====
---- Maximizing data ----
Instead of using N number of samples each epoch, use N_C_2 samples instead

Method:
Computes itertool combination based on total samples(N_C_2)
[(1,2),(5,65),....]

Samples the training pair index from len(N_C_2)
train_index_sampler = SubsetRandomSampler(np.arange(n_training_PAIRS, dtype=np.int64))

__len__ of dataset becomes len(N_C_2)

Constraints: each class need to have same num samples
'''

class SiameseAFM(Dataset):
	"""
	Train: For each sample creates randomly a positive or a negative pair
	Test: Creates fixed pairs for testing

	FUTURE UPDATE: 
	- Directly inherit the AFM class?
	- Create more testing pairs?
	"""

	def __init__(self, dataset):

		self.dataset = dataset
		self.train = self.dataset.train
		# self.transform = self.dataset.transform # Used torch.tensor() instead as ToTensor() takes in 2/3 dim data
		self.data = self.dataset.data

		if self.train:
			print('Structuring training data in siamese format...')
		else:
			print('Structuring testing data in siamese format...')

		# self.classes = self.dataset.classes # unique targets in INT
		# self.class_to_idx = self.dataset.class_to_idx # {'0':0, '1':1, '2':2, ...}

		self.labels = np.array([str(t) for t in self.dataset.targets]) # str version of self.dataset.targets
		self.labels_set = set(self.labels) # Unique STR labels:: {'0','1','2',...}

		# NOTE: label_to_indices =/= class_to_idx from torchvision dataset imports (label_to_indices=ALL index to specific class, class_to_idx=assigned numeric label to each class)
		# Dictionary of {'0':array([0,2,5,6]), '1':array([1,3,4,7]), '2':array([8,9,10,11]),...}
		self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

		if not self.train:
			# generate fixed pairs for testing
			random_state = np.random.RandomState(4)

			positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i]]), 
							# 0] # Similar = 0 (Suppose)
							1] # Similar = 1 (Ori)
							  for i in range(0, len(self.data), 2)]

			negative_pairs = [[i, random_state.choice(self.label_to_indices[
													   np.random.choice(
														   list(self.labels_set - set([self.labels[i]]))
													   )
												   ]),
							   # 1] # Dissimilar = 1 (Suppose)
							   0] # Dissimilar = 0 (Ori)
							  for i in range(1, len(self.data), 2)]

			self.test_pairs = positive_pairs + negative_pairs

	def __getitem__(self, index):
		if self.train:
			target = np.random.randint(0, 2)
			data1, label1 = self.data[index], self.labels[index]
			if target == 1:
			# if target == 0: # If similar (= 0)
				siamese_index = index
				while siamese_index == index:
					siamese_index = np.random.choice(self.label_to_indices[label1])
			else:
				siamese_label = np.random.choice(list(self.labels_set - set([label1])))
				siamese_index = np.random.choice(self.label_to_indices[siamese_label])
			data2 = self.data[siamese_index]
		else:
			data1 = self.data[self.test_pairs[index][0]]
			data2 = self.data[self.test_pairs[index][1]]
			target = self.test_pairs[index][2]

		#### Process Data ####
		data1 = torch.tensor(data1)
		data2 = torch.tensor(data2)
		return (data1, data2), target

	def __len__(self):
		return len(self.dataset)


class TripletAFM(Dataset):
	"""
	Train: For each sample (anchor) randomly chooses a positive AND negative samples
	Test: Creates fixed triplets for testing

	FUTURE UPDATE: 
	- Directly inherit the AFM class?
	- Create more testing pairs?
	"""

	def __init__(self, dataset):
		self.dataset = dataset
		self.train = self.dataset.train
		# self.transform = self.dataset.transform # Used torch.tensor() instead as ToTensor() takes in 2/3 dim data
		self.data = self.dataset.data

		if self.train:
			print('Structuring training data in triplet format...')
		else:
			print('Structuring testing data in triplet format...')

		# self.classes = self.dataset.classes # unique targets in INT
		# self.class_to_idx = self.dataset.class_to_idx # {'0':0, '1':1, '2':2, ...}

		self.labels = np.array([str(t) for t in self.dataset.targets]) # str version of self.dataset.targets
		self.labels_set = set(self.labels) # Unique STR labels:: {'0','1','2',...}

		# NOTE: label_to_indices =/= class_to_idx from torchvision dataset imports (label_to_indices=ALL index to specific class, class_to_idx=assigned numeric label to each class)
		# Dictionary of {'0':array([0,2,5,6]), '1':array([1,3,4,7]), '2':array([8,9,10,11]),...}
		self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

		if not self.train:
			# generate fixed triplets for testing
			random_state = np.random.RandomState(4)

			triplets = [[i,
						 random_state.choice(self.label_to_indices[self.labels[i]]),
						 random_state.choice(self.label_to_indices[
												 np.random.choice(
													 list(self.labels_set - set([self.labels[i]]))
												 )
											 ])
						 ]
						for i in range(len(self.data))]
			self.test_triplets = triplets

	def __getitem__(self, index):
		if self.train:
			data1, label1 = self.data[index], self.labels[index]
			positive_index = index
			while positive_index == index:
				positive_index = np.random.choice(self.label_to_indices[label1])
			negative_label = np.random.choice(list(self.labels_set - set([label1])))
			negative_index = np.random.choice(self.label_to_indices[negative_label])
			data2 = self.data[positive_index]
			data3 = self.data[negative_index]
		else:
			data1 = self.data[self.test_triplets[index][0]]
			data2 = self.data[self.test_triplets[index][1]]
			data3 = self.data[self.test_triplets[index][2]]

		#### Process Data ####
		data1 = torch.tensor(data1)
		data2 = torch.tensor(data2)
		data3 = torch.tensor(data3)
		return (data1, data2, data3), []

	def __len__(self):
		return len(self.dataset)


class SiameseEM(Dataset):
	"""
	Train: (Creates N labeled samples x M unlabeled samples) pairs EACH __getitem__ call
	Test: refer to eval_epoch()
	"""

	def __init__(self, labeled_dataset, unlabeled_dataset):

		self.labeled_dataset = labeled_dataset
		self.unlabeled_dataset = unlabeled_dataset

		print('Structuring training data in siamese EM format...')

		self.labeled_labels   = np.array([str(t) for t in self.labeled_dataset.targets])   # str version of self.labeled_dataset.targets
		self.unlabeled_labels = np.array([str(t) for t in self.unlabeled_dataset.targets]) # str version of self.unlabeled_dataset.targets

		self.labels_set = set(self.labeled_labels) # Unique STR labels:: {'0','1','2',...}

		# NOTE: label_to_indices =/= class_to_idx from torchvision dataset imports (label_to_indices=ALL index to specific class, class_to_idx=assigned numeric label to each class)
		# Dictionary of {'0':array([0,2,5,6]), '1':array([1,3,4,7]), '2':array([8,9,10,11]),...}
		self.labeled_label_to_indices   = {label: np.where(self.labeled_labels   == label)[0] for label in self.labels_set}
		self.unlabeled_label_to_indices = {label: np.where(self.unlabeled_labels == label)[0] for label in self.labels_set}

		# Convertions for faster/easier __getitem__()
		self.labeled_dataset.data   = [torch.tensor(d) for d in self.labeled_dataset.data]   # ndarray(ndarray) --> list(tensor)
		self.unlabeled_dataset.data = [torch.tensor(d) for d in self.unlabeled_dataset.data] # ndarray(ndarray) --> list(tensor)

		# CONVERT TO NDARRAY!!!
		self.labeled_dataset.targets   = np.array([int(d) for d in self.labeled_dataset.targets])   # list(np.int64) --> ndarray(int)
		self.unlabeled_dataset.targets = np.array([int(d) for d in self.unlabeled_dataset.targets]) # list(np.int64) --> ndarray(int)

		# ==== No 'test' in EM ====

	def __getitem__(self, index): # Returns [(labeled_1, labeled_2, ...), (unlabeled_1, unlabeled_2, ...)], [target_1, target_2, ...]
		bin_targets	= np.ones(len(self.labeled_dataset.targets),dtype=int) # First create all one targets
		bin_targets[self.labeled_dataset.targets==self.unlabeled_dataset.targets[index]] = 0 # Replace similar target/class to zero
		
		# Return 'data' in single tensor
		return (torch.stack(self.labeled_dataset.data), torch.stack([self.unlabeled_dataset.data[index]]*len(self.labeled_dataset.targets))), bin_targets, self.labeled_dataset.targets, self.unlabeled_dataset.targets[index]

	def __len__(self):
		return len(self.unlabeled_dataset)


class SiameseMNIST(Dataset):
	"""
	Train: For each sample creates randomly a positive or a negative pair
	Test: Creates fixed pairs for testing
	"""

	def __init__(self, dataset):
		self.dataset = dataset
		self.train = self.dataset.train
		self.transform = self.dataset.transform
		self.data = self.dataset.data

		if self.train:
			print('Structuring training data in siamese format...')
		else:
			print('Structuring testing data in siamese format...')

		self.labels = np.array([str(t) for t in self.dataset.targets]) # str version of self.dataset.targets
		self.labels_set = set(self.labels) # Unique STR labels:: {'0','1','2',...}

		# NOTE: label_to_indices =/= class_to_idx from torchvision dataset imports (label_to_indices=ALL index to specific class, class_to_idx=assigned numeric label to each class)
		# Dictionary of {'0':array([0,2,5,6]), '1':array([1,3,4,7]), '2':array([8,9,10,11]),...}
		self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}

		if not self.train:
			# generate fixed pairs for testing
			random_state = np.random.RandomState(4)

			positive_pairs = [[i, random_state.choice(self.label_to_indices[self.labels[i]]), 
							1]
							  for i in range(0, len(self.data), 2)]

			negative_pairs = [[i, random_state.choice(self.label_to_indices[
													   np.random.choice(
														   list(self.labels_set - set([self.labels[i]]))
													   )
												   ]),
							   0]
							  for i in range(1, len(self.data), 2)]
			self.test_pairs = positive_pairs + negative_pairs

	def __getitem__(self, index):
		if self.train:
			target = np.random.randint(0, 2)
			data1, label1 = self.data[index], self.labels[index]
			if target == 1:
				siamese_index = index
				while siamese_index == index:
					siamese_index = np.random.choice(self.label_to_indices[label1])
			else:
				siamese_label = np.random.choice(list(self.labels_set - set([label1])))
				siamese_index = np.random.choice(self.label_to_indices[siamese_label])
			data2 = self.data[siamese_index]
		else:
			data1 = self.data[self.test_pairs[index][0]]
			data2 = self.data[self.test_pairs[index][1]]
			target = self.test_pairs[index][2]

		#### Process Data ####
		data1 = Image.fromarray(data1.numpy(), mode='L')
		data2 = Image.fromarray(data2.numpy(), mode='L')
		if self.transform is not None:
			data1 = self.transform(data1)
			data2 = self.transform(data2)
		return (data1, data2), target

	def __len__(self):
		return len(self.dataset)


class TripletMNIST(Dataset):
	"""
	Train: For each sample (anchor) randomly chooses a positive and negative samples
	Test: Creates fixed triplets for testing
	"""

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset
		self.train = self.mnist_dataset.train
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_labels = self.mnist_dataset.train_labels
			self.train_data = self.mnist_dataset.train_data
			self.labels_set = set(self.train_labels.numpy())
			self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
									 for label in self.labels_set}

		else:
			self.test_labels = self.mnist_dataset.test_labels
			self.test_data = self.mnist_dataset.test_data
			# generate fixed triplets for testing
			self.labels_set = set(self.test_labels.numpy())
			self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
									 for label in self.labels_set}

			random_state = np.random.RandomState(29)

			triplets = [[i,
						 random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
						 random_state.choice(self.label_to_indices[
												 np.random.choice(
													 list(self.labels_set - set([self.test_labels[i].item()]))
												 )
											 ])
						 ]
						for i in range(len(self.test_data))]
			self.test_triplets = triplets

	def __getitem__(self, index):
		if self.train:
			img1, label1 = self.train_data[index], self.train_labels[index].item()
			positive_index = index
			while positive_index == index:
				positive_index = np.random.choice(self.label_to_indices[label1])
			negative_label = np.random.choice(list(self.labels_set - set([label1])))
			negative_index = np.random.choice(self.label_to_indices[negative_label])
			img2 = self.train_data[positive_index]
			img3 = self.train_data[negative_index]
		else:
			img1 = self.test_data[self.test_triplets[index][0]]
			img2 = self.test_data[self.test_triplets[index][1]]
			img3 = self.test_data[self.test_triplets[index][2]]

		img1 = Image.fromarray(img1.numpy(), mode='L')
		img2 = Image.fromarray(img2.numpy(), mode='L')
		img3 = Image.fromarray(img3.numpy(), mode='L')
		if self.transform is not None:
			img1 = self.transform(img1)
			img2 = self.transform(img2)
			img3 = self.transform(img3)
		return (img1, img2, img3), []

	def __len__(self):
		return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
	"""
	BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
	Returns batches of size n_classes * n_samples
	"""

	def __init__(self, labels, n_classes, n_samples):
		self.labels = labels
		self.labels_set = list(set(self.labels.numpy()))
		self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
								 for label in self.labels_set}
		for l in self.labels_set:
			np.random.shuffle(self.label_to_indices[l])
		self.used_label_indices_count = {label: 0 for label in self.labels_set}
		self.count = 0
		self.n_classes = n_classes
		self.n_samples = n_samples
		self.n_dataset = len(self.labels)
		self.batch_size = self.n_samples * self.n_classes

	def __iter__(self):
		self.count = 0
		while self.count + self.batch_size < self.n_dataset:
			classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
			indices = []
			for class_ in classes:
				indices.extend(self.label_to_indices[class_][
							   self.used_label_indices_count[class_]:self.used_label_indices_count[
																		 class_] + self.n_samples])
				self.used_label_indices_count[class_] += self.n_samples
				if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
					np.random.shuffle(self.label_to_indices[class_])
					self.used_label_indices_count[class_] = 0
			yield indices
			self.count += self.n_classes * self.n_samples

	def __len__(self):
		return self.n_dataset // self.batch_size
