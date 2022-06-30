import torch
from torchvision.datasets import DatasetFolder
import os
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

'''
This code loads and format data into similar format as torchvision's dataset import
'''

class AFM(torch.utils.data.Dataset):
	"""docstring for AFM dataset"""
	def __init__(self, mode='2class', train=True, transform=None, winsz=5, unlabeled_percent=0.0, seed=0,
		flip=False, num_train_per_cls=-1, scale_data='none', supp_process=False, scaler=None):
		self.mode = mode
		self.train = train
		# self.transform = transform
		self.winsz = winsz
		self.scaler = scaler
		self.num_train_per_cls = num_train_per_cls
		self.seed = seed

		if self.train:
			print(f'Loading {self.mode} training data...')
			self.root = os.path.join("data",self.mode,"train")
		else:
			print(f'Loading {self.mode} testing data...')
			self.root = os.path.join("data",self.mode,"test")

		self.DF = DatasetFolder(self.root,self.txt_loader,extensions=('txt'))#,transform=self.transform)

		'''
		num_train_per_cls = needed_unlabeled_sample = not possible: Only when 0.0%
		num_train_per_cls > needed_unlabeled_sample = trim
		num_train_per_cls < needed_unlabeled_sample = not possible: Assert!
		'''

		if self.train:
			'''
			Future Update: Can be more efficient w/ 1 trim if compute num_train_per_cls first according to unlabeled_percent
			'''
			# Trim TRAIN data to defined num_train_per_cls
			if self.num_train_per_cls != -1: # Will return same split of data if num_train_per_cls is same!
				self.trim_data(self.num_train_per_cls)
			else:
				self.num_train_per_cls = int(len(self.DF.targets)/len(self.DF.classes)) # updates -1 to true num_train_per_cls

			# If wanted to add unlabeled training data:
			if unlabeled_percent > 0:
				# Further trim TRAIN data to acommodate unlabeled_percent
				num_labeled_needed_per_cls = int(self.num_train_per_cls*(1-unlabeled_percent)) # Num labeled data if w/ unlabeled data
				self.trim_data(num_labeled_needed_per_cls)
				# self.replace_w_unlabeled_data(unlabeled_percent, num_labeled_needed_per_cls) # Fulfils unlabeled % while keeping constant total num train sample (By removing labeled data with unlabeled)

		self.classes = self.DF.classes
		self.class_to_idx = self.DF.class_to_idx

		loader = torch.utils.data.DataLoader(self.DF, batch_size=len(self.DF.samples)) # batch_size = entire dataset

		for (data, targets) in loader: # Single loop only since batch_size = entire dataset
			self.data = data
			self.data = torch.squeeze(self.data).numpy() # keep as np.ndarray, following cifar10 data import standards AND for data processing later
			self.targets = list(targets.numpy()) # keep as list, following cifar10 data import standards

		if self.train and unlabeled_percent > 0: # Update: Should be moved to above as commented
			# self.add_unlabeled_data(unlabeled_percent) # Fulfils unlabeled % while keeping as much labeled data as possible (total num train sample varies with %)
			self.replace_w_unlabeled_data(unlabeled_percent, num_labeled_needed_per_cls) # Fulfils unlabeled % while keeping constant total num train sample (By removing labeled data with unlabeled)

		if not supp_process:
			self.process_data()

		self.scale_data(scale_data)

		self.sample_dim = np.shape(self.data[0])[0] # Use to set model input dim

		'''
		Other attr from official cifar10 dataset import:
		'base_folder','extra_repr','filename','functions','meta','register_datapipe_as_function',
		'register_function','target_transform','test_list','tgz_md5','train_list','transforms',
		 '''

	def txt_loader(self, path: str) -> np.ndarray:
		return np.expand_dims(np.loadtxt(path,delimiter=','), axis=0) # (1, 412)

	def trim_data(self, sample_per_cls):
		# Trim TRAIN data to sample_per_cls
		sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_per_cls*len(self.DF.classes), random_state=self.seed)
		for train_index, _ in sss.split(self.DF.samples, self.DF.targets):
			self.DF.samples = [self.DF.samples[i] for i in train_index]
			self.DF.targets = [self.DF.targets[i] for i in train_index]

	def process_data(self):
		print("Applying preprocessing block...")
		'''
		Process in sequencial order:
		- Rolling mean window
		- Flip data order
		- Scale data to [0,1]
		'''
		self.data = np.transpose(self.data)

		# Old preprocessing
		# self.data = pd.DataFrame(self.data).rolling(window=self.winsz).mean().dropna()

		# New preprocessing block
		self.data = np.diff(self.data,axis=0)
		self.data = pd.DataFrame(self.data).rolling(window=self.winsz).mean()
		self.data = pd.DataFrame(self.data).rolling(window=self.winsz).mean()
		self.data = pd.DataFrame(self.data).rolling(window=self.winsz).mean().dropna().values
		self.data = np.transpose(self.data)

	def scale_data(self, mode):
		print(f"Scaling data in {mode} mode...")
		if mode=='minmax':
			self.data = np.transpose(self.data)
			self.data = np.flip(self.data,axis=0)
			max_ = np.max(self.data,axis=0)
			min_ = np.min(self.data,axis=0)
			self.data = (self.data-min_)/(max_-min_)
			self.data = np.transpose(self.data)
		elif mode=='standard':
			if self.scaler is None:
				self.scaler = StandardScaler().fit(self.data)
			self.data = self.scaler.transform(self.data)
		elif mode=='none':
			pass

	def add_unlabeled_data(self, unlabeled_percent):
		'''
		Get all unlabeled data path
		compute class data len
		compute num required unlabeled sample
		assert required num sample with min sample avail
		if unlabel % too high, use all unlabeled data and reduce labeled data to meet the req percentage
		extract data according to req len
		add unlabeled data to self.data

		'''
		print(f"Adding {unlabeled_percent*100}% mismatch labeled data...")
		class_folder = list(np.arange(4).astype(str))

		# Get all data paths
		all_paths = []
		num_unlabeled_sample = []
		for c in class_folder:
			all_paths.append(glob.glob(os.path.join("data","RL Train","Mismatching Labels",c,"*.txt")))
			num_unlabeled_sample.append(len(all_paths[-1]))

		# Find class with least samples
		min_sample = np.min(num_unlabeled_sample)
		min_sample_idx = np.argmin(num_unlabeled_sample)

		# Calculate num unlabeled sample needed for percentage
		c1, c2 = np.unique(self.DF.targets,return_counts=True)
		assert len(np.unique(c2))==1, f"Classes doesn't have equal num. samples! Classes:{c1}, Counts:{c2}"
		labeled_sample_per_cls = c2[0] # Usiang first class num samples as reference, assuming all class have equal num samples
		needed_sample = int((unlabeled_percent*labeled_sample_per_cls)/(1-unlabeled_percent)) # Check equation

		num_reduce = 0
		if needed_sample > min_sample: # If more unlabeled sample needed than available
			needed_sample = min_sample # set it to however many available, then
			# Compute matching label data needed to remove instead to met that unlabeled_percent
			num_reduce = labeled_sample_per_cls - int(((1-unlabeled_percent)*needed_sample)/unlabeled_percent)

			print("================ Warning ================")
			print(f"Class {min_sample_idx} only have {min_sample} unlabeled samples.")
			print(f"Removing {num_reduce} matching labels data per class to fulfil the {unlabeled_percent} unlabeled_percent!")
			print("=========================================")

		# Trim num unlabeled data
		all_paths = [p[:needed_sample] for p in all_paths]

		# Placeholders
		unlabeled_data = []
		unlabeled_label = []
		running_new_label = 0 # Running new label for different num class cases

		if self.mode.startswith("3class"):
			# Combine class 0 and 3 (~ half and half)
			paths = all_paths[0][:int(needed_sample/2)] + all_paths[3][:int(needed_sample-int(needed_sample/2))]
			# Load class 0 and 3
			for p in paths:
				unlabeled_data.append(np.loadtxt(p,delimiter=',')) # check dimension
				unlabeled_label.append(running_new_label) # Can be more efficient
			running_new_label += 1		

		elif self.mode.startswith("4class"):
			for c in class_folder:
				paths = all_paths[int(c)] # already trimmed with needed_sample
				for p in paths:
					unlabeled_data.append(np.loadtxt(p,delimiter=','))
					unlabeled_label.append(running_new_label)
				running_new_label += 1

		if (self.mode.startswith("3class")) or (self.mode.startswith("2class")):
			# Load class 1 and 2
			class_folder = [1, 2]
			for c in class_folder:
				paths = all_paths[c] # already trimmed with needed_sample
				for p in paths:
					unlabeled_data.append(np.loadtxt(p,delimiter=','))
					unlabeled_label.append(running_new_label)
				running_new_label += 1

		if num_reduce > 0:
			# Remove labeled data per class to fulfil the unlabeled_percent
			new_labeled_data = []
			new_labeled_targets = []

			classes = np.unique(self.targets)
			for c in classes:
				idx = np.argwhere(self.targets==c).flatten()
				new_labeled_data.append(self.data[idx[:-num_reduce]]) # Trim labeled data
				new_labeled_targets.append(np.array(self.targets)[idx[:-num_reduce]]) # Trim labeled targets
				
			self.data = np.array(new_labeled_data).reshape(-1, np.shape(new_labeled_data)[-1]) # Stack them back
			self.targets = list(np.array(new_labeled_targets).flatten())

		unlabeled_data = np.array(unlabeled_data)
		self.data = np.concatenate((self.data, unlabeled_data))
		self.targets += unlabeled_label

		# Shuffle data? Already randomed in dataset.py

	def replace_w_unlabeled_data(self, unlabeled_percent, num_labeled_needed_per_cls):
		'''
		### Add unlabeled data while maintaining same num training samples ###
		Get all unlabeled data path
		compute class data len
		compute num required unlabeled sample
		assert required num sample with min sample avail
		if unlabel % too high, use all unlabeled data and reduce labeled data to meet the req percentage
		extract data according to req len
		add unlabeled data to self.data

		'''
		print(f"Replacing training data w/ {unlabeled_percent*100}% mismatch labeled data...")
		class_folder = list(np.arange(4).astype(str))

		# Get all data paths
		all_paths = []
		num_unlabeled_sample = []
		for c in class_folder:
			all_paths.append(glob.glob(os.path.join("data","RL Train","Mismatching Labels",c,"*.txt")))
			num_unlabeled_sample.append(len(all_paths[-1]))

		# Find class with least samples
		min_sample = np.min(num_unlabeled_sample)
		min_sample_idx = np.argmin(num_unlabeled_sample)

		# Calculate num unlabeled sample needed for percentage
		c1, c2 = np.unique(self.DF.targets,return_counts=True)
		assert len(np.unique(c2))==1, f"Classes doesn't have equal num. samples! Classes:{c1}, Counts:{c2}"
		labeled_sample_per_cls = c2[0] # Using first class num samples as reference, assuming all class have equal num samples

		# ==== Different from add_unlabeled_data from here on ====
		needed_sample = int(self.num_train_per_cls - num_labeled_needed_per_cls) # Needed unlabeled samples

		if needed_sample>min_sample: # Needed unlabeled samples > unlabeled sample class with least num samples
			print("================ Warning ================")
			print(f"Class {min_sample_idx} only have {min_sample} unlabeled samples.")
			print(f"Please select a smaller unlabeled_percent value!")
			print("=========================================")
			assert needed_sample<=min_sample,'Insufficient unlabeled data!'
		# ================================================================
		else: # Insert unlabeled data
			# Trim num unlabeled data
			all_paths = [p[:needed_sample] for p in all_paths]

			# Placeholders
			unlabeled_data = []
			unlabeled_label = []
			running_new_label = 0 # Running new label for different num class cases

			if self.mode.startswith("3class"):
				# Combine class 0 and 3 (~ half and half)
				paths = all_paths[0][:int(needed_sample/2)] + all_paths[3][:int(needed_sample-int(needed_sample/2))]
				# Load class 0 and 3
				for p in paths:
					unlabeled_data.append(np.loadtxt(p,delimiter=',')) # check dimension
					unlabeled_label.append(running_new_label) # Can be more efficient
				running_new_label += 1		

			elif self.mode.startswith("4class"):
				for c in class_folder:
					paths = all_paths[int(c)] # already trimmed with needed_sample
					for p in paths:
						unlabeled_data.append(np.loadtxt(p,delimiter=','))
						unlabeled_label.append(running_new_label)
					running_new_label += 1

			if (self.mode.startswith("3class")) or (self.mode.startswith("2class")):
				# Load class 1 and 2
				class_folder = [1, 2]
				for c in class_folder:
					paths = all_paths[c] # already trimmed with needed_sample
					for p in paths:
						unlabeled_data.append(np.loadtxt(p,delimiter=','))
						unlabeled_label.append(running_new_label)
					running_new_label += 1

			unlabeled_data = np.array(unlabeled_data)
			self.data = np.concatenate((self.data, unlabeled_data))
			self.targets += unlabeled_label

			# Shuffle data? Already randomed in dataset.py

	def __len__(self):
		return len(self.DF.samples)


	def __getitem__(self, index):
		data   = torch.tensor(self.data[index])
		target = torch.tensor(self.targets[index])
		return data, target

'''
==== Optional components ====
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

n_training_samples = 1000
n_val_samples = 100
n_test_samples = 100

train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

# import torchvision.transforms as transforms
# transform = transforms.Compose([transforms.ToTensor()]) # Used torch.tensor() instead as ToTensor() takes in 2/3 dim data
'''
