import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR

from losses import ContrastiveLoss, TripletLoss

import numpy as np

from sam.sam import SAM

'''
This code contains the loader for:
- Some pre-defined list of models
- Optimizer
- Loss func
- LR scheduler
'''

#  ================ Few-Shot Parent Networks ================
class FewShotNet(nn.Module):
	def __init__(self, input_dim, out_dim):
		super().__init__()
		pass

	def forward(self, x):
		return self.enc(x)

	def get_embedding(self, x):
		return self.forward(x)


#  ================ Siamese networks ================
class ToyNet(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 100),
			nn.ReLU(),
			nn.Linear(100, 50),
			nn.ReLU(),
			nn.Linear(50, out_dim)
			)


class ToyNetL(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 150),
			nn.ReLU(),
			nn.Linear(150, 100),
			nn.ReLU(),
			nn.Linear(100, 50),
			nn.ReLU(),
			nn.Linear(50, out_dim)
			)


class ToyNetS(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 100),
			nn.ReLU(),
			nn.Linear(100, out_dim)
			)


class ToyNetS2(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.Linear(200, 100),
			nn.Linear(100, out_dim)
			)


class ToyNetS3(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, out_dim)
			)


class ToyNetXS(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.ReLU(),
			nn.Linear(200, out_dim)
			)


#  ================ Triplet networks ================
class CerberusNet(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.Linear(256, 128),
			nn.Linear(128, out_dim))

class CerberusNetL(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.Linear(256, 128),
			nn.Linear(128,64),
			nn.Linear(64, out_dim))

class CerberusNetXL(FewShotNet):
	def __init__(self, input_dim, out_dim):
		super().__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Linear(input_dim, 256),
			nn.Linear(256, 128),
			nn.Linear(128,64),
			nn.Linear(64,32),
			nn.Linear(32, out_dim))


#  ================ General networks ================

class Convo1DNet(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNet, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(32, 64, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNet2(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNet2, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 16, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(16, 32, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNet3(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNet3, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 8, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(8, 16, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNetDrp(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNetDrp, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(32, 64, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Dropout(p=0.25),
			nn.Linear(512, 256),nn.PReLU(),
			nn.Dropout(p=0.25),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Dropout(p=0.25),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNetDrp2(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNetDrp2, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(32, 64, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Dropout(p=0.25),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNetS(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNetS, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=3, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(32, 64, kernel_size=3, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

class Convo1DNetS2(FewShotNet):
	"""docstring for Convo1DNet"""
	def __init__(self, input_dim=1, out_dim=10):
		super(Convo1DNetS2, self).__init__(input_dim, out_dim)
		self.enc = nn.Sequential(
			nn.Conv1d(1, 32, kernel_size=5, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Conv1d(32, 64, kernel_size=3, stride=1),nn.PReLU(),
			nn.MaxPool1d(4, stride=4),
			nn.Flatten(),
			nn.LazyLinear(512),nn.PReLU(), # LazyLinear infers the input dim
			nn.Linear(512, 256),nn.PReLU(),
			nn.Linear(256, 128),nn.PReLU(),
			nn.Linear(128, out_dim)
			)		

	def forward(self, x):
		output = torch.unsqueeze(x, 1) # (batchsz, 412) ==> (batchsz, 1, 412)
		output = self.enc(output)
		return output

#  ================ MNIST networks ================
class MNISTNet(nn.Module):
	def __init__(self, input_dim=1, out_dim=2):
		super(MNISTNet, self).__init__(input_dim, out_dim)
		self.convnet = nn.Sequential(nn.Conv2d(input_dim, 32, 5), nn.PReLU(),
									 nn.MaxPool2d(2, stride=2),
									 nn.Conv2d(32, 64, 5), nn.PReLU(),
									 nn.MaxPool2d(2, stride=2))

		self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
								nn.PReLU(),
								nn.Linear(256, 256),
								nn.PReLU(),
								nn.Linear(256, out_dim)
								)

	def forward(self, x):
		output = self.convnet(x)
		output = output.view(output.size()[0], -1)
		output = self.fc(output)
		return output

	def get_embedding(self, x):
		return self.forward(x)


#  ================ Few Shot Networks ================
class SiameseNet(nn.Module):
	def __init__(self, embedding_net):
		super(SiameseNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x1, x2):
		output1 = self.embedding_net(x1)
		output2 = self.embedding_net(x2)
		return output1, output2

	def get_embedding(self, x):
		return self.embedding_net(x)


class TripletNet(nn.Module):
	def __init__(self, embedding_net):
		super(TripletNet, self).__init__()
		self.embedding_net = embedding_net

	def forward(self, x1, x2, x3):
		output1 = self.embedding_net(x1)
		output2 = self.embedding_net(x2)
		output3 = self.embedding_net(x3)
		return output1, output2, output3

	def get_embedding(self, x):
		return self.embedding_net(x)



#  ================ Classification Layers after Few Shots ================

class FewShotClf_1(nn.Module):
	"""docstring for FewShotClf_1"""
	def __init__(self, input_dim=10, out_dim=3):
		super(FewShotClf_1, self).__init__()
		self.fc = nn.Linear(input_dim, out_dim)

	def forward(self, x):
		x = self.fc(x)
		return x


class FewShotClf_2(nn.Module):
	"""docstring for FewShotClf_2"""
	def __init__(self, input_dim=10, out_dim=3):
		super(FewShotClf_2, self).__init__()
		self.fc1 = nn.Linear(input_dim, 5)
		self.fc2 = nn.Linear(5, out_dim)

	def forward(self, x):
		x = self.fc1(x)
		x = self.fc2(x)
		return x


model_list = {'toy':ToyNet, 'toyL':ToyNetL, 'toyS':ToyNetS, 'toyS2':ToyNetS2,'toyS3':ToyNetS3, 
			'toyXS':ToyNetXS, 'cerb':CerberusNet, 'cerbL':CerberusNetL, 'cerbXL':CerberusNetXL,
			'convo1D':Convo1DNet, 'convo1DS':Convo1DNetS, 'convo1DDrp':Convo1DNetDrp,
			'convo1DDrp2':Convo1DNetDrp2, 'convo1DS2':Convo1DNetS2, 'convo1D2':Convo1DNet2,
			'convo1D3':Convo1DNet3, 'FSclf1':FewShotClf_1,'FSclf2':FewShotClf_2,
			}

opt_list = {'sgd':SGD, 'adam': Adam}
loss_list = {'contrast':ContrastiveLoss, 'triplet':TripletLoss}


def LoadModel(args):
	print(f'Initializing {args.model} model...')

	if args.model_type == 'siamese':
		model = SiameseNet(model_list[args.model](input_dim=args.sample_dim, out_dim=args.model_out_dim)).to(args.device).double()
		loss_fn = ContrastiveLoss(margin=args.margin)
	elif args.model_type == 'triplet':
		model = TripletNet(model_list[args.model](input_dim=args.sample_dim, out_dim=args.model_out_dim)).to(args.device).double()
		loss_fn = TripletLoss(margin=args.margin)

	if args.opt=='sam':
		opt = SAM(model.parameters(), SGD, lr=args.lr, momentum=0.9)
	else:
		opt = opt_list[args.opt](model.parameters(), lr=args.lr)
	
	LR_sche = StepLR(opt, step_size=args.LR_sche_step, gamma=args.LR_sche_gamma, last_epoch=-1)
	return model, opt, LR_sche, loss_fn


def EM_LoadModel(args):
	print(f'Initializing {args.model} EM model...')

	model = SiameseNet(model_list[args.model](input_dim=args.sample_dim, out_dim=args.model_out_dim)).to(args.device).double()

	# ================ Load model weights ================
	# Loads here so the weight can be passed into opt
	checkpoint = torch.load(args.tar_path)
	model.embedding_net.load_state_dict(checkpoint['model_state_dict'])

	loss_fn = nn.BCELoss()

	if args.opt=='sam':
		opt = SAM(model.parameters(), SGD, lr=args.lr, momentum=0.9)
	else:
		opt = opt_list[args.opt](model.parameters(), lr=args.lr)
	
	LR_sche = StepLR(opt, step_size=args.LR_sche_step, gamma=args.LR_sche_gamma, last_epoch=-1)
	return model, opt, LR_sche, loss_fn


def FewShotClf_LoadModel(args):
	print(f'Initializing {args.model}(frozen) and {args.classifier_model} model...')

	# Initialize trained model's arch
	trained_model = model_list[args.model](input_dim=args.sample_dim, out_dim=args.model_out_dim).to(args.device).double()

	# ================ Load trained model weights ================
	# Loads here so the weight can be passed into opt
	checkpoint = torch.load(args.tar_path)
	trained_model.load_state_dict(checkpoint['model_state_dict'])

	# Initialize classifier model arch (temp. solution of using )
	model = model_list[args.classifier_model](input_dim=args.sample_dim, out_dim=int(args.data[0])).to(args.device).double()

	loss_fn = nn.CrossEntropyLoss()

	if args.opt=='sam':
		opt = SAM(model.parameters(), SGD, lr=args.lr, momentum=0.9)
	else:
		opt = opt_list[args.opt](model.parameters(), lr=args.lr)
	
	LR_sche = StepLR(opt, step_size=args.LR_sche_step, gamma=args.LR_sche_gamma, last_epoch=-1)
	return trained_model, model, opt, LR_sche, loss_fn
