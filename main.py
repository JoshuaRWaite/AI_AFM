import argparse
import os
import torch
from datetime import datetime
from data import AFM
from datasets import SiameseAFM, TripletAFM
from models import LoadModel
from trainer import logger
from trainer import fit
from trainer import save_loss_plot
from eval import *
import matplotlib.pyplot as plt
import random # for seed
import numpy as np # for seed

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='AI-AFM')
	parser.add_argument('-d','--data',type=str,default='2class', choices=['3class_matching', '2class_s_vs_r', '2class_n_vs_r'], help="Choices of dataset (default='2class')")
	parser.add_argument('-opt','--opt',type=str,default='sgd', choices=['sgd', 'adam', 'sam'], help="Choices of optimizer (default='sgd')")
	parser.add_argument('-v','--verbosity', type=int, default=1, help='Verbosity. 0: No output, 1: Epoch-level output, 2: Batch-level output')
	parser.add_argument('-g','--gpu', type=int, default=0, help='GPU id')
	parser.add_argument('-ep','--epochs', type=int, default=10, help='Number of training epochs')
	parser.add_argument('-m','--model',type=str,default='toy', choices=['toy', 'toyL', 'toyS', 'toyS2', 'toyS3', 'toyXS', 'cerb', 'cerbL', 'cerbXL', 'convo1D', 'convo1DS', 'convo1DDrp', 'convo1DDrp2', 'convo1DS2', 'convo1D2', 'convo1D3'], help="Choices of models (default='toy')")
	parser.add_argument('-mt','--model_type',type=str,default='siamese', choices=['siamese', 'triplet'], help="Model type (default='siamese')")
	parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
	parser.add_argument('-bs','--batch_size', type=int, default=16, help='Training batch size (default = 16)')
	parser.add_argument('-mod','--model_out_dim', type=int, default=10, help='Model output dim (default = 10)')
	parser.add_argument('-log','--log_interval', type=int, default=10, help='Saving model at every specified interval (default = 10)')
	parser.add_argument('-winsz','--winsz', type=int, default=5, help='AFM data processing window size (default = 5)')
	parser.add_argument('-k','--top_k', type=int, default=1, help='Top-k majority for classification (default=1)')
	parser.add_argument('-s','--seed', type=int, default=0, help='Random seed (default=0)')
	parser.add_argument('-up','--unlabeled_percent', type=float, default=0.0, help='Percent of unlabeled data with range of [0.0,1.0) (default=0.0)')
	parser.add_argument('-mgn','--margin', type=float, default=1.0, help='Loss function margin (default=1.0)')
	parser.add_argument('-exp','--exp_num',type=str,default='0', help="Define experiment number")
	parser.add_argument('-nt','--num_train_per_cls',type=int,default=-1, help="Number of training samples per class (default=-1, all data)")
	parser.add_argument('-ds','--data_seed', type=int, default=0, help='Random seed for data sampling (default=0)')

	parser.add_argument('-scale','--scale_data',type=str,choices=['none','minmax','standard'],default='minmax',help="Scale data with minmax(default), normalization or no scaling")
	parser.add_argument('-pp','--suppress_preprocess', action="store_true", default=False, help='Augmentation: Suppress Data Processing Block')

	parser.add_argument('--use_cuda', action="store_true", default=True, help='Use CUDA if available') # Conflicting: Always True
	parser.add_argument('-sche_step','--LR_sche_step', type=int, default=1, help='Stepsize for LR scheduler') # For StepLR
	parser.add_argument('-sche_gamma','--LR_sche_gamma', type=float, default=1.0, help='Gamma for LR scheduler') # For StepLR

	parser.add_argument('-tar','--tar_name', type=str, default="", help='Specify model weight (without .tar) to be loaded')

	args = parser.parse_args()
	print(args)

	# Create logs folder
	folder_name = os.path.join("logs",args.exp_num,args.data,args.model,args.opt)
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)

	# Sample file_name = "logs/data/model/opt/<date_time>.json"
	file_name = os.path.join(folder_name,datetime.now().strftime("%d-%m-%Y_%H%M%S"))

	if args.use_cuda:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	# Fix random seeds
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic=True
	torch.backends.cudnn.benchmark=False
	if args.use_cuda:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	# Select devices
	args.device = torch.device("cuda" if args.use_cuda else "cpu")

	# Import Data
	AFM_kwargs = {
	'mode':args.data, 'winsz':args.winsz, 'scale_data':args.scale_data,
	'supp_process':args.suppress_preprocess, 'seed':args.data_seed}

	AFM_train = AFM(**AFM_kwargs, train=True, unlabeled_percent=args.unlabeled_percent, num_train_per_cls=args.num_train_per_cls)
	AFM_test = AFM(**AFM_kwargs, train=False, scaler=AFM_train.scaler)
	args.AFM_train = AFM_train
	args.AFM_test = AFM_test
	args.sample_dim = AFM_train.sample_dim

	# Structure into few-shot format:
	if args.model_type == 'triplet':
		fewshot_AFM_train = TripletAFM(AFM_train)
		fewshot_AFM_test  = TripletAFM(AFM_test)
	elif args.model_type == 'siamese':
		fewshot_AFM_train = SiameseAFM(AFM_train)
		fewshot_AFM_test  = SiameseAFM(AFM_test)

	# Set up data loaders
	train_loader = torch.utils.data.DataLoader(fewshot_AFM_train, batch_size=args.batch_size,num_workers=1, shuffle=True)#, sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(fewshot_AFM_test, batch_size=args.batch_size, num_workers=2)#, sampler=test_sampler)

	model, optimizer, scheduler, loss_fn = LoadModel(args)

	# Dry run to initialize LazyLinear() layer in Convo1D model
	if args.model in ['convo1D', 'convo1DS', 'convo1DDrp', 'convo1DDrp2', 'convo1DS2', 'convo1D2', 'convo1D3']:
		print('Performing dry run...')
		num_x_needed = 2 if args.model_type == 'siamese' else 3
		sample_data = [torch.rand(args.batch_size, args.sample_dim).to(args.device).double() for _ in range(num_x_needed)]
		_ = model(*sample_data)

	# Xavier weight initialization (aka Glorot initialization)
	print("Performing xavier initialization...")
	for p in model.parameters():
		if p.dim() > 1:
			torch.nn.init.xavier_uniform_(p)

	log = logger()
	log.folder_name = folder_name
	log.fname = file_name

	log.log['data'] = args.data
	log.log['model'] = args.model
	log.log['opt'] = args.opt
	log.log['batch_size'] = args.batch_size
	log.log['num_epochs'] = args.epochs
	log.log['lr'] = args.lr
	log.log['lr step'] = args.LR_sche_step
	log.log['lr gamma'] = args.LR_sche_gamma

	fit(args, train_loader, val_loader, model, loss_fn, optimizer, scheduler, args.epochs, args.use_cuda, args.log_interval, log, [])
	print('Training complete!')

	# Plot Train and Val Loss
	print("Plotting losses...")
	save_loss_plot(args, log)

	# Get Testing Accuracies
	print("Evaluating Performance...")
	eval_test(args) # Need to include args for scaler 
