import argparse
import os 
import glob

import torch
from models import *
from data import *
from datasets import *

import numpy as np
# from STPN.util import eval
from metrics import eval_

'''
This code loads the specified saved model, 
evaluate and save its performance on test data.
'''

def top_k_majority(dist, label, k=1):
	assert (len(label)==len(dist)) and (k>=1) and (len(label)>=k)
	dist, label = zip(*sorted(zip(dist, label))) #, key=lambda x:x[0]
	label = label[:k]
	u_label, u_count = np.unique(label,return_counts=True)
	idx = np.argmax(u_count)
	pred_label = u_label[idx]
	return pred_label

def eval_test(args):
	log_folder_name = os.path.join("logs",args.exp_num,args.data,args.model,args.opt,'*.tar')
	tar_files = glob.glob(log_folder_name)

	for tf in tar_files: # to loop over multiple tar files (if there is any)
		save_fname = os.path.splitext(os.path.basename(tf))[0]

	# Import Data
	AFM_kwargs = {
	'mode':args.data, 'winsz':args.winsz, 'scale_data':args.scale_data,
	'supp_process':args.suppress_preprocess}

	AFM_train = AFM(**AFM_kwargs, train=True, unlabeled_percent=args.unlabeled_percent)
	AFM_val = AFM(**AFM_kwargs, train=False, scaler=AFM_train.scaler)
	args.sample_dim = AFM_train.sample_dim


	# Select devices
	args.device = torch.device("cuda" if args.use_cuda else "cpu")

	# Load Model
	if args.tar_name:
		folder_name = os.path.join("logs",args.exp_num,args.data,args.model,args.opt)
		file_name = os.path.join(folder_name,args.tar_name)
		checkpoint_path = os.path.join(folder_name,args.tar_name+'.tar')

		print(checkpoint_path)

		if not os.path.exists(folder_name):
			print(f'Could not find {folder_name} folder!')
		if not os.path.exists(checkpoint_path):
			print(f'Could not find {checkpoint_path} file!')
	else:
		checkpoint_folder_path = os.path.join("logs",args.exp_num,args.data,args.model,args.opt)
		list_of_files = glob.glob(checkpoint_folder_path+'/*.tar')
		checkpoint_path = max(list_of_files, key=os.path.getctime) #get most recent checkpoint
		print(checkpoint_path)

	model = model_list[args.model](input_dim=args.sample_dim, out_dim=args.model_out_dim).to(args.device).double()
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])

	model.eval()

	save_path = os.path.join('resu',args.exp_num,args.data,args.model,args.opt,save_fname)

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# Load Data
	test_data = AFM_val.data
	support_data = AFM_train.data

	test_labels = AFM_val.targets
	support_labels = AFM_train.targets

	# Params
	k = args.top_k

	if args.data.startswith('4class'):
		target_names = ['no_rupture', 'single_rupture', 'double_rupture', 'multi_rupture']
	elif args.data.startswith('3class'):
		target_names = ['no_and_multi_rupture', 'single_rupture', 'double_rupture']
	elif args.data.startswith('2class'):
		target_names = ['single_rupture', 'double_rupture']

	# Get Embeddings
	test_emb = [model.get_embedding(torch.tensor(np.array([x])).to(args.device)) for x in test_data]
	supp_emb = [model.get_embedding(torch.tensor(np.array([y])).to(args.device)) for y in support_data]

	# Get Distances
	preds = []
	test_id = 0
	for te in test_emb:
		dists = []
		for se in supp_emb:
			dist = (te - se).pow(2).sum(1)
			dists.append(dist.detach().cpu().numpy())

		np.save(os.path.join(save_path,f'{test_id}_dists.npy'), dists) # save distances with each support data

		pred = top_k_majority(dists, support_labels, k=k)
		preds.append(pred)
		test_id += 1

	np.save(os.path.join(save_path,f'top{k}_preds.npy'), preds) # save predictions for each test data

	# print(test_labels)
	# print(np.shape(test_labels))
	# print(preds)
	# print(np.shape(preds))

	resu = eval_(test_labels, preds, target_names, save_resu=True, save_path=save_path)

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

	eval_test(args)
