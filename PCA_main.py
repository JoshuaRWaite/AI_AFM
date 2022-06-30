import argparse
import os
import torch
from datetime import datetime
from data import AFM
from models import FewShotClf_LoadModel
from eval import *
import matplotlib.pyplot as plt
import random # for seed
import numpy as np # for seed

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

'''
FUTURE UPDATE:
- create figs/ folder
- fix z axis label
- plot name: n_components, before or after encoding
legends, grid, etc details
'''

def plot_embeddings(data, labels, class_name, n_components, plot_num):
	color = ['r','g','b']
	fig = plt.figure()

	if n_components==2:
		ax = fig.add_subplot()
		u_labels = np.unique(labels)
		for i, label in enumerate(u_labels):
			idx = np.argwhere(np.array(labels)==label).flatten()
			temp = data[idx]
			ax.scatter(temp[:, 0], temp[:, 1], c=[color[i]]*len(temp), edgecolor="k", label=f'class {i}')
		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])
		ax.legend(['First line', 'Second line'])
	elif n_components==3:
		plt.clf()
		ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
		ax.set_position([0, 0, 0.95, 1])
		plt.cla()
		u_labels = np.unique(labels)
		for i, label in enumerate(u_labels):
			idx = np.argwhere(np.array(labels)==label).flatten()
			temp = data[idx]
			ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c=[color[i]]*len(temp), edgecolor="k", label=f'class {i}')
		ax.w_xaxis.set_ticklabels([])
		ax.w_yaxis.set_ticklabels([])
		ax.w_zaxis.set_ticklabels([])
		ax.legend(['First line', 'Second line', '3'])


	plt.title(f'{plot_num}')

	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')

	# if n_components==3: # Fix error!
		# ax.set_zlabel('PC3')

	plt.legend(loc='upper right')
	plt.savefig(f'figs/{plot_num}.jpg')
	plt.close()



def run_PCA(data, labels, class_name, n_components, plot_num):
	pca = PCA(n_components=n_components, random_state=0)
	pca.fit(data)
	print(pca.explained_variance_ratio_)
	plot_embeddings(pca.transform(data), labels, class_name, n_components, plot_num)


def run_tSNE(data, labels, class_name, n_components, plot_num):
	tsne = TSNE(n_components=n_components, learning_rate='auto', init='pca', random_state=0, n_jobs=-1)
	plot_embeddings(tsne.fit_transform(data), labels, class_name, n_components, plot_num)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='AI-AFM')
	parser.add_argument('-d','--data',type=str,default='2class', choices=['3class_matching', '2class_s_vs_r','2class_n_vs_r'], help="Choices of dataset (default='2class')")
	parser.add_argument('-opt','--opt',type=str,default='sgd', choices=['sgd', 'adam', 'sam'], help="Choices of optimizer (default='sgd')")
	parser.add_argument('-v','--verbosity', type=int, default=1, help='Verbosity. 0: No output, 1: Epoch-level output, 2: Batch-level output')
	parser.add_argument('-g','--gpu', type=int, default=0, help='GPU id')
	parser.add_argument('-ep','--epochs', type=int, default=10, help='Number of training epochs')
	parser.add_argument('-m','--model',type=str,default='toy', choices=['toy', 'toyL', 'toyS', 'toyS2', 'toyS3', 'toyXS', 'cerb', 'cerbL', 'cerbXL', 'convo1D', 'convo1DS', 'convo1DDrp', 'convo1DDrp2', 'convo1DS2', 'convo1D2', 'convo1D3'], help="Choices of models (default='toy')")
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

	parser.add_argument('--flip_data', action="store_true", default=False, help='Augmentation: Flips Input Data')
	parser.add_argument('-scale','--scale_data',type=str,choices=['none','minmax','standard'],default='minmax',help="Scale data with minmax(default), normalization or no scaling")
	parser.add_argument('-pp','--suppress_preprocess', action="store_true", default=False, help='Augmentation: Suppress Data Processing Block')
	parser.add_argument('-ns','--num_sym', type=int, default=0, help='Augmentation: Symbolize Data (default=0, no symbolize)')

	parser.add_argument('--use_cuda', action="store_true", default=True, help='Use CUDA if available') # Conflicting: Always True
	parser.add_argument('-sche_step','--LR_sche_step', type=int, default=1, help='Stepsize for LR scheduler') # For StepLR
	parser.add_argument('-sche_gamma','--LR_sche_gamma', type=float, default=1.0, help='Gamma for LR scheduler') # For StepLR

	# For freeze classifier
	parser.add_argument('-tar','--tar_name', type=str, required=True, help='Specify model weight (without .tar) to be loaded')
	parser.add_argument('-clf','--classifier_model',type=str,default='FSclf1', choices=['FSclf1','FSclf2'], help="Choices of models (default='FSclf1')")

	args = parser.parse_args()
	print(args)

	# ================ Find folder_name ================
	# Create logs folder
	# Example: /home/tsyong98/AI_AFM/165_170/logs/165/3class_hc/convo1D/sgd/06-05-2022_122025_best.tar
	folder_name = os.path.join("logs",args.exp_num,args.data,args.model,args.opt)
	file_name = os.path.join(folder_name,args.tar_name)
	tar_path = os.path.join(folder_name,args.tar_name+'.tar')
	
	if not os.path.exists('figs'):
		os.mkdir('figs')

	if not os.path.exists(folder_name):
		print(f'Could not find {folder_name} folder!')
	if not os.path.exists(tar_path):
		print(f'Could not find {tar_path} file!')

	if args.use_cuda:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	# ================ Fix random seeds ================
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

	# ================ Import Data ================
	AFM_kwargs = {
	'mode':args.data, 'winsz':args.winsz, 'scale_data':args.scale_data,
	'supp_process':args.suppress_preprocess, 'seed':args.seed}

	AFM_train = AFM(**AFM_kwargs, train=True, unlabeled_percent=args.unlabeled_percent, num_train_per_cls=args.num_train_per_cls)
	AFM_test = AFM(**AFM_kwargs, train=False, scaler=AFM_train.scaler)

	args.AFM_train = AFM_train
	args.AFM_test = AFM_test
	args.sample_dim = args.model_out_dim # Updated to the encoded data dim.
	args.n_labeled = len(AFM_train)

	# Get raw data
	loader = torch.utils.data.DataLoader(AFM_train.DF, batch_size=len(AFM_train.DF.samples)) # batch_size = entire dataset
	for (data, targets) in loader: # Single loop only since batch_size = entire dataset
		raw_train_data = torch.squeeze(data).numpy()
		raw_train_targets = list(targets.numpy())

	loader = torch.utils.data.DataLoader(AFM_test.DF, batch_size=len(AFM_test.DF.samples)) # batch_size = entire dataset
	for (data, targets) in loader: # Single loop only since batch_size = entire dataset
		raw_test_data = torch.squeeze(data).numpy()
		raw_test_targets = list(targets.numpy())


	# ---------------- Run PCA ----------------
	run_PCA(raw_train_data, raw_train_targets, None, n_components=2, plot_num=1)
	# run_PCA(raw_train_data, raw_train_targets, None, n_components=3, plot_num=2)
	run_PCA(raw_test_data, raw_test_targets, None, n_components=2, plot_num=3)
	# run_PCA(raw_test_data, raw_test_targets, None, n_components=3, plot_num=4)

	# ---------------- Run tSNE ----------------
	run_tSNE(raw_train_data, raw_train_targets, None, n_components=2, plot_num=5)
	# run_tSNE(raw_train_data, raw_train_targets, None, n_components=3, plot_num=6)
	run_tSNE(raw_test_data, raw_test_targets, None, n_components=2, plot_num=7)
	# run_tSNE(raw_test_data, raw_test_targets, None, n_components=3, plot_num=8)

	# ================ Stacked raw data ================
	raw_data = np.vstack((raw_train_data,raw_test_data))
	raw_targets = raw_train_targets + raw_test_targets

	# ---------------- Run PCA ----------------
	run_PCA(raw_data, raw_targets, None, n_components=2, plot_num=17)

	# ---------------- Run tSNE ----------------
	run_tSNE(raw_data, raw_targets, None, n_components=2, plot_num=18)


	# ================ Load trained model (encoder), classifier model, optimizer, scheduler, loss_fn ================
	# Also loads model weight from tar_path
	args.tar_path = tar_path
	encoder, model, optimizer, scheduler, loss_fn = FewShotClf_LoadModel(args)

	# Encode input data (N, datalen) --> (N, EncodedLen)
	encoder.eval()
	args.AFM_train.data = encoder(torch.tensor(args.AFM_train.data).to(args.device)).to('cpu').detach().numpy()
	args.AFM_test.data  = encoder(torch.tensor(args.AFM_test.data).to(args.device)).to('cpu').detach().numpy()
	del encoder

	# ---------------- Run PCA ----------------
	run_PCA(args.AFM_train.data, args.AFM_train.targets, None, n_components=2, plot_num=9)
	# run_PCA(args.AFM_train.data, args.AFM_train.targets, None, n_components=3, plot_num=10)
	run_PCA(args.AFM_test.data, args.AFM_test.targets, None, n_components=2, plot_num=11)
	# run_PCA(args.AFM_test.data, args.AFM_test.targets, None, n_components=3, plot_num=12)
	
	# ---------------- Run tSNE ----------------
	run_tSNE(args.AFM_train.data, args.AFM_train.targets, None, n_components=2, plot_num=13)
	# run_tSNE(args.AFM_train.data, args.AFM_train.targets, None, n_components=3, plot_num=14)
	run_tSNE(args.AFM_test.data, args.AFM_test.targets, None, n_components=2, plot_num=15)
	# run_tSNE(args.AFM_test.data, args.AFM_test.targets, None, n_components=3, plot_num=16)


	# ================ Stacked few shot embedded data ================
	AFM_data = np.vstack((args.AFM_train.data,args.AFM_test.data))
	AFM_targets = args.AFM_train.targets + args.AFM_test.targets

	# ---------------- Run PCA ----------------
	run_PCA(AFM_data, AFM_targets, None, n_components=2, plot_num=19)

	# ---------------- Run tSNE ----------------
	run_tSNE(AFM_data, AFM_targets, None, n_components=2, plot_num=20)
