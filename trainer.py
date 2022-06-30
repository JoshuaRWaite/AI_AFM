import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from metrics import eval_
from eval import top_k_majority

'''
This code contains:
- The training and validation loop func
- Loss logging and model saving
'''

class logger(object):
	"""docstring for logger"""
	def __init__(self):
		super(logger, self).__init__()
		self.log = {}
		self.log['train_loss'] = []
		self.log['val_loss'] = []
		# self.log['train_acc'] = [] # Inference is not performed during training for few shot
		# self.log['val_acc'] = [] # Inference is not performed during training for few shot
		self.folder_name = None
		self.fname = None

	def save_progress(self, ep, model, opt, type_=''):
		with open(self.fname+'.json', 'w') as fp1:
			json.dump(self.log, fp1, indent=4, sort_keys=False)

		checkpoint_path = os.path.join(self.fname+type_+'.tar')

		torch.save({'epoch': ep,
					'model_state_dict': model.embedding_net.state_dict(),
					'optimizer_state_dict': opt.state_dict()}, checkpoint_path)

def save_loss_plot(args, log):
	plt.figure(1)
	plt.plot(log.log['train_loss'], label='Training Loss', color='b')
	plt.plot(log.log['val_loss'], label='Validation Loss', linestyle='--', color='b')
	plt.title(str(args.data)+', '+str(args.model)+', '+str(args.opt))
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(loc='upper right')
	plt.savefig(log.fname+'_loss.jpg')
	plt.close()


def fit(args, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, log, metrics=[]):
	print('Starting training...')

	# Accuracy placeholders
	best_acc, current_acc, best_ep = 0.0, 0.0, 0.0 # To save the best model
	best_resu = None

	for epoch in range(1, n_epochs+1):

		# Training stage
		train_loss, metrics = train_epoch(args, train_loader, model, loss_fn, optimizer, cuda, args.verbosity, metrics)
		log.log["train_loss"].append(train_loss)
		# log.log["train_acc"].append(metrics[0].value()) # Refer to __init__ log reasoning

		if args.verbosity > 0:
			message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch, n_epochs, train_loss)
			for metric in metrics:
				message += '\t{}: {}'.format(metric.name(), metric.value())

		# Validation stage
		val_loss, metrics = val_epoch(val_loader, model, loss_fn, cuda, metrics)
		val_loss /= len(val_loader)
		log.log["val_loss"].append(val_loss)
		# log.log["val_acc"].append(metrics[0].value()) # Refer to __init__ log reasoning

		# Evaluation stage
		resu = eval_epoch(args, model)
		current_acc = resu['clf_rep']['accuracy']

		if args.verbosity > 0:
			message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch, n_epochs, val_loss)
			for metric in metrics:
				message += '\t{}: {}'.format(metric.name(), metric.value())		

			print(message)

		if current_acc > best_acc: # If equal: find least std dev in class acc (not implemented)
			best_acc = current_acc # Update best acc with current acc
			best_resu = resu # Update best resu with resu
			best_ep = epoch
			print(f'Saving best model with {best_acc*100:.2f} % acc...')
			log.save_progress(epoch, model, optimizer, type_='_best')

		elif (((epoch-1) % log_interval == 0) or (epoch==n_epochs)): # Save model at every log_interval AND very last epoch
			print(f"Saving progress at epoch {epoch}...")
			log.save_progress(epoch, model, optimizer)

			if epoch==n_epochs: # Print the best results for easier comparison with final model eval
				print('='*70)
				print(f'Best Epoch: {best_ep}')
				print(f"Best Overall Acc: {best_resu['clf_rep']['accuracy']*100:.2f}")
				print("Class Accuracies:",best_resu['class_acc'])
				print('Confusion Matrix:')
				print(best_resu['cm'])
				print('='*70)

		scheduler.step()


def train_epoch(args, train_loader, model, loss_fn, optimizer, cuda, verbosity, metrics):
	for metric in metrics:
		metric.reset()

	model.train()
	losses = []
	total_loss = 0

	for batch_idx, (data, target) in enumerate(train_loader):
		target = target if len(target) > 0 else None
		if not type(data) in (tuple, list):
			data = (data,)
		if cuda:
			data = tuple(d.cuda() for d in data)
			# data = tuple(d.to(args.device) for d in data)
			if target is not None:
				target = target.cuda()
				# target = target.to(args.device)

		optimizer.zero_grad()
		outputs = model(*data) # (model_type, batchsz, model_out_dim)

		if type(outputs) not in (tuple, list):
			outputs = (outputs,)

		loss_inputs = outputs
		if target is not None:
			target = (target,)
			loss_inputs += target

		loss_outputs = loss_fn(*loss_inputs)
		loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

		losses.append(loss.item())
		total_loss += loss.item()

		loss.backward()

		if args.opt == 'sam':
			optimizer.first_step(zero_grad=True)
			
			outputs = model(*data)
			if type(outputs) not in (tuple, list):
				outputs = (outputs,)
			loss_inputs = outputs
			if target is not None:
				loss_inputs += target
			loss_outputs2 = loss_fn(*loss_inputs)
			loss = loss_outputs2[0] if type(loss_outputs2) in (tuple, list) else loss_outputs2
			loss.backward()
			optimizer.second_step(zero_grad=True)
		else:
			optimizer.step()

		for metric in metrics:
			metric(outputs, target, loss_outputs)

		if verbosity > 1:
			message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				batch_idx * len(data[0]), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), np.mean(losses))
			for metric in metrics:
				message += '\t{}: {}'.format(metric.name(), metric.value())

			print(message)
			losses = []

	total_loss /= (batch_idx + 1)
	return total_loss, metrics


def val_epoch(val_loader, model, loss_fn, cuda, metrics):
	with torch.no_grad():
		for metric in metrics:
			metric.reset()
		model.eval()
		val_loss = 0
		for batch_idx, (data, target) in enumerate(val_loader):
			target = target if len(target) > 0 else None
			if not type(data) in (tuple, list):
				data = (data,)
			if cuda:
				data = tuple(d.cuda() for d in data)
				if target is not None:
					target = target.cuda()

			outputs = model(*data)

			if type(outputs) not in (tuple, list):
				outputs = (outputs,)
			loss_inputs = outputs
			if target is not None:
				target = (target,)
				loss_inputs += target

			loss_outputs = loss_fn(*loss_inputs)
			loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
			val_loss += loss.item()

			for metric in metrics:
				metric(outputs, target, loss_outputs)

	return val_loss, metrics

def eval_epoch(args, model):
	'''
	---- Pseudo ----
	- Get Embeddings
	- Get dist
	- Top_k
	- Temp. store pred
	- Compute and return acc
	'''

	# Defining target_names ==> Can be moved to data.py OR rename the folders directly
	if args.data.startswith('4class'):
		target_names = ['no_rupture', 'single_rupture', 'double_rupture', 'multi_rupture']
	elif args.data.startswith('3class'):
		target_names = ['no_and_multi_rupture', 'single_rupture', 'double_rupture']
	elif args.data.startswith('2class'):
		target_names = ['single_rupture', 'double_rupture']

	model.eval()

	# Get Embeddings
	test_emb = [model.get_embedding(torch.tensor(np.array([x])).to(args.device)) for x in args.AFM_test.data]
	supp_emb = [model.get_embedding(torch.tensor(np.array([y])).to(args.device)) for y in args.AFM_train.data]

	# Get Distances
	preds = []
	for te in test_emb:
		dists = []
		for se in supp_emb:
			dist = (te - se).pow(2).sum(1)
			dists.append(dist.detach().cpu().numpy())

		pred = top_k_majority(dists, args.AFM_train.targets, k=args.top_k)
		preds.append(pred)

	resu_ = eval_(args.AFM_test.targets, preds, target_names, print_resu=False, save_resu=False)

	return resu_