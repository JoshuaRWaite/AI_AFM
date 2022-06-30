import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

'''
This code contains the metrics and func for training and evaluation process
'''

class Metric:
	def __init__(self):
		pass

	def __call__(self, outputs, target, loss):
		raise NotImplementedError

	def reset(self):
		raise NotImplementedError

	def value(self):
		raise NotImplementedError

	def name(self):
		raise NotImplementedError


def __class_acc(cm, target_names):
	assert len(cm)==len(target_names)

	TP = np.diagonal(cm) # True positive
	samples_per_class = np.sum(cm,axis=1) # Total sample in each class
	acc = [tp/s for tp,s in zip(TP,samples_per_class)]  

	class_acc = {}
	for i in range(len(TP)):
		class_acc[target_names[i]] = acc[i]

	return class_acc


def eval_(truth, pred, target_names, print_resu=True, save_resu=False, save_path=os.getcwd()):
	'''
	Returns dictionary containing:
	- class accuracy
	- confusion matrix
	- classification_report(clf_rep)
		- precision
		- recall
		- F1-score
		- overall accuracy
		- macro average
		- weighted average
	
	if binary target class, also returns:
	- tn_fp_fn_tp

	'''
	truth = np.array(truth).astype(int)
	pred = np.array(pred).astype(int)
	target_names = [str(i) for i in target_names]

	cm = confusion_matrix(truth,pred)
	cm = np.asarray(cm)

	class_acc = __class_acc(cm, target_names)

	if len(target_names) == 2:
		tn_fp_fn_tp = confusion_matrix(truth,pred).ravel() # Deepcopy cm instead?
		tn_fp_fn_tp = np.asarray(tn_fp_fn_tp)

	# precision, recall, f1-score, support(num samples)
	clf_rep = classification_report(truth,pred, target_names=target_names, digits=6) # For printing

	if print_resu:
		print('='*70)
		print(clf_rep)
		print("Class Accuracies:",class_acc)
		print('Confusion Matrix:')
		print(cm)
		print('='*70)
		print('\n')

	clf_rep = classification_report(truth,pred, target_names=target_names,output_dict=True) # To be output as dict

	# ================ Save Performances ================
	performance = {}
	performance['class_acc'] = class_acc
	performance['cm'] = cm
	performance['clf_rep'] = clf_rep
	if len(target_names) == 2:
		performance['tn_fp_fn_tp'] = tn_fp_fn_tp

	if save_resu:
		save_path = os.path.join(save_path,'eval_resu')
		np.save(save_path,performance)

	return performance




class AccumulatedAccuracyMetric(Metric):
	"""
	Works with classification model
	"""

	def __init__(self):
		self.correct = 0
		self.total = 0

	def __call__(self, outputs, target, loss):
		pred = outputs[0].data.max(1, keepdim=True)[1]
		self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
		self.total += target[0].size(0)
		return self.value()

	def reset(self):
		self.correct = 0
		self.total = 0

	def value(self):
		return 100 * float(self.correct) / self.total

	def name(self):
		return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
	'''
	Counts average number of nonzero triplets found in minibatches
	'''

	def __init__(self):
		self.values = []

	def __call__(self, outputs, target, loss):
		self.values.append(loss[1])
		return self.value()

	def reset(self):
		self.values = []

	def value(self):
		return np.mean(self.values)

	def name(self):
		return 'Average nonzero triplets'
