import os, sys, math, time, random
from collections import Counter

import numpy as np
import scipy

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from gensim.sklearn_api import TfIdfTransformer

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv
#from itr_process import process_data, retrieve_data

if (sys.version[0] == '2'):
	import cPickle as pickle
else:
	import pickle



def open_as_raw(ex):
	return np.load(ex['sp_path'])
	
def open_as_sparse(ex):
	data = open_as_raw(ex)

	idx = np.nonzero(data)[0]
	value = data[idx]

	return list(zip(idx, value))

class DataIterable:
	def __init__(self, data, parse_function):
		self.a = 0
		self.data = data
		self.parse_function = parse_function

	def __iter__(self):
		self.a = 0
		return self

	def __next__(self):#use next(self) for Python2
		if(self.a < len(self.data)):

			if(self.a % 100 == 0):
				print("a: {0}/{1}".format(self.a, len(self.data)))

			x = self.parse_function(self.data[self.a])
			self.a += 1

			return x
		else:
			raise StopIteration

class Params:
	def __init__(self, num_classes = -1, examples_per_class = -1):
		self.num_classes = num_classes if num_classes > 0 else sys.maxsize
		self.examples_per_class = examples_per_class if examples_per_class > 0 else sys.maxsize

class ITRDataset:
	def __init__(self, csv_contents, param_list=None):
		self.csv_contents = csv_contents

		# Modify the dataset according to the rules laid forth by the param_list
		self.params = param_list

		# limit data by specific classes
		self.csv_contents = [ex for ex in self.csv_contents if ex['label'] < self.params.num_classes]
		
		# limit data by examples per class
		count = [0]*self.params.num_classes
		for ex in self.csv_contents:
			ex['class_count'] = count[ex['label']]   # label each example based on if they are teh 1st, 2nd, Nth example of that label
			count[ex['label']] += 1
		self.csv_contents = [ex for ex in self.csv_contents if ex['class_count'] < self.params.examples_per_class]

		self.shape = open_as_raw(self.csv_contents[0]).shape # inputs shape

	def __len__(self):
		return len(self.csv_contents)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.csv_contents[idx]
		data = scipy.sparse.load_npz(ex['sp_path']).toarray()
		data = data.reshape(-1)


		return {'data': np.array(data), 'label': np.array([ex['label']])}

	


def organize_data(csv_filename, dataset_dir, model_type, dataset_type, dataset_id, #layer, 
	num_classes,
		generate_itrs, train_param_list, test_param_list, batch_size):

	# -----------------
	# CSV Parsing and ITR Extraction
	# -----------------

	# extract ITR counts and dave them to file for quick, iterative learning
	'''
	if(generate_itrs):
		process_data(dataset_dir, model_type, dataset_type, dataset_id, #layer, 
			csv_filename, num_classes, num_procs=8)
	'''
	# open the csv file
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	# add current path to csv context
	for ex in csv_contents:
		ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}.npy'.format(ex['example_id']))
		ex['pp_path'] = os.path.join(dataset_dir, 'pp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}.npz'.format(ex['example_id']))

		#ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))
		#ex['pp_path'] = os.path.join(dataset_dir, 'pp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy.npz'.format(ex['example_id'], layer))


	# -----------------
	# Dataset Definition
	# -----------------

	# Setup train data reader
	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
	train_dataset = ITRDataset(train_data, param_list=train_param_list)
	print("Training Dataset Size: {0}".format(len(train_data)))


	test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	test_dataset = ITRDataset(test_data, param_list=test_param_list)
	print("Evaluation Dataset Size: {0}".format(len(test_data)))
	
	# -----------------
	# Dataset Loaders
	# -----------------

	# balance drawing of training samples, determine what class weights currently are
	sample_data = train_dataset.csv_contents[:]

	label_counts = [ex['label'] for ex in sample_data]
	class_sample_count = [Counter(label_counts)[x] for x in range(train_param_list.num_classes)]
	weights = (1 / torch.Tensor(class_sample_count).double())

	sample_weights = [0]*len(sample_data)
	for i, ex in enumerate(sample_data):
		sample_weights[i] = weights[ex['label']]

	# build weighted sampler
	weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_data))

	trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
									  sampler=weighted_sampler, num_workers=8, pin_memory = True) # do not set shuffle to true when using a sampler

	testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
									 shuffle=False, num_workers=2, pin_memory = True)

	return train_dataset, trainloader, test_dataset, testloader



def gen_tfidf(dataset, save_name):
	# fit TF-IDF
	print("Fitting TFIDF")
	iterable = iter(DataIterable(dataset.csv_contents, open_as_sparse))

	tfidf = TfIdfTransformer()
	tfidf.fit(iterable)

	print("Fit TFIDF Done")

	# save tfidf
	with open(save_name+'.pk', 'wb') as file_loc:
		pickle.dump(tfidf, file_loc)

	return tfidf

def load_tfidf(save_name):
	return pickle.load(open(save_name+'.pk', "rb"))

def gen_scaler(dataset, save_name):
	# fit TF-IDF
	
	
	iterable = iter(DataIterable(dataset.csv_contents, open_as_sparse))

	scaler = MinMaxScaler()

	for ex in iterable:
	#for i in range(len(dataset.csv_contents)):
		print("ex:", type(ex), len(ex))#, ex["data"].shape)
		print(ex[0])
		print(ex[1])

		scaler.partial_fit(ex.reshape(1, -1))
	
	print("Fit Scaler Done")

	# save tfidf
	with open(save_name+'.pk', 'wb') as file_loc:
		pickle.dump(scaler, file_loc)

	return scaler

def load_scaler(save_name):
	return pickle.load(open(save_name+'.pk', "rb"))


def define_model(input_size, num_classes, alpha=0.001):

	
	#from sklearn.svm import SVC
	from sklearn.linear_model import SGDClassifier
	clf = SGDClassifier(loss='hinge', alpha=alpha, n_jobs=4)#4)#, verbose=1)

	'''
	from sklearn.ensemble import AdaBoostClassifier
	clf = AdaBoostClassifier(
			#SVC(probability=True, kernel='linear'),
			SGDClassifier(loss='hinge'),
		#n_estimators=50,       
		#learning_rate=1.0, 
		algorithm='SAMME'
	)
	'''
	return clf, None


def viz_confusion_matrix(label_list, predictions):

	target_names = range(num_classes)

	def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

	plt.figure(figsize=(20,10))

	cm = confusion_matrix(label_list, predictions)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	# plot and save the confusion matrix
	plot_confusion_matrix(cm)
	plt.savefig('cm.png')

def data_to_sparse_matrix(dataloader, single=False):

	if(single):
		batch = next(iter(dataloader))
		data, labels = [batch['data'].numpy()], [batch['label'].numpy().reshape(-1)]
	else:
		data, labels = [],[]
		for i, batch in enumerate(dataloader, start=0):
			# get the inputs; data is a list of [inputs, labels]
			inp_data, inp_label = batch['data'].numpy(), batch['label'].numpy().reshape(-1)

			data.append(inp_data)
			labels.append(inp_label)

	data = scipy.sparse.coo_matrix(np.array(data)[0])
	labels = np.array(labels)[0]

	return data, labels

def train(net, trainloader, testloader, device, num_classes, num_epochs=10, alpha=0.0001, model_name='model.ckpt', scaler=None):

	avg_file_io = 0
	avg_train = 0

	t_s = time.time()
	for e in range(num_epochs):
		for i, batch in enumerate(trainloader, start=0):
			#print("i:", i)
			if (i % 50 == 0):
				#print("file IO:", avg_file_io / 50)
				#print("train:", avg_train / 50)

				avg_file_io = 0
				avg_train = 0

				print("i:", i, time.time()-t_s)
				t_s = time.time()

			t_i = time.time()
			# get the inputs; data is a list of [inputs, labels]
			inp_data, inp_label = batch['data'].numpy(), batch['label'].numpy().reshape(-1)
			

			inp_data = scipy.sparse.coo_matrix(np.array(inp_data))
			inp_label = np.array(inp_label)

			avg_file_io += time.time()-t_i

			t_i = time.time()


			net.partial_fit(inp_data, inp_label, classes=np.arange(num_classes))

			avg_train += time.time()-t_i
		#print("train elapsed:", time.time()-t_s)
		


		print("train accuracy:", evaluate(net, trainloader, device, max_iter=200))
		

		#test_data, test_labels = data_to_sparse_matrix(testloader, single=True)
		#print("eval accuracy:", net.score(test_data, test_labels))
		print("test accuracy:", evaluate(net, testloader, device))
		
		print('------------------')










	'''
	train_data, train_labels = data_to_sparse_matrix(trainloader, single=False)

	t_s = time.time()
	net.fit(train_data, train_labels)
	print("train elapsed:", time.time()-t_s)
	print("train accuracy:", net.score(train_data, train_labels))

	# Test Quick
	test_data, test_labels = data_to_sparse_matrix(testloader, single=True)
	print("eval accuracy:", net.score(test_data, test_labels))
	'''
'''

def fast_evaluate(net, testloader, start, end):

	pred_label, actual_label = [],[] 

	iter_tl = iter(testloader)
	for i in range(start, end):
		batch = iter_tl[i]

		if(i % 50 == 0):
			print("j:", i)

		if(max_iter > 0 and i > max_iter):
			break
		
		# get the inputs; data is a list of [inputs, labels]
		inp_data, inp_label = batch['data'].numpy(), batch['label'].numpy().reshape(-1)
		
		test_data = scipy.sparse.coo_matrix(np.array(inp_data))
		test_labels = np.array(inp_label)

		pred = net.predict(test_data)

		pred_label.append(pred)
		actual_label.append(test_labels)

	actual_label = np.concatenate(actual_label).reshape(-1)
	pred_label = np.concatenate(pred_label).reshape(-1)

	return (pred_label, actual_label)
def evaluate(net, testloader, device, max_iter=-1):

	from multiprocessing import Pool

	bundle = []
	for 


	p = Pool(4)
    print(p.map(fast_evaluate, bundle))

	return accuracy_score(y_true = actual_label, y_pred = pred_label )


'''
def evaluate(net, testloader, device, max_iter=-1):

	pred_label, actual_label = [],[] 

	for i, batch in enumerate(testloader, start=0):
		if(i % 50 == 0):
			print("j:", i)

		if(max_iter > 0 and i > max_iter):
			break
		
		# get the inputs; data is a list of [inputs, labels]
		inp_data, inp_label = batch['data'].numpy(), batch['label'].numpy().reshape(-1)
		
		test_data = scipy.sparse.coo_matrix(np.array(inp_data))
		test_labels = np.array(inp_label)

		pred = net.predict(test_data)

		pred_label.append(pred)
		actual_label.append(test_labels)

	actual_label = np.concatenate(actual_label).reshape(-1)
	pred_label = np.concatenate(pred_label).reshape(-1)

	#print("actual:", actual_label.shape, "pred:", pred_label.shape)
	#print("actual:", actual_label, "pred:", pred_label)

	return accuracy_score(y_true = actual_label, y_pred = pred_label )


def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, 
	#layer,
		num_classes,
		parse_data, num_procs):

	num_classes = 3#174#10#174#3
	examples_per_class = 20#100000#100#50

	train_param_list = Params(num_classes=num_classes, examples_per_class=examples_per_class)
	test_param_list = Params(num_classes=num_classes)
	batch_size = 100
	generate_itrs = True
	num_epochs = 100
	alpha = 0.0001
	load_model = False
	model_name = "svm.ckpt"
	tfidf_name = "tfidf_128"
	scaler_name = "scaler_128"

	fit_scaler = True#False#True#False#True
	fit_tfidf = True#False#True#False#True#False#True

	train_dataset, trainloader, test_dataset, testloader = organize_data(
		csv_filename, dataset_dir, model_type, dataset_type, dataset_id, #layer, 
		num_classes,
		generate_itrs, train_param_list, test_param_list, batch_size)

	#TF-IDF
	if(fit_tfidf):
		tfidf = gen_tfidf(train_dataset, tfidf_name)
	else:
		tfidf = load_tfidf(tfidf_name)

	train_dataset.sparse_parsers = [tfidf]
	test_dataset.sparse_parsers = [tfidf]

	#Scaler
	if(fit_scaler):
		scaler = gen_scaler(train_dataset, scaler_name)
	else:
		scaler = load_scaler(scaler_name)

	# define network
	input_size  = train_dataset.shape[0]
	net, _ = define_model(input_size, num_classes, alpha=alpha)

	# add parsers to model
	train_dataset.dense_parsers = [scaler]
	test_dataset.dense_parsers = [scaler]

	device = None
	train(net, trainloader, testloader, device, num_classes, num_epochs, alpha, model_name, scaler)
	print("Final Eval accuracy:", evaluate(net, testloader, device))


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'trn', 'tsm'])

	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')

	parser.add_argument('--parse_data', type=bool, default=True, help='whether to parse the data again or load from file')
	parser.add_argument('--num_procs', type=int, default=1, help='number of process to split IAD generation over')

	FLAGS = parser.parse_args()

	if(FLAGS.model_type == 'i3d'):
		from gi3d_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'rn50'):
		from rn50_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'trn'):
		from trn_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'tsm'):
		from tsm_wrapper3 import DEPTH_SIZE, CNN_FEATURE_COUNT

	'''		
	layer = DEPTH_SIZE-1
	main(FLAGS.model_type,
		FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		layer,
		FLAGS.num_classes,
		FLAGS.parse_data,
		FLAGS.num_procs
		)
		'''
	main(FLAGS.model_type,
		FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		#layer,
		FLAGS.num_classes,
		FLAGS.parse_data,
		FLAGS.num_procs
		)
