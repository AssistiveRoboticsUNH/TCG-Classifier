import os, sys, math, time, random
from collections import Counter

import numpy as np
import scipy

from sklearn.metrics import confusion_matrix
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
from itr_process import process_data, retrieve_data

if (sys.version[0] == '2'):
	import cPickle as pickle

'''
generate_parser=generate_parser, limit_examples=num_example_limit
train_data = [ex for ex in train_data if ex['label'] < num_classes]

scaler = None
	if(not gen_scaler):
		scaler = pickle.load(open(scaler_name+'.pk', "rb"))

if(gen_scaler):
		with open(scaler_name+'.pk', 'wb') as file_loc:
			pickle.dump(train_batcher.get_scaler(), file_loc)

parsers = []	
'''

def open_as_raw(ex):
	return np.load(ex['sp_path'])
	
def open_as_sparse(ex):
	data = open_as_raw(ex)

	idx = np.nonzero(data)[0]
	value = data[idx]

	return zip(idx, value)

class DataIterable:
	def __init__(self, data, parse_function):
		self.a = 0
		self.data = data
		self.parse_function = parse_function

	def __iter__(self):
		self.a = 0
		return self

	def next(self):
		if(self.a < len(self.data)):

			if(self.a % 100 == 0):
				print("a:", self.a)

			x = self.parse_function(self.data[self.a])
			self.a += 1

			return x
		else:
			raise StopIteration

class Params:
	def __init__(self, num_classes = -1, examples_per_class = -1):
		self.num_classes = num_classes if num_classes > 0 else sys.maxint
		self.examples_per_class = examples_per_class if examples_per_class > 0 else sys.maxint

class ITRDataset:
	def __init__(self, csv_contents, param_list=None, parsers=[], device=torch.device("cpu")):
		self.csv_contents = csv_contents
		self.device = device

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

		self.parsers = parsers

		self.shape = open_as_raw(self.csv_contents[0]).shape # inputs shape
		print ("len(self.csv_contents):", len(self.csv_contents))

	def __len__(self):
		print ("len(self.csv_contents):", len(self.csv_contents))
		return len(self.csv_contents)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		ex = self.csv_contents[idx]

		data = open_as_sparse(ex) 

		# apply any preprocessing defined by the parsers
		for parser in self.parsers:
			data = parser.transform(data)

		unzipped_data = np.array(zip(*(data[0])))
		print ("unzipped:", np.array(unzipped_data).shape)
		print(unzipped_data[0][:10], max(unzipped_data[0]), np.array(unzipped_data[0]).dtype)
		print(unzipped_data[1][:10], max(unzipped_data[1]), np.array(unzipped_data[1]).dtype)

		data = np.zeros(128*128*7)
		print("data:", data.shape)
		data[unzipped_data[0]] = unzipped_data[1]

		return {'data': np.array(data), 'label': np.array([ex['label']])}

	


def organize_data(csv_filename, dataset_dir, model_type, dataset_type, dataset_id, layer, num_classes,
		generate_itrs, train_param_list, test_param_list, batch_size):

	# -----------------
	# CSV Parsing and ITR Extraction
	# -----------------

	# extract ITR counts and dave them to file for quick, iterative learning
	if(generate_itrs):
		process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs=8)
	
	# open the csv file
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	# add current path to csv context
	for ex in csv_contents:
		ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))

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
									  sampler=weighted_sampler, num_workers=2) # do not set shuffle to true when using a sampler

	testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
									 shuffle=False, num_workers=2)

	return train_dataset, trainloader, test_dataset, testloader



def gen_tfidf(dataset, save_name):
	# fit TF-IDF
	iterable = iter(DataIterable(dataset.csv_contents, open_as_sparse))

	tfidf = TfIdfTransformer()
	tfidf.fit(iterable)

	# save tfidf
	with open(save_name+'.pk', 'wb') as file_loc:
		pickle.dump(tfidf, file_loc)

	return tfidf

def load_tfidf(save_name):
	return pickle.load(open(save_name+'.pk', "rb"))


def define_model(input_size, num_classes):

	#define device being used
	if torch.cuda.is_available():
		# you can continue going on here, like cuda:1 cuda:2....etc. 
		device = torch.device("cuda:0") 
		print("Running on the GPU")
	else:
		device = torch.device("cpu")
		print("Running on the CPU")

	#model
	class Net(nn.Module):
		def __init__(self, input_size, num_classes):
			super(Net, self).__init__()
			n_hidden = 512

			self.dense1 = nn.Linear(input_size, n_hidden)
			self.dense2 = nn.Linear(n_hidden, num_classes)	

			#self.dropout = torch.nn.Dropout(p=0.1)			

		def forward(self, x):
			x = F.leaky_relu(self.dense1(x))
			x = self.dense2(x)
			
			return x


	net = Net(input_size, num_classes).to(device)
	return net, device


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

def train(net, trainloader, testloader, device, num_epochs=10, alpha=0.0001, model_name='model.ckpt'):

	import torch.optim as optim
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=alpha)

	t_s = time.time()
	for epoch in range(num_epochs):  # loop over the dataset multiple times

		# -------------------
		# Optimize for the training epoch
		# -------------------

		net.train()
		running_loss = 0.0
		for i, data in enumerate(trainloader, start=0):

			# get the inputs; data is a list of [inputs, labels]
			batch = data
			inputs, labels = batch['data'], batch['label'].reshape(-1)

			inputs = inputs.to(device).float()
			labels = labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs).reshape(-1, outputs.shape[-1])
			_, train_predicted = torch.max(outputs.data, 1)


			# claculate regularization
			l1_lambda = l1_norm_g
			l2_lambda = l2_norm_g
			l2_regularization = torch.tensor(0, dtype=torch.float32, device=device)
			l1_regularization = torch.tensor(0, dtype=torch.float32, device=device)
			for param in net.parameters():
				l2_regularization += torch.norm(param, 2)
				l1_regularization += torch.norm(param, 1)
			
			#loss = criterion(outputs, labels) + l1_regularization * l1_lambda + l2_regularization * l2_lambda#+ l1_regularization * l1_lambda
			loss = criterion(outputs, labels) 
			
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()

		# save model after each epoch	
		torch.save(net.state_dict(), model_name)

		# update running loss
		print('[%d, %5d] loss: %.3f' % (epoch + 1, len(trainloader), running_loss / 2000))
		running_loss = 0.0


		# -------------------
		# Evaluate for the training epoch
		# -------------------

		# place net in evaluate mode 
		net.eval()

		# evaluate on random batch from test loader
		batch = next(iter(testloader))
		inputs, labels = batch['data'], batch['label'].reshape(-1)

		inputs = inputs.to(device).float()
		labels = labels.to(device)

		outputs = net(inputs).reshape(-1, outputs.shape[-1])
		_, predicted = torch.max(outputs.data, 1)

		print("train accuracy: ", (train_predicted == train_labels).sum().item() / float(len(train_labels)), "val accuracy: ", (predicted == labels).sum().item() / float(len(labels)))
	
	print("train elapsed:", time.time()-t_s)


def evaluate(net, testloader, device):

	net.eval()

	print("evaluating model...")
	t_s = time.time()
	correct = 0
	total = 0

	pred_list = []
	label_list = []

	with torch.no_grad():
		for i, data in enumerate(testloader, start=0):
			batch = data
			inputs, labels = batch['data'], batch['label'].reshape(-1)

			inputs = inputs.to(device).float()
			labels = labels.to(device)

			outputs = net(inputs).reshape(-1, outputs.shape[-1])
			outputs = outputs

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()

			'''
			# used for confusion matrix
			predicted = predicted.cpu().data.numpy().tolist()
			labels = labels.cpu().data.numpy().tolist()

			pred_list += predicted
			label_list += labels
			'''
	#viz_confusion_matrix(label_list, label_list)
			
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
	print("test elapsed:", time.time()-t_s)




def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer,
		num_classes,
		parse_data, num_procs):

	num_classes = 3
	examples_per_class = 50

	train_param_list = Params(num_classes=num_classes, examples_per_class=examples_per_class)
	test_param_list = Params(num_classes=num_classes)
	batch_size = 100
	generate_itrs = False
	num_epochs = 10
	alpha = 0.0001
	load_model = False
	model_name = "model.ckpt"
	tfidf_name = "tfidf"

	fit_tfidf = True

	train_dataset, trainloader, test_dataset, testloader = organize_data(
		csv_filename, dataset_dir, model_type, dataset_type, dataset_id, layer, num_classes,
		generate_itrs, train_param_list, test_param_list, batch_size)

	#TF-IDF
	if(fit_tfidf):
		tfidf = gen_tfidf(train_dataset, tfidf_name)
	else:
		tfidf = load_tfidf(tfidf_name)

	'''
	#Scaler
	if(fit_scaler):
		gen_scaler(dataset)
	else:
		load_scaler()
	'''

	# define network
	input_size  = train_dataset.shape[0]
	net, device = define_model(input_size, num_classes)
	if(load_model):
		net.load_state_dict(torch.load(load_model))

	# add parsers to model
	parsers = [tfidf]
	train_dataset.parsers = parsers
	train_dataset.device = device

	test_dataset.parsers = parsers
	test_dataset.device = device


	train(net, trainloader, testloader, device, num_epochs, alpha, model_name)

	evaluate(net, testloader, device)


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
		from tsm_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT

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
