import os, sys, math, time
import numpy as np
from collections import Counter

import matplotlib
matplotlib.use('Agg')

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

from sklearn import metrics
from sklearn.linear_model import SGDClassifier

import scipy
import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor

#from itr_process import process_data, retrieve_data

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler ,RobustScaler
from sklearn.pipeline import Pipeline

import random

from joblib import dump, load
import torch

from torch.utils.data import Dataset#, DataLoader

if (sys.version[0] == '2'):
	from sets import Set
	import cPickle as pickle
	from sklearn.svm import SVC

from torch_geometric.data import Data

def save_model(clf, name):
	dump(clf, name+'.joblib') 



def load_model(name):
	return load(name+'.joblib') 



class MyDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, dataset, transform=None, scaler=None, prune=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.dataset = dataset
		self.pipe = None
		self.dataset_shape = None
		self.prune = prune

		print("fit scaler")
		#print(self.prune)

		
		if(self.dataset_shape == None):
			d = np.load(self.dataset[0]['sp_path'])

			self.dataset_shape = d.shape

			if(self.prune != None):
				self.dataset_shape = d[self.prune].shape
				print("prune shape:", self.dataset_shape)#, self.prune.shape)
		
		'''
		if (scaler == None):
			


			self.scaler= MinMaxScaler()#StandardScaler(with_mean=False)


			num = 1000
			for i in range(0, len(self.dataset), num):
				print(i)
				data = []
				for j in range(num):
					if (i+j < len(self.dataset)):
						file = self.dataset[i + j]

						d = np.load(file['sp_path'])

						data.append(d)
				data = np.array(data)

				self.scaler.partial_fit(data)



			print("scaler fit")




		else:
			self.scaler = scaler
		'''
		self.scaler= None#StandardScaler()


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):


		if torch.is_tensor(idx):
			idx = idx.tolist()

		t_s = time.time()
		data, label = [], []

		file = self.dataset[idx]

		d = np.load(file['sp_path']) 

		d = d.reshape(128, 128, 7)


		sparse_edges = []
		for itr in range(7):
			sparse_edges.append(scipy.sparse.coo_matrix(d[..., itr]))
		x = torch.tensor(np.arange(128).reshape(1, -1), dtype=torch.float)
		#print("x:", x.shape)

		d = Data(x=x, edge_index=sparse_edges[0], y=file['label'])
		return d

		#data.append( d )
		#label.append( file['label'] )



		
		#if(self.scaler != None):
		#	#data = self.pipe.transform(data)
		#	data = self.scaler.transform(data)
		#if(self.prune != None):
		#	data = data[..., self.prune]
		

		sample = {'data': d, 'label': np.array(label)}
		#print(type(np.array(data)), np.array(data).dtype)
		#print(type(np.array(label)), np.array(label).dtype)

		#if self.transform:
		#    sample = self.transform(sample)

		return sample

	def get_scaler(self):
		return self.scaler
		

def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):


	batch_size_g = 5#100#1000
	train_limit_g = 200#100000#400
	num_classes_g = 174
	alpha_g = 0.001
	l2_norm_g = 0.25#0.5
	l1_norm_g = 0##0.0001
	n_hidden_g = 64#128
	epoch_g = 150
	prunt_value = 0

	parse_data = False
	model_name = "model3.ckpt"
	scaler_name = "scaler"
	gen_scaler = False#True



	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = num_classes_g
		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)

		#if(parse_data):
		#	process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs=8)
		

		#batch_size = 1000

		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		count = [0]*174#num_classes

		for ex in csv_contents:
			ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))
			ex['class_count'] = count[ex['label']] 
			count[ex['label']] += 1


		##################
		# DATASET IO
		##################


		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		train_data = [ex for ex in train_data if ex['label'] < num_classes]

		count_limit = train_limit_g#500
		train_data = [ex for ex in train_data if ex['class_count'] < count_limit]


		train_idx = []
		for ex in train_data:
			if (len(train_idx) ==  0):
				train_idx = np.load(ex['sp_path'])
			else:
				train_idx += np.load(ex['sp_path'])

		train_prune = np.where(train_idx > prunt_value)

		scaler = None

		print("Training Dataset Size: {0}".format(len(train_data)))
		train_batcher = MyDataset(train_data, prune=train_prune, scaler = scaler)

		

		test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		print("Evaluation Dataset Size: {0}".format(len(test_data)))
		test_batcher = MyDataset(test_data, scaler = train_batcher.get_scaler(), prune=train_prune)


		##################
		# DATASET LOAD AND BALANCE
		##################

		
		batch_size = batch_size_g

		data_label = [ex['label'] for ex in train_data]
		class_sample_count = [Counter(data_label)[x] for x in range(num_classes)]#[10, 5, 2, 1] 
		print("class_sample_count:", class_sample_count)
		weights = (1 / torch.Tensor(class_sample_count).double())
		print("weights:", weights)



		sample_weights = [0]*len(train_data)
		for i, ex in enumerate(train_data):
			sample_weights[i] = weights[ex['label']]


		weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_data))


		from torch_geometric.data import DataLoader
		trainloader = DataLoader(train_batcher, batch_size=batch_size,
										  #shuffle=True, 
										  sampler=weighted_sampler, 
										  num_workers=2)
										  #shuffle=True, num_workers=2)

		testloader = DataLoader(test_batcher, batch_size=batch_size,
										 shuffle=False, num_workers=2)


		##################
		# DEFINE MODEL
		##################


		if torch.cuda.is_available():
			device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
			print("Running on the GPU")
		else:
			device = torch.device("cpu")
			print("Running on the CPU")
  

		#model
		import torch.nn as nn
		import torch.nn.functional as F
		import torch_geometric.nn as gcn 
		class Net(nn.Module):
			def __init__(self, dataset, num_classes):
				super(Net, self).__init__()
				n_hidden = n_hidden_g#32
				#n_hidden2 = 16

				#self.dense1 = nn.Linear(input_size, n_hidden)
				#self.dense2 = nn.Linear(n_hidden, num_classes)	

				self.conv1 = gcn.GCNConv(128, n_hidden)
				self.conv2 = gcn.GCNConv(n_hidden, num_classes)

				#self.dense2a = nn.Linear(n_hidden, n_hidden2)	
				#self.dense3a = nn.Linear(n_hidden2, num_classes)	

				#self.dense = nn.Linear(input_size, num_classes)	

				self.dropout = torch.nn.Dropout(p=0.5)			

			def forward(self, data):
				x, edge_index = data.x, data.edge_index
				print("data.x:", data.x)
				print("data.edge_index:", data.edge_index)

				x = self.conv1(x, edge_index)
				x = F.relu(x)
				x = F.dropout(x, training=self.training)
				x = self.conv2(x, edge_index)

				#x = self.dense(x)

				
				
				#x = F.leaky_relu(self.dense1(x))#.double()
				#x = self.dense2(x)
				
				#x = F.leaky_relu(self.dense1(x))
				#x = F.leaky_relu(self.dense2a(x))
				#x = self.dense3a(x)



				return x#self.dropout1(x)

		net = Net(trainloader, num_classes).to(device)
		#net.load_state_dict(torch.load(model_name))



		##################
		# TRAIN 
		##################


		counts = [0]*num_classes

		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		#optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
		optimizer = optim.Adam(net.parameters(), lr=alpha_g)

		t_s = time.time()
		for epoch in range(epoch_g):  # loop over the dataset multiple times

			net.train()
			running_loss = 0.0
			for i, data in enumerate(trainloader):
				# get the inputs; data is a list of [inputs, labels]
				inputs = data
				labels = data.y
				#print(batch)
				#inputs, labels = batch['data'], batch['label']
				#labels = labels.reshape(-1)

				for l in labels:
					counts[l] += 1

				#print(inputs[0].min(), inputs[0].max())

				inputs = inputs.to(device)#.float()
				labels = labels.to(device)

				#print(inputs[0].min(), inputs[0].max())


				train_labels = labels

				#print("inp: ", inputs.shape)
				#print("labels: ", labels.shape)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				print("inp:", inputs)
				outputs = net(inputs)

				outputs = outputs.reshape(-1, outputs.shape[-1])


				_, train_predicted = torch.max(outputs.data, 1)
				#print("train_predicted:", train_predicted.shape)
				#print("train_labels:", train_labels.shape)

				#outputs = np.squeeze(outputs)
				#print("outputs: ", outputs.shape)


				l1_lambda = l1_norm_g
				l2_lambda = l2_norm_g
				l2_regularization = torch.tensor(0, dtype=torch.float32, device=device)
				l1_regularization = torch.tensor(0, dtype=torch.float32, device=device)
				for param in net.parameters():
					#print("l1_regularization:", l1_regularization, "param:", torch.norm(param, 1))
					#print("l1_regularization:", l1_regularization.dtype, "param:", torch.norm(param, 1).dtype)
					l2_regularization += torch.norm(param, 2)#.type_as(output)
					l1_regularization += torch.norm(param, 1)

				loss = criterion(outputs, labels) + l1_regularization * l1_lambda + l2_regularization * l2_lambda#+ l1_regularization * l1_lambda
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
			





			torch.save(net.state_dict(), model_name)

			print('[%d, %5d] loss: %.3f' % (epoch + 1, len(trainloader), running_loss / 2000))
			running_loss = 0.0

			batch = next(iter(testloader))
			inputs, labels = batch['data'], batch['label']
			labels = labels.reshape(-1)

			#print(inputs[0].min(), inputs[0].max())

			inputs = inputs.to(device)#.float()
			labels = labels.to(device)

			net.eval()

			outputs = net(inputs)
			#outputs = np.squeeze(outputs)
			outputs = outputs.reshape(-1, outputs.shape[-1])

			_, predicted = torch.max(outputs.data, 1)

			print("train accuracy: ", (train_predicted == train_labels).sum().item() / float(len(train_labels)), "val accuracy: ", (predicted == labels).sum().item() / float(len(labels)))
			#print("predicted:", predicted)
			#print("labels:", labels.shape)





		net.eval()
		print("train elapsed:", time.time()-t_s)
		'''
		print("counts:")
		for sample_cnt, prob, cnt in zip(class_sample_count, weights, counts ):
			print(sample_cnt, prob, cnt)
		'''	

		print("evaluating model...")
		t_s = time.time()
		correct = 0
		total = 0

		pred_list = []
		label_list = []

		with torch.no_grad():
			for data in testloader:
				batch = data
				inputs, labels = batch['data'], batch['label']
				labels = labels.reshape(-1)

				inputs = inputs.to(device)#.float()
				labels = labels.to(device)

				outputs = net(inputs)
				#print("outputs.shape1", outputs.shape)
				outputs = outputs.reshape(-1, outputs.shape[-1])
				#outputs = np.squeeze(outputs)

				#print("outputs.shape2", outputs.shape)

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				predicted = predicted.cpu().data.numpy().tolist()
				labels = labels.cpu().data.numpy().tolist()

				pred_list += predicted
				label_list += labels

				#for p, l in zip(predicted, labels):
				#	print(p, l)
				#print(correct, total)
		print("test elapsed:", time.time()-t_s)

		print('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))


		
		import matplotlib.pyplot as plt

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


		#print(label_list, pred_list)

		plt.figure(figsize=(20,10))


		print(np.array(label_list).shape, np.array(pred_list).shape)

		from sklearn.metrics import confusion_matrix
		cm = confusion_matrix(label_list, pred_list)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


		plot_confusion_matrix(cm)
		plt.savefig('cm.png')


		# if model accuracy is good then replace the old model with new save data
		#if(cur_accuracy > max_accuracy):
		#	tcg.save_model(os.path.join(save_dir, "model"))
		#	max_accuracy = cur_accuracy

		#print("Training layer: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(layer, iteration, repeat, cur_accuracy, max_accuracy))
		



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

	parser.add_argument('--num_procs', type=int, default=1, help='number of process to split IAD generation over')
	parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat training the model')
	parser.add_argument('--parse_data', type=bool, default=True, help='whether to parse the data again or load from file')


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
		FLAGS.repeat,
		FLAGS.parse_data,
		FLAGS.num_procs
		)
