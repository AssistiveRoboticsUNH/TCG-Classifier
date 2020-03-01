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

from itr_process import process_data, retrieve_data

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import random

from joblib import dump, load
import torch

from torch.utils.data import Dataset, DataLoader


def save_model(clf, name):
	dump(clf, name+'.joblib') 

def load_model(name):
	return load(name+'.joblib') 



class MyDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, dataset, transform=None, scaler=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.dataset = dataset
		self.pipe = None

		print("fit scaler")

		'''
		if (scaler == None):
			self.scaler= StandardScaler()

			num = 1000
			for i in range(0, len(self.dataset)/4, num):
				print(i)
				data = []
				for j in range(num):
					if (i+j < len(self.dataset)):
						file = self.dataset[i + j]
						data.append(np.load(file['sp_path']))
				data = np.array(data)

				self.scaler.partial_fit(data)
			print("scaler fit")
		else:
			self.scaler = scaler
		'''
		self.scaler= StandardScaler()


	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):


		if torch.is_tensor(idx):
			idx = idx.tolist()

		t_s = time.time()
		data, label = [], []

		file = self.dataset[idx]
		data.append( np.load(file['sp_path']) )
		label.append( file['label'] )

		'''
		if(self.pipe != None):
			#data = self.pipe.transform(data)
			data = self.scaler.transform(data)
		'''

		sample = {'data': np.array(data), 'label': np.array(label)}
		#print(type(np.array(data)), np.array(data).dtype)
		#print(type(np.array(label)), np.array(label).dtype)

		#if self.transform:
		#    sample = self.transform(sample)

		return sample

	def get_scaler(self):
		return self.scaler


def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 20
		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)

		parse_data = False
		if(parse_data):
			process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs)
		

		#batch_size = 1000

		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		for ex in csv_contents:
			ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		train_data = [ex for ex in train_data if ex['label'] < num_classes]
		print("Training Dataset Size: {0}".format(len(train_data)))
		train_batcher = MyDataset(train_data)

		test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		print("Evaluation Dataset Size: {0}".format(len(test_data)))
		test_batcher = MyDataset(test_data, scaler = train_batcher.get_scaler())



		batch_size = 10

		data_label = [ex['label'] for ex in train_data]
		class_sample_count = [Counter(data_label)[x] for x in range(num_classes)]#[10, 5, 2, 1] 
		print("class_sample_count:", class_sample_count)
		weights = (1 / torch.Tensor(class_sample_count))
		print("weights:", weights)
		

		weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights, num_samples=len(train_data))

		trainloader = torch.utils.data.DataLoader(train_batcher, batch_size=batch_size,
										  #shuffle=True, 
										  sampler=weighted_sampler, 
										  num_workers=2)
										  #shuffle=True, num_workers=2)

		testloader = torch.utils.data.DataLoader(test_batcher, batch_size=batch_size,
										 shuffle=False, num_workers=2)





		'''
		#hashvect = CountVectorizer(token_pattern=r"\d+\w\d+")#HashingVectorizer(n_features=2**17, token_pattern=r"\d+\w\d+")
		tfidf = TfidfTransformer(sublinear_tf=True)
		scale = StandardScaler(with_mean=False)


		pipe = Pipeline([
			('tfidf', tfidf),
			('scale', scale),
		])
		'''

		'''
		#enabled for TRN/I3D not for TSm
		#apply processing
		data_standard = train_batcher.get_sized_batch(batch_size*15)
		pipe.fit(data_standard)


		train_batcher.assign_pipe(pipe)
		test_batcher.assign_pipe(pipe)
		'''
		#from thundersvm import SVC
		#clf = SVC(max_iter=1000, tol=1e-4, probability=True, kernel='linear', decision_function_shape='ovr')
		


		

		

		if torch.cuda.is_available():
			device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
			print("Running on the GPU")
		else:
			device = torch.device("cpu")
			print("Running on the CPU")
  

		#model
		import torch.nn as nn
		import torch.nn.functional as F
		class Net(nn.Module):
			def __init__(self, input_size, num_classes):
				super(Net, self).__init__()
				n_hidden = 100

				#self.dense1 = nn.Linear(input_size, n_hidden)
				#self.dense2 = nn.Linear(n_hidden, num_classes)	

				self.dense = nn.Linear(input_size, num_classes)				

			def forward(self, x):
				return self.dense(x)
				
				x = self.dense1(x)#.double()
				x = F.relu(self.dense2(x))
				return x#.double()
				

		data_in = np.load(train_data[0]['sp_path'])
		print(data_in.shape)

		net = Net(data_in.shape[0], num_classes).to(device)



		counts = [0]*num_classes

		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


		t_s = time.time()
		for epoch in range(1):#20):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				batch = data
				inputs, labels = batch['data'], batch['label']
				labels = labels.reshape(-1)

				for l in labels:
					counts[l] += 1

				inputs = inputs.to(device).float()
				labels = labels.to(device)





				#print("inp: ", inputs.shape)
				#print("labels: ", labels.shape)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				outputs = np.squeeze(outputs)
				#print("outputs: ", outputs.shape)

				l1_regularization = torch.tensor(0, dtype=torch.float32, device=device)
				for param in net.parameters():
					#print("l1_regularization:", l1_regularization, "param:", torch.norm(param, 1))
					#print("l1_regularization:", l1_regularization.dtype, "param:", torch.norm(param, 1).dtype)
					l1_regularization += torch.norm(param, 1)#.type_as(output)

				loss = criterion(outputs, labels)# + l1_regularization
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
			print('[%d, %5d] loss: %.3f' % (epoch + 1, len(trainloader), running_loss / 2000))
			running_loss = 0.0

			correct = 0
			total = 0

			
			'''
			with torch.no_grad():
				for data in testloader:
					batch = data
					inputs, labels = batch['data'], batch['label']
					labels = labels.reshape(-1)

					inputs = inputs.to(device).float()
					labels = labels.to(device)

					outputs = net(inputs)
					outputs = np.squeeze(outputs)


					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					#print("predicted:", predicted)
					#print("labels:", labels.shape)




					correct += (predicted == labels).sum().item()

					#for p, l in zip(predicted, labels):
					#	print(p, l)
					#print(correct, total)

			print('eval: %d %%' % (
				100 * correct / total))
			'''

		print("train elapsed:", time.time()-t_s)
		print("counts:")
		for sample_cnt, prob, cnt in zip(class_sample_count, weights, counts ):
			print(sample_cnt, prob, cnt)
			

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

				inputs = inputs.to(device).float()
				labels = labels.to(device)

				outputs = net(inputs)
				outputs = np.squeeze(outputs)

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
