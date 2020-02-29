import os, sys, math, time
import numpy as np
from collections import Counter

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



def save_model(clf, name):
	dump(clf, name+'.joblib') 

def load_model(name):
	return load(name+'.joblib') 

from multiprocessing import Pool

class BatchParser:
	def __init__(self, dataset, batch_size=1, shuffle=False, num_procs=1):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.i = 0
		self.epoch = 0
		self.pool = Pool(num_procs)
		self.pipe = None

		if(self.shuffle):
			random.shuffle(self.dataset)

	def get_batch(self):

		end = self.i+self.batch_size 
		if(end > len(self.dataset)):
			print("resample")
			diff = len(self.dataset) - end

			batch = self.dataset[self.i: len(self.dataset)] + self.dataset[0: diff]
			self.epoch += 1


			if(self.shuffle):
				random.shuffle(self.dataset)
			self.i = 0

		else:
			print("no resample")
			t_s = time.time()
			batch = self.dataset[self.i:end]
			self.i = end
			print("get files: ", time.time()-t_s)

		return self.parse_batch(batch)

	def get_sized_batch(self, batch_size):
		data, label = [], []
		for file in self.dataset[:batch_size]:
			data.append( np.load(file['sp_path']) )
			label.append( file['label'] )

		#print("min: {0}, max: {1}".format(np.array(data).min(), np.array(data).max()))
		print(np.array(data).shape)

		return scipy.sparse.csr_matrix( np.array(data) )

	def parse_batch(self, batch):
		



	def assign_pipe(self, pipeline):
		self.pipe = pipeline

class MyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
		self.dataset = dataset
		self.pipe = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):


        if torch.is_tensor(idx):
            idx = idx.tolist()

        t_s  =time.time()
		data, label = [], []
		for file in self.dataset[idx]:
			data.append( np.load(file['sp_path']) )
			label.append( file['label'] )


		if(self.pipe != None):
			data = self.pipe.transform(data)


        sample = {'data': np.array(data), 'label': np.array(label)}

        #if self.transform:
        #    sample = self.transform(sample)

        return sample


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
		

		batch_size = 1000

		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		for ex in csv_contents:
			ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		train_data = [ex for ex in train_data if ex['label'] < num_classes]
		train_batcher = MyDataset(train_data)

		test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		test_batcher = MyDataset(test_data)

		trainloader = torch.utils.data.DataLoader(train_batcher, batch_size=10,
                                          shuffle=True, num_workers=2)

		testloader = torch.utils.data.DataLoader(test_batcher, batch_size=10,
                                          shuffle=False, num_workers=2)


		print("Training Dataset Size: {0}".format(len(train_data)))
		print("Evaluation Dataset Size: {0}".format(len(test_data)))




		#hashvect = CountVectorizer(token_pattern=r"\d+\w\d+")#HashingVectorizer(n_features=2**17, token_pattern=r"\d+\w\d+")
		tfidf = TfidfTransformer(sublinear_tf=True)
		scale = StandardScaler(with_mean=False)


		pipe = Pipeline([
			('tfidf', tfidf),
			('scale', scale),
		])

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
		

		batch_size = 10


		class_sample_count = [Counter(data_label)[x] for x in range(num_classes)]#[10, 5, 2, 1] 
		weights = (1 / torch.Tensor(class_sample_count))
		print("weights:", weights)
		
		weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, data_in.shape[0])

		trainloader = torch.utils.data.DataLoader(zip(data_in, data_label), batch_size=batch_size,
										  sampler=weighted_sampler, num_workers=2)
										  #shuffle=True, num_workers=2)

		testloader = torch.utils.data.DataLoader(zip(eval_in, eval_label), batch_size=batch_size,
										 shuffle=False, num_workers=2)


		

		if torch.cuda.is_available():
		    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
		    print("Running on the GPU")
		else:
		    device = torch.device("cpu")
		    print("Running on the CPU")


		#model
		import torch.nn as nn
		class Net(nn.Module):
			def __init__(self, input_size, num_classes):
				super(Net, self).__init__()
				n_hidden= 1024

				#self.dense1 = nn.Linear(input_size, n_hidden)
				#self.dense2 = nn.Linear(n_hidden, num_classes)	

				self.dense = nn.Linear(input_size, num_classes)				

			def forward(self, x):
				return self.dense(x)
				'''
				x = self.dense1(x)#.double()
				return self.dense2(x)#.double()
				'''

		data_in = np.load(train_data[0]['sp_path'])

		net = Net(data_in.shape[1], num_classes).to(device)




		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		t_s = time.time()
		for epoch in range(10):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				inputs = inputs.to(device).float()
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
			print('[%d, %5d] loss: %.3f' % (epoch + 1, len(trainloader), running_loss / 2000))
			running_loss = 0.0

			correct = 0
			total = 0
			with torch.no_grad():
				for data in testloader:
					inputs, labels = data

					inputs = inputs.to(device).float()
					labels = labels.to(device)

					outputs = net(inputs)

					_, predicted = torch.max(outputs.data, 1)
					total += labels.size(0)
					correct += (predicted == labels).sum().item()

					#for p, l in zip(predicted, labels):
					#	print(p, l)
					#print(correct, total)

			print('eval: %d %%' % (
				100 * correct / total))


		print("train elapsed:", time.time()-t_s)
			

		print("evaluating model...")
		t_s = time.time()
		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				inputs, labels = data

				inputs = inputs.to(device).float()
				labels = labels.to(device)

				outputs = net(inputs)

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				#for p, l in zip(predicted, labels):
				#	print(p, l)
				#print(correct, total)
		print("test elapsed:", time.time()-t_s)

		print('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))

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
