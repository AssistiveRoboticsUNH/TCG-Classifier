import numpy as np 





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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler ,RobustScaler
from sklearn.pipeline import Pipeline

import random

from joblib import dump, load
import torch

from torch.utils.data import Dataset, DataLoader

if (sys.version[0] == '2'):
	from sets import Set
	import cPickle as pickle
	from sklearn.svm import SVC

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
		
		if(self.dataset_shape == None):
			d = np.load(self.dataset[0]['sp_path'])
			self.dataset_shape = d.shape

			if(self.prune != None):
				self.dataset_shape = d[self.prune].shape
				print("prune shape:", self.dataset_shape)


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

		else:
			self.scaler = scaler
		

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):


		if torch.is_tensor(idx):
			idx = idx.tolist()

		t_s = time.time()
		data, label = [], []

		file = self.dataset[idx]

		d = np.load(file['sp_path']) 
		
		data.append( d )
		label.append( file['label'] )

		if(self.scaler != None):
			#data = self.pipe.transform(data)
			data = self.scaler.transform(data)
		if(self.prune != None):
			data = data[..., self.prune]
		
		return {'data': np.array(data), 'label': np.array(label)}

	def get_scaler(self):
		return self.scaler


def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):


	batch_size_g = 100#1000
	train_limit_g = 100#200#100000#400
	num_classes_g = 3#174
	alpha_g = 0.000001
	l2_norm_g = 0.25#0.5
	l1_norm_g = 0##0.0001
	n_hidden_g = 64#128
	epoch_g = 1000
	prunt_value = 0#3

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

		if(parse_data):
			process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs=8)
		

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


		




		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		train_data = [ex for ex in train_data if ex['label'] < num_classes]

		count_limit = train_limit_g#500
		train_data = [ex for ex in train_data if ex['class_count'] < count_limit]

		'''
		train_idx = []
		for ex in train_data:
			if (len(train_idx) ==  0):
				train_idx = np.load(ex['sp_path'])
			else:
				train_idx += np.load(ex['sp_path'])

		train_prune = np.where(train_idx > prunt_value)

		scaler = None
		if(not gen_scaler):
			scaler = pickle.load(open(scaler_name+'.pk', "rb"))


		print("Training Dataset Size: {0}".format(len(train_data)))
		train_batcher = MyDataset(train_data, prune=train_prune, scaler = scaler)

		if(gen_scaler):
			with open(scaler_name+'.pk', 'wb') as file_loc:
				pickle.dump(train_batcher.get_scaler(), file_loc)

		test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		print("Evaluation Dataset Size: {0}".format(len(test_data)))
		test_batcher = MyDataset(test_data, scaler = train_batcher.get_scaler(), prune=train_prune)

		max_v, min_v = 0,0

		for batch in train_batcher:
			max_v = max(max_v, batch["data"].max())
			min_v = max(min_v, batch["data"].min())

			#print(batch["data"].min(), batch["data"].max())
		print("max_v:", max_v, "min_v:", min_v)
		'''




		class CSVIteratable:
			def __init__(self, csv_list):
				self.csv_list = csv_list
				self.a = 1

				self.dict = False

				print("len DB: ", len(self.csv_list))

			def __iter__(self):
				self.a = 0
				return self

			#def __next__(self):
			def next(self):
				if(self.a < len(self.csv_list)):
					if(self.a % 100 == 0):
						print("{0}/{1}".format(self.a, len(self.csv_list)))
					x = self.open_file(self.csv_list[self.a])
					self.a += 1

					if(self.dict):
						return dict(x)
					return x
				else:
					raise StopIteration

			def open_file(self, file):
				data =  np.load(file['sp_path'])
				#convert to [(id, count), (id, count)]

				idx = np.nonzero(data)[0]
				value = data[idx]

				return zip(idx, value)



		train_data = train_data[:5]
		csv_itr = CSVIteratable(train_data)
		csv_iter = iter(csv_itr)


		

		from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
		#from gensim.models import TfidfModel
		from gensim.sklearn_api import TfIdfTransformer


		tfidf = TfIdfTransformer()
		tfidf.fit(csv_iter)#, normalize=True)

		csv_itr.dict = True
		csv_iter = iter(csv_itr)

		#x = csv_itr.next()
		#print(type(x))
		#print(x)

		
		data = tfidf.transform(csv_iter)
		print(tfidf)
		print(data.shape)
		








			


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
