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

import random

from joblib import dump, load
def save_model(clf, name):
	dump(clf, name+'.joblib') 

def load_model(name):
	return load(name+'.joblib') 

class BatchParser:
	def __init__(self, dataset, batch_size=1, shuffle=False):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.i = 0
		self.epoch = 0

		if(self.shuffle):
			random.shuffle(self.dataset)

	def get_batch(self):

		end = self.i+self.batch_size 
		if(end > len(self.dataset)):
			diff = len(self.dataset) - end

			batch = self.dataset[self.i: len(self.dataset)] + self.dataset[0: diff]
			self.epoch += 1

			if(self.shuffle):
				random.shuffle(self.dataset)

		else:
			batch = self.dataset[self.i:end]
			self.i = end

		return self.parse_batch(batch)

	def parse_batch(self, batch):
		data, label = [], []
		for file in batch:
			data.append( np.load(file['sp_path']) )
			label.append( file['label'] )

		data = scipy.sparse.csr_matrix( np.array(data) )
		label = np.array(label)
		return data, label




def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 3
		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)

		print("process_data")
		parse_data = True
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
		train_batcher = BatchParser(train_data, batch_size=batch_size, shuffle=True)

		test_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		test_batcher = BatchParser(test_data, batch_size=batch_size, shuffle=False)





		#apply processing












		#from thundersvm import SVC
		#clf = SVC(max_iter=1000, tol=1e-4, probability=True, kernel='linear', decision_function_shape='ovr')
		clf = SGDClassifier(n_jobs=num_procs)




		# TRAIN
		print("fitting model...")
		t_s = time.time()

		dataset_size = len(train_data)

		n_iter = 1000
		while train_batcher.epoch < n_iter:
			batch_data, batch_label = train_batcher.get_batch()
			clf.partial_fit(batch_data, batch_label, classes=np.arange(num_classes))



		print("elapsed:", time.time()-t_s)
		
		print("evaluating model...")
		t_s = time.time()
		pred, eval_label = [], []

		while test_batcher.epoch < 1:
			batch_data, batch_label = test_batcher.get_batch()
			pred += clf.predict(batch_data, batch_label, classes=np.arange(num_classes))
			eval_label += batch_label

		cur_accuracy = metrics.accuracy_score(eval_label, pred)
		print("elapsed:", time.time()-t_s)


		# if model accuracy is good then replace the old model with new save data
		if(cur_accuracy > max_accuracy):
			save_model(clf, os.path.join(save_dir, "model"))
			max_accuracy = cur_accuracy

		print("ACCURACY: layer: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(layer, iteration, repeat, cur_accuracy, max_accuracy))
		print('------------')

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


	for layer in range(DEPTH_SIZE-1, -1, -1):
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
	
	
