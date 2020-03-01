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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.pipeline import Pipeline

import random

from joblib import dump, load
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
		t_s  =time.time()
		data, label = [], []
		for file in batch:
			data.append( np.load(file['sp_path']) )
			label.append( file['label'] )

		#print("min: {0}, max: {1}".format(np.array(data).min(), np.array(data).max()))

		data = scipy.sparse.csr_matrix( np.array(data) )

		if(self.pipe != None):
			data = self.pipe.transform(data)

		label = np.array(label)

		print("parse files: ", time.time()-t_s)
		return data, label



	def assign_pipe(self, pipeline):
		self.pipe = pipeline




def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 5#20
		
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

		count = [0]*174#num_classes

		for ex in csv_contents:
			ex['sp_path'] = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.npy'.format(ex['example_id'], layer))
			ex['class_count'] = count[ex['label']] 
			count[ex['label']] += 1

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		train_data = [ex for ex in train_data if ex['label'] < num_classes]

		count_limit = 200
		train_data = [ex for ex in train_data if ex['class_count'] < count_limit]

		train_batcher = BatchParser(train_data, batch_size=batch_size, shuffle=True)

		test_data = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]
		test_batcher = BatchParser(test_data, batch_size=batch_size, shuffle=False)



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
		clf = SGDClassifier(penalty='l2',)#verbose=1, tol=1e-4)#n_jobs=num_procs, 




		# TRAIN
		print("fitting model...")
		t_s = time.time()

		dataset_size = len(train_data)

		n_iter = 5
		cur_epoch = 0
		t_i = time.time()


		while train_batcher.epoch < n_iter:

			t_n = time.time()
			batch_data, batch_label = train_batcher.get_batch()

			#print(type(batch_data))
			#print(batch_label.dtype)

			clf.partial_fit(batch_data, batch_label, classes=np.arange(num_classes))
			print("{0}  elapsed: {1}".format(train_batcher.i, time.time()-t_n))


			if(train_batcher.epoch != cur_epoch):
				print("TRAIN {0}/{1}: ".format(train_batcher.epoch, n_iter, time.time() - t_i))
				cur_epoch = train_batcher.epoch
				t_i = time.time()

				print("elapsed:", time.time()-t_s)
		
				print("evaluating model...")
				t_s = time.time()
				pred, eval_label = [], []

				while test_batcher.epoch < 1:
					print(test_batcher.i)
					batch_data, batch_label = test_batcher.get_batch()

					#p = clf.predict(batch_data)
					#print(type(p.tolist()))

					pred += clf.predict(batch_data).tolist()
					eval_label += batch_label.tolist()
				test_batcher.epoch = 0
				test_batcher.i = 0

				cur_accuracy = metrics.accuracy_score(eval_label, pred)
				print("elapsed:", time.time()-t_s)


				# if model accuracy is good then replace the old model with new save data
				if(cur_accuracy > max_accuracy):
					save_model(clf, os.path.join(save_dir, "model"))
					max_accuracy = cur_accuracy

				print("ACCURACY: layer: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(layer, iteration, repeat, cur_accuracy, max_accuracy))
				print('------------')


'''

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
'''
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

	'''
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
	'''
	
	
