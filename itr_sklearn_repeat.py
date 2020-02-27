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



from joblib import dump, load
def save_model(clf, name):
	dump(clf, name+'.joblib') 

def load_model(name):
	return load(name+'.joblib') 

def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True, num_procs=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		#num_classes = 5
		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)


		parse_data = True
		if(parse_data):
			process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs)
		data_in, data_label, eval_in, eval_label = retrieve_data(dataset_dir, model_type, dataset_type, dataset_id, layer)

		print("data_in.shape:", data_in.shape)
		print("data_label.shape:", data_label.shape)
		print("eval_in.shape:", eval_in.shape)
		print("eval_label.shape:", eval_label.shape)


		#from thundersvm import SVC
		#clf = SVC(max_iter=1000, tol=1e-4, probability=True, kernel='linear', decision_function_shape='ovr')
		clf = SGDClassifier(max_iter=1000, tol=1e-4, n_jobs=num_procs)

		# TRAIN
		print("fitting model...")
		t_s = time.time()
		clf.fit(data_in, data_label)
		print("elapsed:", time.time()-t_s)
		
		print("evaluating model...")
		t_s = time.time()
		pred = clf.predict(eval_in)
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


	for layer in range(DEPTH_SIZE):
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
	
	
