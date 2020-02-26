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



def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 10


		tcg = ITR_Extractor(num_classes)		
		
		#open files
		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		path = 'b_path_{0}'.format(layer)
		for ex in csv_contents:
			ex[path] = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.b'.format(ex['example_id'], layer))

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]

		train_data = [ex for ex in train_data if ex['label'] < num_classes]
		test_data = [ex for ex in test_data if ex['label'] < num_classes]

		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)


		train_filename = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'train_{0}_{1}.spz.npz'.format(dataset_id, layer))
		test_filename  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'test{0}_{1}.spz.npz'.format(dataset_id, layer))
		train_label_filename = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'train_label_{0}_{1}.spz.npy'.format(dataset_id, layer))
		test_label_filename  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'test_label{0}_{1}.spz.npy'.format(dataset_id, layer))

		parse_data = True#not os.path.exists(train_filename)

		if(parse_data):
			# TRAIN
			in_files = [ex[path] for ex in train_data]
			in_labels = [ex['label'] for ex in train_data]

			print("adding train data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))


			data_in = tcg.tfidf.fit_transform(tcg.corpus)
			data_label = np.array(tcg.labels)

			scipy.sparse.save_npz(train_filename, data_in)
			np.save(train_label_filename, data_label)


			in_files = [ex[path] for ex in test_data]
			in_labels = [ex['label'] for ex in test_data]
			print("adding eval data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_eval_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))

			eval_in = tcg.tfidf.transform(tcg.evalcorpus)
			eval_label = np.array(tcg.evallabels)

			scipy.sparse.save_npz(test_filename, eval_in)
			np.save(test_label_filename, eval_label)


		else:
			data_in = scipy.sparse.load_npz(train_filename)
			data_label = np.load(train_label_filename)

			eval_in = scipy.sparse.load_npz(test_filename)
			eval_label = np.load(test_label_filename)

		from thundersvm import SVC
		#clf = SVC(max_iter=1000, tol=1e-4, probability=True, kernel='linear', decision_function_shape='ovr')
		clf = SGDClassifier(max_iter=1000, tol=1e-4, verbose=1)

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
			#tcg.save_model(os.path.join(save_dir, "model"))
			max_accuracy = cur_accuracy

		print("Training layer: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(layer, iteration, repeat, cur_accuracy, max_accuracy))
		

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
	if(FLAGS.model_type == 'trn'):
		from trn_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'tsm'):
		from tsm_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT


	for layer in range(2):#2,DEPTH_SIZE):
		main(FLAGS.model_type,
			FLAGS.dataset_dir, 
			FLAGS.csv_filename,
			FLAGS.dataset_type,
			FLAGS.dataset_id,
			layer,
			FLAGS.num_classes,
			FLAGS.repeat,
			FLAGS.parse_data
			)
	
	
