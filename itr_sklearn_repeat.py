import os, sys, math, time
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor

def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1):

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
		test_data = [ex for ex in train_data if ex['label'] < num_classes]

		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)


		train_filename = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'train_{0}_{1}.npz'.format(ex['example_id'], layer))
		test_filename  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'test{0}_{1}.npz'.format(ex['example_id'], layer))

		#parse_data = not os.path.exists(train_filename)

		if(parse_data):
			# TRAIN
			in_files = [ex[path] for ex in train_data]
			in_labels = [ex['label'] for ex in train_data]

			print("adding train data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))


			data_in = np.array(tcg.tfidf.fit_transform(tcg.corpus).toarray())
			data_label = np.array(tcg.labels)

			np.savez_compressed(train_filename, data=data_in, label=data_label)


			in_files = [ex[path] for ex in test_data]
			in_labels = [ex['label'] for ex in test_data]
			print("adding eval data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_eval_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))

			eval_in = np.array(tcg.tfidf.transform(tcg.evalcorpus).toarray())
			eval_label = np.array(tcg.evallabels)

			np.savez_compressed(test_filename, data=eval_in, label=eval_label)

		else:
			f = np.load(train_filename)
			data_in, data_label = f["data"], f["label"]
			f = np.load(test_filename)
			eval_in, eval_label = f["data"], f["label"]


		# TRAIN
		print("fitting model...")
		t_s = time.time()
		tcg.fit()
		print("elapsed:", time.time()-t_s)
		
		print("evaluating model...")
		t_s = time.time()
		cur_accuracy = tcg.eval()
		print("elapsed:", time.time()-t_s)


		# if model accuracy is good then replace the old model with new save data
		if(cur_accuracy > max_accuracy):
			tcg.save_model(os.path.join(save_dir, "model"))
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


	FLAGS = parser.parse_args()

	if(FLAGS.model_type == 'i3d'):
		from gi3d_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'trn'):
		from trn_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'tsm'):
		from tsm_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT


	for layer in range(2):#DEPTH_SIZE):
		main(FLAGS.model_type,
			FLAGS.dataset_dir, 
			FLAGS.csv_filename,
			FLAGS.dataset_type,
			FLAGS.dataset_id,
			layer,
			FLAGS.num_classes,
			FLAGS.repeat
			)
	
	
