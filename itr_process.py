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

from itr_parser import extract_itr_seq
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

from sklearn.pipeline import Pipeline

def extract_wrapper(file):
	return extract_itr_seq(file)

def parse_files(file_list, num_procs=1):
	pool = Pool(num_procs)
	return pool.map(extract_wrapper, file_list)

def get_filenames(dataset_dir, model_type, dataset_type, dataset_id, layer):
	file_path = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	
	train_filename = os.path.join(file_path, 'train_{0}_{1}.npz'.format(dataset_id, layer))
	test_filename  = os.path.join(file_path, 'test_{0}_{1}.npz'.format(dataset_id, layer))
	train_label_filename = os.path.join(file_path, 'train_label_{0}_{1}.npy'.format(dataset_id, layer))
	test_label_filename  = os.path.join(file_path, 'test_label_{0}_{1}.npy'.format(dataset_id, layer))
	
	return train_filename, test_filename, train_label_filename, test_label_filename

def retrieve_data(dataset_dir, model_type, dataset_type, dataset_id, layer):
	print("Retrieving file data!")
	train_filename, test_filename, train_label_filename, test_label_filename = get_filenames(dataset_dir, model_type, dataset_type, dataset_id, layer)

	data_in = scipy.sparse.load_npz(train_filename)
	data_label = np.load(train_label_filename)

	eval_in = scipy.sparse.load_npz(test_filename)
	eval_label = np.load(test_label_filename)

	return data_in, data_label, eval_in, eval_label


def process_data(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs):
	print("Generating new files!")
	#tcg = ITR_Extractor(num_classes, num_procs)		
		
	#open files
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	print("Organizing csv_contents")
	path = 'b_path_{0}'.format(layer)
	for ex in csv_contents:
		ex[path] = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.b'.format(ex['example_id'], layer))

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
	train_data = [ex for ex in train_data if ex['label'] < num_classes]

	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	test_data = [ex for ex in test_data if ex['label'] < num_classes]


	print("Generating file names")
	train_filename, test_filename, train_label_filename, test_label_filename = get_filenames(dataset_dir, model_type, dataset_type, dataset_id, layer)
	

	hashvect = HashingVectorizer(n_features=2**17, token_pattern=r"\b\d+\w\d+\b")
	tfidf = TfidfTransformer(sublinear_tf=True)
	scale = StandardScaler(with_mean=False)


	pipe = Pipeline([
		#('count', CountVectorizer(token_pattern=r"\b\w+-\w+-\w+\b")),
		('hash', hashvect),
		#('hash', HashingVectorizer(n_features=2**18, token_pattern=r"\b\w+-\w+-\w+\b")),
		('tfidf', tfidf),#, token_pattern=r"\b\w+-\w+-\w+\b")),
		#('tfidf', TfidfVectorizer(sublinear_tf=True, token_pattern=r"\b\d+\w\d+\b")),#, token_pattern=r"\b\w+-\w+-\w+\b")),
		('scale', scale),
	])

	# TRAIN
	in_files = [ex[path] for ex in train_data]

	print("adding train data...{0}".format(len(train_data)))
	t_s = time.time()
	corpus = parse_files(in_files, num_procs=num_procs)
	print("data added - time: {0}".format(time.time() - t_s))


	print("fit train data...")
	t_s = time.time()
	data_in = hashvect.fit_transform(corpus)
	print("hashvect fit - time: {0}".format(time.time() - t_s))
	print("data-in shape: ", data_in.shape)
	t_s = time.time()
	data_in = tfidf.fit_transform(data_in)
	print("tfidf fit - time: {0}".format(time.time() - t_s))
	t_s = time.time()
	data_in = scale.fit_transform(data_in)
	print("scale fit - time: {0}".format(time.time() - t_s))






	#data_in = pipe.fit_transform(corpus)
	print("data shape: ",  data_in.shape)
	print("data fit - time: {0}".format(time.time() - t_s))

	data_label = [ex['label'] for ex in train_data]

	scipy.sparse.save_npz(train_filename, data_in)
	np.save(train_label_filename, data_label)


	# EVALUATE
	in_files = [ex[path] for ex in test_data]

	print("adding eval data...{0}".format(len(test_data)))
	t_s = time.time()
	corpus = parse_files(in_files, num_procs=num_procs)
	print("eval data added - time: {0}".format(time.time() - t_s))

	print("fit eval data...")
	t_s = time.time()
	eval_in = pipe.transform(corpus)
	print("data shape: ",  eval_in.shape)
	print("eval data fit - time: {0}".format(time.time() - t_s))

	eval_label = [ex['label'] for ex in test_data]

	scipy.sparse.save_npz(test_filename, eval_in)
	np.save(test_label_filename, eval_label)





if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'rn50', 'trn', 'tsm'])

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


	for layer in range(1,DEPTH_SIZE):
		process_data(FLAGS.dataset_dir, 
			FLAGS.model_type, 
			FLAGS.dataset_type, 
			FLAGS.dataset_id, 
			layer, 
			FLAGS.csv_filename, 
			FLAGS.num_classes,
			FLAGS.num_procs)

	
