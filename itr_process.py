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

import itr_parser
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler

from multiprocessing import Pool

from sklearn.pipeline import Pipeline

def extract_wrapper(ex):

	out = itr_parser.extract_itr_seq_into_counts(ex['b_path'])

	print(out.shape)
	out = out.reshape(out.shape[0], -1).astype(np.uint8)
	
	#out = scipy.sparse.csr_matrix(out)
	np.save(ex['sp_path'], out)

	return ex['sp_path']

	

def parse_files(csv_contents, num_procs=1, empty_locs=[]):
	#file_list= file_list[:10]

	t_s = time.time()

	pool = Pool(num_procs)
	for i, c in enumerate(pool.imap_unordered( extract_wrapper, csv_contents, chunksize=10 )):
		if(i % 1000 == 0):
			print("elapsed time {0}: {1}".format(i,  time.time()-t_s))
	pool.close()
	pool.join()




	'''

	mylist = []
	for g in corpus:
		mylist.append(g)
	corpus = mylist

	corpus = np.array(corpus)
	corpus = corpus.reshape(corpus.shape[0], -1)

	if(len(empty_locs) == 0):
		empty_locs = np.where(corpus.any(axis=0))[0]

	corpus = corpus[:, empty_locs]
	corpus = scipy.sparse.csr_matrix(corpus)

	print("parsing data: {0}s".format(time.time() - t_s))

	return  corpus, empty_locs

	'''

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

	b_dir_name  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	sp_dir_name = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	if(not os.path.exists(sp_dir_name)):
		os.makedirs(sp_dir_name)

	print("Organizing csv_contents")
	for ex in csv_contents:
		ex['b_path'] = os.path.join(b_dir_name, '{0}_{1}.b'.format(ex['example_id'], layer))
		ex['sp_path'] = os.path.join(sp_dir_name, '{0}_{1}.npy'.format(ex['example_id'], layer))

	dataset = [ex for ex in csv_contents if ex['label'] < num_classes]
	print("dataset_length:", len(dataset))
	'''
	hashvect = CountVectorizer(token_pattern=r"\d+\w\d+")#HashingVectorizer(n_features=2**17, token_pattern=r"\d+\w\d+")
	tfidf = TfidfTransformer(sublinear_tf=True)
	scale = StandardScaler(with_mean=False)


	pipe = Pipeline([
		('tfidf', tfidf),
		('scale', scale),
	])
	'''

	# TRAIN
	parse_files(dataset, num_procs=num_procs)
	




	'''
	print("fit train data...")
	t_s = time.time()
	data_in = pipe.fit_transform(data_in)
	print("fit data: {0}".format(time.time() - t_s))

	data_label = [ex['label'] for ex in train_data]

	print("data_in shape:", data_in.shape)
	scipy.sparse.save_npz(train_filename, data_in)
	np.save(train_label_filename, data_label)
	print('')

	# EVALUATE
	in_files = [ex[path] for ex in test_data]

	print("adding test data...{0}".format(len(test_data)))
	data_in, _ = parse_files(in_files, num_procs=num_procs, empty_locs=empty_locs)

	print("fit eval data...")
	t_s = time.time()
	eval_in = pipe.transform(data_in)
	print("fit data: {0}".format(time.time() - t_s))


	eval_label = [ex['label'] for ex in test_data]

	print("eval_in shape:", eval_in.shape)
	scipy.sparse.save_npz(test_filename, eval_in)
	np.save(test_label_filename, eval_label)
	print('--------')
	'''




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

	
