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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MinMaxScaler

from multiprocessing import Pool, Process

from sklearn.pipeline import Pipeline

if (sys.version[0] == '2'):
	import cPickle as pickle


#### EVENT TO ITR #####

def extract_wrapper(ex):
	''' extract the ITRs from a single event binary file. The output is saved to the
	sp_path directory. '''

	out = itr_parser.extract_itr_seq_into_counts(ex['b_path'])
	out = out.reshape(-1).astype(np.uint8)

	np.save(ex['sp_path'], out)

	return ex['sp_path']

def convert_event_to_itr(csv_contents, num_procs=1, empty_locs=[]):
	''' convert a binary event file to a list of ITRs. This fnuction is done
	vie multiple concurrent process calls to "extract_wrapper" '''

	t_s = time.time()
	pool = Pool(num_procs)
	for i, c in enumerate(pool.imap_unordered( extract_wrapper, csv_contents, chunksize=10 )):
		if(i % 1000 == 0):
			print("elapsed time {0}: {1}".format(i,  time.time()-t_s))
	pool.close()
	pool.join()

#### PRE-PROCESS ITR #####

def tfidf_and_scale(ex_list):
	''' extract the ITRs from a single event binary file. The output is saved to the
	sp_path directory. '''
	print("len(ex_list):", len(ex_list))


	tfidf = pickle.load(open("tfidf"+'.pk', "rb"))
	scaler = pickle.load(open("scaler"+'.pk', "rb"))

	# open ex as sparse format
	for ex in ex_list:
		print(ex["example_id"])


		data = np.load(ex['sp_path'])

		idx = np.nonzero(data)[0]
		value = data[idx]

		data = zip(idx, value)

		# Apply pre-processing to sparse data
		data = tfidf.transform(data)

		# format data as dense
		unzipped_data = np.array(zip(*(data[0])))		
		data = np.zeros(128*128*7)
		data[unzipped_data[0].astype(np.int32)] = unzipped_data[1]
		data = data.reshape(1, -1)

		# Apply pre-processing to dense data
		data = scaler.transform(data)

		data = data.reshape(-1)

		#save data as a sparse matrix for efficiency? 
		data = scipy.sparse.coo_matrix(np.array(data))
		scipy.sparse.save_npz(ex['pp_path'], data)

	#return ex['pp_path']

def pre_process_itr(csv_contents, num_procs=1, empty_locs=[]):
	''' convert a binary event file to a list of ITRs. This fnuction is done
	vie multiple concurrent process calls to "extract_wrapper" '''

	t_s = time.time()

	chunk_size = len(csv_contents)/float(num_procs)
	chunk_size = int(math.ceil(chunk_size))

	print("chunk_size:", chunk_size)

	procs = []
	for i in range(num_procs):
		chunk = csv_contents[i*chunk_size:i*chunk_size+chunk_size]
		p = Process(target=tfidf_and_scale, args=(chunk,))
		p.start()

	for i in range(num_procs):
		p.join()
	

	'''
	pool = Pool(num_procs)
	for i, c in enumerate(pool.imap_unordered( tfidf_and_scale, csv_contents, chunksize=10 )):
		if(i % 1000 == 0):
			print("elapsed time {0}: {1}".format(i,  time.time()-t_s))
	pool.close()
	pool.join()
	'''

#### FILE I/O ####

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


def main(dataset_dir, model_type, dataset_type, dataset_id, layer, csv_filename, num_classes, num_procs):
	print("Generating new files!")
		
	#open files
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	b_dir_name  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	sp_dir_name = os.path.join(dataset_dir, 'sp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	pp_dir_name = os.path.join(dataset_dir, 'pp_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
	
	if(not os.path.exists(sp_dir_name)):
		os.makedirs(sp_dir_name)
	if(not os.path.exists(pp_dir_name)):
		os.makedirs(pp_dir_name)

	print("Organizing csv_contents")
	for ex in csv_contents:
		ex['b_path'] = os.path.join(b_dir_name, '{0}_{1}.b'.format(ex['example_id'], layer))
		ex['sp_path'] = os.path.join(sp_dir_name, '{0}_{1}.npy'.format(ex['example_id'], layer))
		ex['pp_path'] = os.path.join(pp_dir_name, '{0}_{1}.npy'.format(ex['example_id'], layer))

	dataset = [ex for ex in csv_contents if ex['label'] < num_classes]
	print("dataset_length:", len(dataset))


	dataset = dataset[:41]

	# CONVERT BINARY EVENTS TO ITRS
	#convert_event_to_itr(dataset, num_procs=num_procs)

	# PRE-PROCESS ITRS
	pre_process_itr(dataset, num_procs=num_procs)
	





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

	layer = DEPTH_SIZE-1

	main(FLAGS.dataset_dir, 
			FLAGS.model_type, 
			FLAGS.dataset_type, 
			FLAGS.dataset_id, 
			layer, 
			FLAGS.csv_filename, 
			FLAGS.num_classes,
			FLAGS.num_procs)
	'''
	for layer in range(DEPTH_SIZE):
		main(FLAGS.dataset_dir, 
			FLAGS.model_type, 
			FLAGS.dataset_type, 
			FLAGS.dataset_id, 
			layer, 
			FLAGS.csv_filename, 
			FLAGS.num_classes,
			FLAGS.num_procs)
	'''
	
