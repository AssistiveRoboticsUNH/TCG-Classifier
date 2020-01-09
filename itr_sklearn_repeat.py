from sets import Set
import os, sys, math
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


from joblib import dump, load


from itr_sklearn import ITR_Extractor

def main(dataset_dir, csv_filename, dataset_type, dataset_id, depth, num_classes, save_name="", repeat=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(depth, iteration, repeat))
	

		tcg = ITR_Extractor(num_classes)
		
		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		for ex in csv_contents:
			ex['txt_path'] = os.path.join(dataset_dir, "gtxt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
		test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		
		# TRAIN
		#print("adding data...")
		for ex in train_data:
			tcg.add_file_to_corpus(ex['txt_path'], ex['label'])
		print("fitting model...")
		tcg.fit()
		
		# CLASSIFY 
		#print("adding eval data...")
		for ex in test_data:
			tcg.add_file_to_eval_corpus(ex['txt_path'], ex['label'], ex['label_name'])
		print("evaluating model...")
		cur_accuracy = tcg.eval()



		if(cur_accuracy > max_accuracy and save_name != ""):

			save_file = os.path.join(save_name, str(dataset_id), dataset_type)
			print(save_file)
			filename = save_file.replace('/', '_')+'_'+str(depth)#+".joblib"
			if (not os.path.exists(save_file)):
				os.makedirs(save_file)

			print(os.path.join(save_file, filename))

			tcg.save_model(os.path.join(save_file, filename))
			max_accuracy = cur_accuracy

		print("Training depth: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(depth, iteration, repeat, cur_accuracy, max_accuracy))
	

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')

	parser.add_argument('--save_name', default="", help='what to save the model as')
	parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat training the model')
	#parser.add_argument('dataset_depth', type=int, help='a csv file denoting the files in the dataset')

	FLAGS = parser.parse_args()

	#i=2

	for dataset_type in ['frames']:#, 'flow', 'both']:
		for dataset_id in [3]:
			#depth = 4
			for depth in range(5):
				print("dataset_type: ", dataset_type)
				print("dataset_id: ", dataset_id)
				print("depth: ", depth)


				main(FLAGS.dataset_dir, 
					FLAGS.csv_filename,
					dataset_type,
					dataset_id,
					depth,
					FLAGS.num_classes,
					FLAGS.save_name,
					FLAGS.repeat
					)
