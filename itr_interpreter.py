from sets import Set
import os, sys, math
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

import matplotlib
matplotlib.use('Agg')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier


import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor




def f_importances(coef, names):
	print(type(coef), len(names))
	print(coef)

	imp = coef.toarray()[0]

	imp,names = zip(*sorted(zip(imp,names)))
	plt.barh(range(len(names)), imp, align='center')
	plt.yticks(range(len(names)), names)
	plt.saveimg("test.png")
			
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	#restore model
	depth = 4

	save_file = os.path.join(save_name, str(dataset_id), dataset_type)
	filename = save_file.replace('/', '_')+'_'+str(depth)
	tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))


	coef = tcg.clf.coef_
	names = tcg.tfidf.get_feature_names()

	f_importances(abs(coef[0]), names)

"""
	# determine which features are most influential
	# Depends on how well I can get SVC to work compared to SGD.







	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for depth in range(5):
		for ex in csv_contents:
			ex['txt_path_'+str(depth)] = os.path.join(dataset_dir, "btxt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')


		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
		test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
		
	# TRAIN
	print("adding data...")
	for ex in train_data:
		txt_files = [ex['txt_path_'+str(d)] for d in range(5)]
		tcg.add_file_to_corpus(txt_files, ex['label'])
	print("fitting model...")
	tcg.fit()
		
	# CLASSIFY 
	print("adding eval data...")
	for ex in test_data:
		txt_files = [ex['txt_path_'+str(d)] for d in range(5)]
		tcg.add_file_to_eval_corpus(txt_files, ex['label'], ex['label_name'])
	print("evaluating model...")

	weight_scheme = []
	for depth in range(5):
		acc = tcg.eval_single(depth) 
		print("depth: {:d}, acc: {:.4f}".format(depth, acc))
		weight_scheme.append(acc)

	weight_scheme = np.array(weight_scheme)
	med = np.median(weight_scheme)

	weight_scheme[np.argwhere(weight_scheme > med)] = 1.0
	weight_scheme[np.argwhere(weight_scheme < med)] = 0.0
	weight_scheme[np.argwhere(weight_scheme == med)] = 0.5
	
	weight_scheme = np.array([weight_scheme])

	print("weight_scheme", weight_scheme)
	print("ensemble, acc: {:.4f}".format(tcg.eval(weight_scheme)))
	

	# GEN PYPLOT
	'''
	fig, ax = plt.subplots()
	ax.imshow(class_acc, cmap='hot', interpolation='nearest')
	ax.set_xticks(np.arange(len(label_names)))
	ax.set_yticks(np.arange(len(label_names)))
	ax.set_xticklabels(label_names)
	ax.set_yticklabels(label_names)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		 rotation_mode="anchor")

	for i in range(len(label_names)):
		for j in range(len(label_names)):
			if(class_acc[i, j] < 0.5):
				text = ax.text(j, i, class_acc[i, j],
							   ha="center", va="center", color="w")
			else:
				text = ax.text(j, i, class_acc[i, j],
							   ha="center", va="center", color="k")

	fig.tight_layout()

	#plt.show()
	'''
"""

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')
	#parser.add_argument('dataset_depth', type=int, help='a csv file denoting the files in the dataset')

	parser.add_argument('--save_name', default="", help='what to save the model as')

	FLAGS = parser.parse_args()

	#i=2

	#for i in range(5):


	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.num_classes,
		FLAGS.save_name
		)
