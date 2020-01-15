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

from sklearn.ensemble import VotingClassifier


import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor

class ITR_Extractor_Ensemble:


	def add_file_to_corpus(self, txt_files, label):
		for depth in range(5):
			txt = self.models[depth].parse_txt_file(txt_files[depth])
			self.models[depth].corpus.append(txt)
		self.labels.append(label)

	def add_file_to_eval_corpus(self, txt_files, label, label_name):
		for depth in range(5):
			txt = self.models[depth].parse_txt_file(txt_files[depth])
			self.models[depth].evalcorpus.append(txt)
		self.evallabels.append(label)
		self.label_names[label] = label_name
	
	def eval(self):
		probs = []
		preds = []

		for depth in range(5):
			data = self.models[depth].tfidf.transform(self.models[depth].evalcorpus)

			prob = self.models[depth].clf.decision_function(data)
			pred = self.models[depth].clf.predict(data)

			acc = metrics.accuracy_score(self.evallabels, pred)

			preds.append( acc )
			probs.append( prob )

			print(acc)

		# for every decision point, check to see if the value is above 0 and choose one classe
		#, or choose the other class
		
		#determine ensemble weighting scheme
		weight_scheme = np.array(preds)
		med = np.median(preds)

		weight_scheme[np.argwhere(weight_scheme > med)] = 1.0
		weight_scheme[np.argwhere(weight_scheme < med)] = 0.0
		weight_scheme[np.argwhere(weight_scheme == med)] = 0.5
		
		weight_scheme = np.array([weight_scheme])
		#weight_scheme = [[1,0,0,0,0]]#np.array([weight_scheme])
		print("weight_scheme:", weight_scheme)
		probs *= np.array(weight_scheme).reshape(5,1,1)

		# make confidence prediction
		probs = np.mean(probs, axis=0)
		ensemble_pred = np.argmax(probs, axis = 1)

		return metrics.accuracy_score(self.evallabels, ensemble_pred)
	




	def __init__(self, num_classes, dataset_id, dataset_type, save_name):
		self.num_classes = num_classes

		self.bound = 0

		self.corpus = [[] for i in range(5)]
		self.labels = []

		self.label_names = ['']* self.num_classes

		self.evalcorpus = [[] for i in range(5)]
		self.evallabels = []

		self.models = []
		for depth in range(5):

			save_file = os.path.join(save_name, str(dataset_id), dataset_type)
			filename = save_file.replace('/', '_')+'_'+str(depth)#+".joblib"
			
			self.models.append(ITR_Extractor(num_classes, os.path.join(save_file, filename)))

			
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	tcg = ITR_Extractor_Ensemble(num_classes, dataset_id, dataset_type, save_name)
	
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for depth in range(5):
		for ex in csv_contents:

			ex['txt_path_'+str(depth)] = os.path.join(dataset_dir, "btxt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')

		test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	
	# CLASSIFY 
	print("adding eval data...")
	for ex in test_data:
		txt_files = [ex['txt_path_'+str(d)] for d in range(5)]
		tcg.add_file_to_eval_corpus(txt_files, ex['label'], ex['label_name'])
	print("evaluating model...")
	
	print("ensemble, acc: {:.4f}".format(tcg.eval()))
	

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
