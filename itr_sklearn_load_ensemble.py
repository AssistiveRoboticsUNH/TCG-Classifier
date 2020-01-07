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

	def fit(self):
		for depth in range(5):
			train_mat = self.models[depth].tfidf.fit_transform(self.models[depth].corpus)
			print(depth, train_mat.shape)
			#self.models[depth].fit(train_mat, np.array(self.labels))



	def pred(self, txt_files):
		#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

		confidence_values = []

		#depth=4
		for depth in range(5):
			txt = txt_files[depth]#self.parse_txt_file(txt_files[depth])
			data = self.models[depth].tfidf.transform([txt])
			confidence_values.append( self.models[depth].clf.predict_proba(data) )

		#print("confidence_values:", confidence_values)
		confidence_values *= np.array([[1.0, 1.0, 0.5, 0.0, 0.0]]).reshape(5,1,1)
		#confidence_values *= np.array([[0.0, 0.0, 0.5, 1.0, 1.0]]).reshape(5,1,1)
		

		#print("confid_shape: ", np.array(confidence_values).shape)
		confidence_values = np.mean(confidence_values, axis=(0,1))
		#print("confid_shape: ", np.array(confidence_values).shape)

		return np.argmax(confidence_values)

	def eval_single(self, depth):
		
		data = self.models[depth].tfidf.transform([self.models[depth].evalcorpus])
		pred = self.models[depth].clf.predict(data) 

		return metrics.accuracy_score(self.evallabels, pred)


	def eval(self):

		pred = []

		for i in range(len(self.models[0].evalcorpus)):

			txt_files = [self.models[depth].evalcorpus[i] for depth in range(5)]
			pred.append( self.pred(txt_files) )

		print("pred:", np.array(pred).shape)
		print("labels:", np.array(self.evallabels).shape)

		#print(metrics.classification_report(self.evallabels, pred, target_names=self.label_names))
		#print(metrics.accuracy_score(self.evallabels, pred))
		return metrics.accuracy_score(self.evallabels, pred)





	def __init__(self, num_classes, save_name):
		self.num_classes = num_classes

		self.bound = 0

		self.corpus = [[] for i in range(5)]
		self.labels = []

		self.label_names = ['']* self.num_classes

		self.evalcorpus = [[] for i in range(5)]
		self.evallabels = []

		self.models = []
		for depth in range(5):
			self.models.append(ITR_Extractor(num_classes, save_name+'_'+str(depth)+".joblib"))

			
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	tcg = ITR_Extractor_Ensemble(num_classes, save_name)
	
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for depth in range(5):
		for ex in csv_contents:
			ex['txt_path_'+str(depth)] = os.path.join(dataset_dir, "gtxt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')


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

	for depth in range(5):
		print("depth: {:d}, acc: {:.4f}".format(depth, tcg.eval_single(depth)))

	
	print("ensemble, acc: {:.4f}".format(depth, tcg.eval()))
	

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
