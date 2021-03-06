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


class ITR_Extractor:

	class AtomicEvent():
		def __init__(self, name, occurence, start=-1, end=-1):
			self.name = name
			self.occurence = occurence
			self.start = start
			self.end = end

		def __lt__(self, other):
			if( self.start < other.start ):
				return True
			return self.start == other.start and self.end < other.end

		def get_itr(self, other):
			return self.get_itr_from_time(self.start, self.end, other.start, other.end)

		def get_itr_from_time(self, a1, a2, b1, b2):

			#before
			if (a2 < b1):
				return 'b'

			#meets
			if (a2 == b1):
				return 'm'

			#overlaps
			if (a1 < b1 and a2 < b2 and b1 < a2):
				return 'o'

			#during
			if (a1 < b1 and b2 < a2):
				return 'd'

			#finishes
			if (b1 < a1 and a2 == b2):
				return 'f'

			#starts
			if (a1 == b1 and a2 < b2):
				return 's'

			#equals
			if (a1 == b1 and a2 == b2):
				return 'eq'

			#startedBy
			if (a1 == b1 and b2 < a2):
				return 'si'

			#contains
			if (b1 < a1 and a2 < b2):
				return 'di'

			#finishedBy
			if (a1 < b1 and a2 == b2):
				return 'fi'

			#overlappedBy
			if (b1 < a1 and b2 < a2 and a1 < b2):
				return 'oi'

			#metBy
			if (b2 == a1):
				return 'mi'

			#after
			if (b2 < a1):
				return 'bi'


	
	def parse_txt_file(self, txt_file):
		f = np.load(txt_file)
		max_v, mean_v, min_v = f["max"], f["mean"], f["min"]

		return np.concatenate((max_v, mean_v, min_v))

	def add_file_to_corpus(self, txt_file, label):
		txt = self.parse_txt_file(txt_file)
		self.corpus.append(txt)
		self.labels.append(label)

	def add_file_to_eval_corpus(self, txt_file, label, label_name):
		txt = self.parse_txt_file(txt_file)
		self.evalcorpus.append(txt)
		self.evallabels.append(label)

		self.label_names[label] = label_name

	def fit(self):
		#train_mat = self.tfidf.fit_transform(self.corpus)

		#print(train_mat.shape)
		#self.clf = MultinomialNB().fit(train_mat, np.array(self.labels))
		#self.clf = svm.SVC().fit(train_mat, np.array(self.labels))
		#self.clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None).fit(train_mat, np.array(self.labels))
		train_mat = np.array(self.corpus)
		self.clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-4, random_state=42,max_iter=500, tol=None, verbose=0).fit(train_mat, np.array(self.labels))

	def pred(self, txt_file):
		#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
		data = self.parse_txt_file(txt_file)
		#data = self.tfidf.transform([txt])
		return self.clf.predict([data])

	def eval(self):
		data = np.array(self.evalcorpus)
		pred = self.clf.predict(data)

		print(metrics.classification_report(self.evallabels, pred, target_names=self.label_names))
		print(metrics.accuracy_score(self.evallabels, pred))



	def __init__(self, num_classes):
		self.num_classes = num_classes

		self.bound = 0

		self.corpus = []
		self.labels = []

		self.label_names = ['']* self.num_classes

		self.evalcorpus = []
		self.evallabels = []

		self.tfidf = TfidfVectorizer(token_pattern=r"\b\w+-\w+-\w+\b", sublinear_tf=True)
		
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, depth, num_classes):

	tcg = ITR_Extractor(num_classes)
	
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for ex in csv_contents:
		ex['txt_path'] = os.path.join(dataset_dir, "txt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt.npz')

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	
	# TRAIN
	for ex in train_data:
		tcg.add_file_to_corpus(ex['txt_path'], ex['label'])
	tcg.fit()
	
	# CLASSIFY 
	for ex in test_data:
		tcg.add_file_to_eval_corpus(ex['txt_path'], ex['label'], ex['label_name'])
	tcg.eval()

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

	FLAGS = parser.parse_args()

	#for i in range(5):
	i=2
	print("depth: ", i)


	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		i,
		FLAGS.num_classes
		)
