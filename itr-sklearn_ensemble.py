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





	def read_file(self, txt_file):
		
		events = {}
		for line in list(open(txt_file, 'r')):
			line = line.split()

			event_tokens = line[0].split('_')
			time = float(line[1])
			
			event_name = event_tokens[0]
			event_occur = int(event_tokens[1])
			event_bound = event_tokens[2]

			event_id = event_name+'_'+str(event_occur)
			if (event_id not in events):
				events[event_id] = self.AtomicEvent(event_name, event_occur)

			if(event_bound == 's'):
				events[event_id].start = time
			else:
				events[event_id].end = time

		return events.values()

	def all_itrs(self, e1, e2, bound):

		itrs = Set()
		for i in range(-bound, bound):

			itr_name = e1.get_itr_from_time(e1.start, e1.end+i, e2.start, e2.end)
			itrs.add(itr_name)
		itr_name = e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)
		itrs.add(itr_name)

		return itrs


	
	def extract_itr_seq(self, txt_file):

		# get events from file
		events = sorted(self.read_file(txt_file)) 

		# get a list of all of the ITRs in the txt_file
		itr_seq = []

		for i in range(len(events)):

			j = i+1
			while(j < len(events) and events[j].name != events[i].name):

				for itr_name in self.all_itrs(events[i], events[j], self.bound):

					if('i' not in itr_name):
						e1 = events[i].name
						e2 = events[j].name

						itr = (e1, itr_name, e2)
						itr_seq.append(itr)
				j+=1
		
		return itr_seq
	
	def parse_txt_file(self, txt_file):
		txt = ''
		for itr in self.extract_itr_seq(txt_file):
			s = "{0}-{1}-{2} ".format(itr[0], itr[1], itr[2])
			txt += s
		return txt

	def add_file_to_corpus(self, txt_files, label):
		for depth in range(5):
			txt = self.parse_txt_file(txt_files[depth])
			self.corpus[depth].append(txt)
		self.labels.append(label)

	def add_file_to_eval_corpus(self, txt_files, label, label_name):
		for depth in range(5):
			txt = self.parse_txt_file(txt_files[depth])
			self.evalcorpus[depth].append(txt)
		self.evallabels.append(label)
		self.label_names[label] = label_name

	def fit(self):
		for depth in range(5):
			train_mat = self.tfidf[depth].fit_transform(self.corpus[depth])
			print(depth, train_mat.shape)
			self.models[depth].fit(train_mat, np.array(self.labels))

	def pred(self, txt_files):
		#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

		confidence_values = []

		#depth=4
		for depth in range(5):
			txt = txt_files[depth]#self.parse_txt_file(txt_files[depth])
			data = self.tfidf[depth].transform([txt])
			confidence_values.append( self.models[depth].predict_proba(data) )

		#print("confidence_values:", confidence_values)
		confidence_values *= np.array([[0.0, 0.0, 0.5, 1.0, 1.0]]).reshape(5,1,1)
		#confidence_values *= np.array([[0.0, 0.0, 0.5, 1.0, 1.0]]).reshape(5,1,1)
		

		#print("confid_shape: ", np.array(confidence_values).shape)
		confidence_values = np.mean(confidence_values, axis=(0,1))
		#print("confid_shape: ", np.array(confidence_values).shape)

		return np.argmax(confidence_values)

	def eval(self):

		pred = []

		for i in range(len(self.evalcorpus[0])):

			txt_files = [self.evalcorpus[depth][i] for depth in range(5)]
			pred.append( self.pred(txt_files) )

		print("pred:", np.array(pred).shape)
		print("labels:", np.array(self.evallabels).shape)

		print(metrics.classification_report(self.evallabels, pred, target_names=self.label_names))
		print(metrics.accuracy_score(self.evallabels, pred))





	def __init__(self, num_classes):
		self.num_classes = num_classes

		self.bound = 0

		self.corpus = [[] for i in range(5)]
		self.labels = []

		self.label_names = ['']* self.num_classes

		self.evalcorpus = [[] for i in range(5)]
		self.evallabels = []

		self.tfidf = []
		self.models = []
		for i in range(5):
			self.tfidf.append(TfidfVectorizer(token_pattern=r"\b\w+-\w+-\w+\b", sublinear_tf=True))
			self.models.append(SGDClassifier(loss='modified_huber', penalty='l2',alpha=1e-4, max_iter=1000, tol=1e-4, verbose=0, n_jobs=-1))
		
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes):

	tcg = ITR_Extractor(num_classes)
	
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

	#i=2

	#for i in range(5):


	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.num_classes
		)
