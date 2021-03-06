


import os, sys, math, time

if (sys.version[0] == '2'):
	from sets import Set
	import cPickle as pickle
	from sklearn.svm import SVC
#else:
	#from thundersvm import SVC



import numpy as np

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

sys.path.append("../IAD-Parser/TCG/")
from parser_utils import read_sparse_matrix


from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from joblib import dump, load

import matplotlib
import matplotlib.pyplot as plt

from multiprocessing import Pool



#import svm_gpu.svm as svm_gpu 

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

def read_file(txt_file):
	
	sparse_matrix = read_sparse_matrix(txt_file)
	
	events = []
	for i, feature in enumerate(sparse_matrix):
		for j, pair in enumerate(feature):
			events.append(AtomicEvent(i, j, start=pair[0], end=pair[1]))
	
	return events

def get_itrs(e1, e2):
	return e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)
	
def extract_itr_seq(txt_file):

	# get events from file
	t_s = time.time()
	events = sorted(read_file(txt_file)) 
	print("sort: ", time.time() - t_s)

	# get a list of all of the ITRs in the txt_file
	itr_seq = []
	t_s = time.time()
	for i in range(len(events)):

		j = i+1
		while(j < len(events) and events[j].name != events[i].name):

			itr_name = get_itrs(events[i], events[j])

			e1 = events[i].name
			e2 = events[j].name

			itr = (e1, itr_name, e2)
			itr_seq.append(itr)

			j+=1

	print("itrs: ", time.time() - t_s)
	
	return itr_seq

def parse_txt_file(txt_file):
	t_s = time.time()
	txt = ''
	for itr in extract_itr_seq(txt_file):
		s = "{0}-{1}-{2} ".format(itr[0], itr[1], itr[2])
		txt += s
	print("done parse: ", time.time() -t_s)
	return txt


class ITR_Extractor:

	

	def add_file_to_corpus(self, txt_file, label):
		txt = parse_txt_file(txt_file)
		self.corpus.append(txt)
		self.labels.append(label)

	def add_files_to_corpus(self, file_list, label_list):
		self.corpus = self.pool.map(parse_txt_file, file_list)
		self.labels += label_list

	def add_files_to_eval_corpus(self, file_list, label_list):
		self.evalcorpus = self.pool.map(parse_txt_file, file_list)
		self.evallabels += label_list

	def add_file_to_eval_corpus(self, txt_file, label, label_name):


		txt = parse_txt_file(txt_file)
		self.evalcorpus.append(txt)
		self.evallabels.append(label)

		self.label_names[label] = label_name

	def pca(self, x):
		
		return pca.fit_transform(x)


		#principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])


	def fit(self):


		t_s = time.time()
		train_mat = self.tfidf.fit_transform(self.corpus)
		print("TF-IDF: ", time.time()-t_s)
		print(train_mat.shape)

		#t_s = time.time()
		train_mat = self.scaler.fit_transform(train_mat)
		#train_mat = self.svd.fit_transform(train_mat)
		#print("TruncatedSVD: ", time.time()-t_s)
		#print(train_mat.shape)

		t_s = time.time()
		self.clf.fit(train_mat, np.array(self.labels))
		print("Train Time: ", time.time()-t_s)
	
	def pred(self, txt_file):
		#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
		txt = self.parse_txt_file(txt_file)
		data = self.tfidf.transform([txt])
		data = self.scaler.transform(data)
		#data = self.svd.transform(data)
		return self.clf.predict(data)

	def eval(self):
		data = self.tfidf.transform(self.evalcorpus)
		data = self.scaler.transform(data)
		#data = self.svd.transform(data)
		pred = self.clf.predict(data)
		return metrics.accuracy_score(self.evallabels, pred)
	'''
	def save_model(self, name='model'):
		# save model
		dump(self.clf, name+'.joblib') 

		#save vectorizer
		with open(name+'.pk', 'wb') as file_loc:
			pickle.dump(self.tfidf, file_loc)

	def load_model(self, name='model'):
		#load model
		self.clf = load(name+'.joblib') 

		#load vectorizer
		self.tfidf = pickle.load(open(name+'.pk', "rb"))
	'''
	def __init__(self, num_classes, save_name="", num_procs=1):
		self.num_classes = num_classes

		self.bound = 0

		self.corpus = []
		self.labels = []

		self.label_names = ['']* self.num_classes

		self.evalcorpus = []
		self.evallabels = []

		self.num_procs = num_procs
		self.pool = Pool(num_procs)

		self.tfidf = TfidfVectorizer(token_pattern=r"\b\w+-\w+-\w+\b", sublinear_tf=True)
		self.scaler = StandardScaler(with_mean=False)
		#self.svd = TruncatedSVD(n_components=10000)
		self.clf = SVC(max_iter=1000, tol=1e-4, probability=True, kernel='linear', decision_function_shape='ovr')
		#self.clf = svm_gpu.SVM(max_iter=1000, tol=1e-4, probability=True, kernel='linear', classification_strategy='ovr')
		
		'''
		if(save_name != ""):
			print("load model:", save_name)
			self.load_model(save_name)
		'''

def main(dataset_dir, csv_filename, dataset_type, dataset_id, depth, num_classes, save_name=""):

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
	print("adding data...")
	for ex in train_data:
		tcg.add_file_to_corpus(ex['txt_path'], ex['label'])
	print("fitting model...")
	tcg.fit()
	
	# CLASSIFY 
	print("adding eval data...")
	for ex in test_data:
		tcg.add_file_to_eval_corpus(ex['txt_path'], ex['label'], ex['label_name'])
	print("evaluating model...")
	cur_accuracy = tcg.eval()

	print(cur_accuracy)

	if(save_name != ""):
		tcg.save_model(save_name+".joblib")
	

	# GEN PYPLOT
	"""
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

	parser.add_argument('--save_name', default="", help='what to save the model as')

	FLAGS = parser.parse_args()

	for depth in range(5):
		main(FLAGS.dataset_dir, 
			FLAGS.csv_filename,
			FLAGS.dataset_type,
			FLAGS.dataset_id,
			i,
			FLAGS.num_classes,
			FLAGS.save_name+'_'+str(depth)
			)
