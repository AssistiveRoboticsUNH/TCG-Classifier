from sets import Set
import os, sys, math
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv


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
		

	def add_file_to_corpus(self, txt_file, label):
		
		tokens = self.extract_itr_seq(txt_file)
		counts = Counter(tokens)

		for k in counts.keys():
			if k not in self.vocabulary:
				self.vocabulary[k] = 0
			self.vocabulary[k] += counts[k]

		self.corpus.append(counts)
		self.labels.append(label)

	def finalize(self):

		print("vocab size: ", len(self.vocabulary.keys()))

		for k in self.vocabulary.keys():
			s = "{0}-{1}-{2} ".format(k[0], k[1], k[2])
			print(s)

		self.tfidf = np.zeros((len(self.corpus), len(self.vocabulary)))
		for d, doc in enumerate(self.corpus):
			for k, key in enumerate(self.vocabulary.keys()):
				#tf = 
				#idf =

				self.tfidf[d][k] = doc[k]





	
	def tf_idf(self, txt_file):

		label_rank = np.zeros(self.num_classes)

		for token in self.extract_itr_seq(txt_file): 

			if(token in self.vocabulary):

				for label in range(self.num_classes):

					#term_frequency - number of times word occurs in the given document
					
					#tf = self.vocabulary[token][label] / float( self.doc_sizes[label]+1 )
					tf = self.vocabulary[token][label] / float( self.doc_sizes[label] )
					


					#inverse document frequency - how much information the word provides
					num_file_containing_word = np.sum(self.vocabulary[token] > 0) + 1
					idf = math.log( self.num_classes / float(num_file_containing_word) ) + 1
					#num_file_containing_word = np.sum(self.vocabulary[token] > 0) + 1
					#idf = math.log( (self.num_classes+ 1) / float(num_file_containing_word) )  


					#only get lower accuracy when IDF losses the +1 not when TF losses the +1

					tfidf = tf * idf

					label_rank[label] += tfidf

		return np.argmax(label_rank)

	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.ngram = 1
		self.bound = 0#2

		self.corpus = []
		self.vocabulary = {}
		self.labels = []

		self.tfidf = []


def main(dataset_dir, csv_filename, dataset_type, dataset_id, depth):

	num_classes = 13
	tcg = ITR_Extractor(num_classes)
	
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for ex in csv_contents:
		ex['txt_path'] = os.path.join(dataset_dir, "txt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	
	for ex in train_data[:5]:
		tcg.add_file_to_corpus(ex['txt_path'], ex['label'])

	print("finalizing vector counts")
	tcg.finalize()
	print("vector counts generated " )
	'''
	class_acc = np.zeros((num_classes, num_classes))
	label_names = [""]* 13
	for ex in test_data:
		print(ex['txt_path'])
		pred = tcg.tf_idf(ex['txt_path'])
		label_names[ex['label']] = ex['label_name']
		print(ex['label_name'], pred, ex['label'])
		class_acc[pred, ex['label']] += 1

	
	sum_corr = 0
	for i in range(num_classes):
		print("{:13}".format(label_names[i]),class_acc[i])
		sum_corr += class_acc[i,i]
	print("TOTAL ACC: ", sum_corr/np.sum(class_acc))
	'''
	

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_depth', type=int, help='a csv file denoting the files in the dataset')

	FLAGS = parser.parse_args()

	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.dataset_depth
		)
