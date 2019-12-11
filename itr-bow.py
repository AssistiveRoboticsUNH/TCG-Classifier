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
						e1 = events[i].name#+"_"+str(events[i].occurence) 
						e2 = events[j].name#+"_"+str(events[j].occurence)

						itr = (e1, itr_name, e2)
						itr_seq.append(itr)

				j+=1

		# generate n-grams here
		if(self.ngram > 1):
			ngram_seq = []
			for i in range( len(itr_seq) - (self.ngram - 1) ):
				data = str(itr_seq[i:i+self.ngram])

				ngram_seq.append(data)

			itr_seq = ngram_seq

		return itr_seq
						
	def add_file_to_corpus(self, txt_file):

		# determine if those ITRS are already in TCG, if not add them, if they are increase their count

		for token in self.extract_itr_seq(txt_file):

			if(token not in self.corpus):
				self.corpus[token] = 0
			self.corpus[token] += 1

		self.num_files += 1

	def finalize_corpus(self):

		self.vocabulary = {}

		for k in self.corpus:
			count = self.corpus[k]
			if( count > 0): #and count < self.num_files ):
				self.vocabulary[k] = [ [] for i in range(self.num_classes)]

		self.doc_sizes = [0]*self.num_classes

	def add_vector_counts(self, txt_file, label):

		tokens = self.extract_itr_seq(txt_file)
		counts = Counter(tokens)

		for k in self.vocabulary:
			cnt = counts[k]
			self.vocabulary[k][label].append(cnt)
			self.doc_sizes[label]+= cnt
	

	def finalize_vector_counts(self):

		for k in self.vocabulary:
			self.vocabulary[k] = np.array(self.vocabulary[k])
			for i in range(self.num_classes):
				self.vocabulary[k][i] = np.sum(self.vocabulary[k][i])

			# if all vocab values have the same values then 
			num_file_containing_word = np.sum(self.vocabulary[k] > 0) + 1
			idf = math.log( self.num_classes / float(num_file_containing_word + 1) ) 
			print("IDF:", idf, self.num_classes, num_file_containing_word, )

		print("vocab size: ", len(self.vocabulary.keys()))		


	def tf_idf(self, txt_file):

		label_rank = np.zeros(self.num_classes)

		for token in self.extract_itr_seq(txt_file): 

			if(token in self.vocabulary):

				for label in range(self.num_classes):

					#term_frequency - number of times word occurs in the given document
					
					#print(token, label)
					#print(label, self.vocabulary[token][label], self.doc_sizes[label]+1)
					tf = self.vocabulary[token][label] / float( self.doc_sizes[label]+1 )
					


					#inverse document frequency - how much information the word provides
					num_file_containing_word = np.sum(self.vocabulary[token] > 0) + 1
					idf = math.log( self.num_classes / float(num_file_containing_word) ) + 1

					tfidf = tf * idf

					label_rank[label] += tfidf

		return np.argmax(label_rank)

	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.ngram = 1
		self.bound = 0#2

		self.documents = []
		for i in range(self.num_classes):
			self.documents.append([])

		self.corpus = {}
		self.num_files = 0
		self.vocabulary = []

		self.label_count = [0]*self.num_classes
		self.label_vector = [None]*self.num_classes

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
	
	for ex in train_data:
		tcg.add_file_to_corpus(ex['txt_path'])#, ex['label'])

	print("finalizing corpus")
	tcg.finalize_corpus()
	print("corpus generated")

	for ex in train_data:
		#print("adding:", ex['txt_path'])
		tcg.add_vector_counts(ex['txt_path'], ex['label'])

	print("finalizing vector counts")
	tcg.finalize_vector_counts()
	print("vector counts generated " )

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
