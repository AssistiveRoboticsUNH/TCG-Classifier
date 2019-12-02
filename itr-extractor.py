from sets import Set
import os, sys
import numpy as np
# I need to sort all of the indiividual actions by feature. Then
# I can get the ITR with the next feature for each each row. I 
# do not need to figure out the relationship between A0 and C7. Just the most adjacent values of A and C.
# progress from left to right means I can ignore reverses operations?

# need to figure this out ASAP!

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
			time = line[1]
			
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

	
	def extract_itr_set(self, txt_file):

		# get events from file
		events = sorted(self.read_file(txt_file)) 

		# get a list of all of the ITRs in the txt_file
		itr_set = Set()

		for i in range(len(events)):

			j = i+1
			while(j < len(events) and events[j].name != events[i].name):
				itr_name = events[i].get_itr( events[j] )

				if('i' not in itr_name):
					e1 = events[i].name#+"_"+str(events[i].occurence) 
					e2 = events[j].name#+"_"+str(events[j].occurence)

					itr = (e1, itr_name, e2)
					itr_set.add(itr)

				j+=1

		return itr_set

	def extract_itr_seq(self, txt_file):

		# get events from file
		events = sorted(self.read_file(txt_file)) 

		# get a list of all of the ITRs in the txt_file
		itr_seq = []

		for i in range(len(events)):

			j = i+1
			while(j < len(events) and events[j].name != events[i].name):
				itr_name = events[i].get_itr( events[j] )

				if('i' not in itr_name):
					e1 = events[i].name#+"_"+str(events[i].occurence) 
					e2 = events[j].name#+"_"+str(events[j].occurence)

					itr = (e1, itr_name, e2)
					itr_set.append(itr)

				j+=1

		return itr_set

	def add_itr_set(self, txt_file, label):

		itr_set = self.extract_itr_set(txt_file)
		
		# determine if those ITRS are already in TCG, if not add them, if they are increase their count
		for itr in itr_set:
			if(itr not in self.tcgs[label]):
				self.tcgs[label][itr] = 0
			self.tcgs[label][itr] += 1
						
	def add_itr_seq(self, txt_file, label):

		itr_seq = self.extract_itr_seq(txt_file)

		# determine if those ITRS are already in TCG, if not add them, if they are increase their count
		for i in range(len(itr_seq)-1):
			itr_cur = itr_seq[i]
			itr_next = itr_seq[i+1]

			if(itr_cur not in self.tcgs[label]):
				self.tcgs[label][itr] = {}
				self.counts[label][itr] = 0
			if(itr_next not in self.tcgs[label][itr_cur]):
				self.tcgs[label][itr_cur][itr_next] = 0
			self.tcgs[label][itr_cur][itr_next] += 1
			self.tcgs[label][itr_cur] += 1
						
		# #maintain a record of what followed that ITR as n-grams

	def view_important_itrs(self):
		for label in range(self.num_classes):
			print("Label: ", label)
			for itr_name in self.tcgs[label].keys():
				itr_count = self.tcgs[label][itr_name]
				if(itr_count > 10):
					print(itr_name, itr_count)

			

	def evaluate_set(self, txt_file):
		itr_seq = self.extract_itr_seq(txt_file)

		sum_values = np.zeros(self.num_classes)

		for label in range(self.num_classes):
			for itr in itr_set:
				if(itr in self.tcgs[label] and self.tcgs[label] > 10):
					sum_values[label] += self.tcgs[label][itr]

		return np.argmax(sum_values)

	def get_dictionary(self):
		all_itrs = []
		for label in range(self.num_classes):
			all_itrs += self.tcgs[label].keys()
		print("Total words:", len(all_itrs))

	def evaluate_seq(self, txt_file):
		itr_set = self.extract_itr_set(txt_file)

		sum_values = np.zeros(self.num_classes)

		for label in range(self.num_classes):
			for itr in itr_set:
				if(itr in self.tcgs[label] and self.tcgs[label] > 10):
					sum_values[label] += self.tcgs[label][itr]

		return np.argmax(sum_values)


	def __init__(self, num_classes):
		self.num_classes = num_classes

		self.tcgs = []
		self.counts = []
		for i in range(num_classes):
			self.tcgs.append({})
			self.counts.append({})

def main(dataset_dir, csv_filename, dataset_type, dataset_id):

	num_classes = 13
	tcg = ITR_Extractor(num_classes)
	
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for ex in csv_contents:
		ex['txt_path'] = os.path.join(dataset_dir, "txt_"+dataset_type+"_"+str(dataset_id), str(0), ex['label_name'], ex['example_id']+'_0.txt')

	train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
	test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]
	
	for ex in train_data:
		tcg.add_itr_seq(ex['txt_path'], ex['label'])

	tcg.get_dictionary()

	'''
	class_acc = np.zeros((num_classes, num_classes))
	for ex in test_data:
		pred = tcg.evaluate(ex['txt_path'])
		print(pred, ex['label'])
		class_acc[pred, ex['label']] += 1

	print(class_acc)
	sum_corr = 0
	for i in range(num_classes):
		sum_corr += class_acc[i,i]
	print("TOTAL ACC: ", sum_corr/np.sum(class_acc))

	#tcg.view_important_itrs()
	'''

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	#parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow'])
	#parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')

	FLAGS = parser.parse_args()

	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		"frames", #FLAGS.dataset_type,
		1 #FLAGS.dataset_id
		)




	


		


