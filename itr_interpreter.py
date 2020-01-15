from sets import Set
import os, sys, math
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

#import matplotlib
#matplotlib.use('Agg')

import cv2

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier


import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor

#https://buhrmann.github.io/tfidf-analysis.html


'''
def f_importances(coef, names, count=5):
	
	# convert Scipy matrix to one dimensional vector
	imp = coef.toarray()[0]

	# sorted into ascending order so I need to reverse 
	imp,names = zip(*sorted(zip(imp,names)))
	imp, names = np.array(imp), np.array(names)

	imp = imp[::-1]
	names = names[::-1]

	#print(imp.shape)

	top, bot = imp[:count], imp[-count:]#np.stack((imp[:top], imp[top:]))
	top_n, bot_n = names[:count], names[-count:]#np.stack((names[:top], names[top:]))

	data = np.concatenate((top, bot))
	labels = np.concatenate((top_n, bot_n))
	colors = ['b']*count + ['r']*count

	# place into chart

	plt.barh(range(count*2), data, align='center', color=colors)
	plt.yticks(range(count*2), labels)
	plt.tight_layout()

	plt.show()
	plt.savefig("test.png")
	
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	#restore model
	depth = 4

	save_file = os.path.join(save_name, str(dataset_id), dataset_type)
	filename = save_file.replace('/', '_')+'_'+str(depth)
	tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

	coef = tcg.clf.coef_
	names = tcg.tfidf.get_feature_names()

	
	#print(coef.shape)

	#select the first class only 
	f_importances((coef[0]), names)
	#f_importances(abs(coef[0]), names)

'''

def generate_top_bottom_table(tcg, label, count=10, out="feature_importance.png"):
	# get the top and bottom most features and save them in a plotable figure

	# convert Scipy matrix to one dimensional vector
	importance = tcg.clf.coef_[label].toarray()[0]
	feature_names = tcg.tfidf.get_feature_names()

	# place features in descending order
	importance, feature_names = zip(*sorted(zip(importance,feature_names)))
	importance, feature_names = np.array(importance), np.array(feature_names)

	importance = importance[::-1]
	feature_names = feature_names[::-1]


	if(count > 0):
		top, bot = importance[:count], importance[-count:]
		top_n, bot_n = feature_names[:count], feature_names[-count:]

		data = np.concatenate((top, bot))
		names = np.concatenate((top_n, bot_n))

		colors = ['b']*count + ['r']*count

		# place into chart
		plt.barh(range(count*2), data, align='center', color = colors)
		plt.yticks(range(count*2), names)
	else:
		colors = ['b']*len(importance[importance > 0]) + ['r']*len(importance[importance < 0])

		# place into chart
		plt.barh(range(len(importance)), importance, align='center', color = colors)
		#plt.yticks(range(len(feature_names)), feature_names)

	#plt.show()
	#plt.savefig(out)

	return importance[importance > 0], None#top_n, bot_n

def find_best_matching_IAD(tcg, label, top_features, csv_contents, out_name='iad.png'):
	# find the IAD that best matches the given IADs and color it and save fig
	tcg.evalcorpus = []
	
	
	files  = [ex for ex in csv_contents if ex["label"] == label]
	
	# files list of files with the same label
	for i, ex in enumerate(files):
		tcg.add_file_to_eval_corpus(ex["txt_path"], label, ex["label_name"])

	data = tcg.tfidf.transform(tcg.evalcorpus)
	prob = tcg.clf.decision_function(data)

	# select the greatest decision function in favor of the class
	top = np.argmax(prob[:, label], axis =0)
	print(prob[top], files[top]["iad_path"])

	'''
	itr_seq = [itr[0]+'-'+itr[1]+'-'+itr[2] for itr in tcg.extract_itr_seq(files[top]["txt_path"])]
	for f in top_features:
		print(f, f in itr_seq)
	'''

	for i, ex in enumerate(files):

		itr_seq = [itr[0]+'-'+itr[1]+'-'+itr[2] for itr in tcg.extract_itr_seq(ex["txt_path"])]
		#itr_seq = [itr[1] for itr in tcg.extract_itr_seq(ex["txt_path"])] #+ ['adi-eq-aaa']

		tally = 0
		for f in top_features:
			if(f in itr_seq):
				tally += 1
			#print(f, f in itr_seq)

		if(count > max_count):
			max_label = ex["txt_path"]
			max_count = count

		print(ex["example_id"], tally)

	#print(max_count, max_label)

	#DEBUG: investigate to see how many of the top_features are present in the specified class

	

	# It looks better if I can just paste it ontop of an existing IAD



	#Generate IAD from txt file

	#iad = np.zeros

	num_features = 128 #get from the num used features
	max_window = 256 
	canvas = np.ones((num_features, max_window))

	events = tcg.read_file(files[top]["txt_path"])
	for i, e in enumerate(events):
		print(e.name, e.start, e.end)
		feature = e.name #figure this out

		canvas[i, int(e.start):int(e.end)] = 0
	
	#cv2.imsave(out_name, canvas)
	#cv2.imshow('img', canvas)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()




	return None
	return iad, frame_durations

def find_video_frames():
	# create a figure that highlights the frames in the iad
	return 0


def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	depth = 4

	#open files
	try:
		csv_contents = read_csv(csv_filename)
	except:
		print("ERROR: Cannot open CSV file: "+ csv_filename)

	for ex in csv_contents:
		ex['iad_path'] = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id), ex['label_name'], ex['example_id']+"_"+str(depth)+".npz")
		ex['txt_path'] = os.path.join(dataset_dir, "gtxt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')
	csv_contents = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]

	# open saved model
	save_file = os.path.join(save_name, str(dataset_id), dataset_type)
	filename = save_file.replace('/', '_')+'_'+str(depth)
	tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

	for label in range(1):#num_classes):

		# generate a plot that shows the top 5 and bottom five features for each label.
		top_features, bottom_features = generate_top_bottom_table(tcg, label, count=50, out='test.png')

		# from there we need to open an IAD and highlight the rows that are described in the table
		# use the same colorsfor the regions specified

		find_best_matching_IAD(tcg, label, top_features, csv_contents)

		# lastly we can look at frames in the video corresponding to those IADs
		find_video_frames()

		print('----------------')


def generate_top_bottom_table(dataset_type, dataset_id, num_classes, save_name, label, out_name="feature_importance.png"):
	# get the top and bottom most features and save them in a plotable figure

	# open saved model
	save_file = os.path.join(save_name, str(dataset_id), dataset_type)
	filename = save_file.replace('/', '_')+'_'+str(depth)
	tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

	

	# convert Scipy matrix to one dimensional vector
	importance = abs(tcg.clf.coef_[label]).toarray()[0]
	feature_names = tcg.tfidf.get_feature_names()

	# place features in descending order
	importance, feature_names = zip(*sorted(zip(imp,feature_names)))
	importance = importance[::-1]
	feature_names = feature_names[::-1]

	count = 5
	top, bot = imp[:count], imp[count:]
	top_n, bot_n = names[:count], names[count:]

	data = #combine top and bot
	names = #combine top_n and bot_n

	# place into chart
	plt.barh(range(count), data, align='center')
	plt.yticks(range(count), names)
	plt.savefig(out_name)

	return tcg, top_n, bot_n

def find_best_matching_IAD(tcg, top_features):
	# find the IAD that best matches the given IADs and color it and save fig

	# files list of files with the same label
	for i, f in enumerate(files):
		tcg.add_file_to_eval_corpus(f, label, label_name)

	data = self.models.tfidf.transform(self.models[depth].evalcorpus)
	prob = self.models.clf.decision_function(data)

	# select the greatest decision function in favor of the class
	top = np.argmax(prob, axis =0)


	#DEBUG: investigate to see how many of the top_features are present in the specified class




	#decorate IAD using ITRs
	



	return iad, frame_durations

def find_video_frames():
	# create a figure that highlights the frames in the iad


def main():

	# for a specific model

	for label in range(num_classes):

		# generate a plot that shows the top 5 and bottom five features for each label.
		tcg, top_features, bottom_features = generate_top_bottom_table()

		# from there we need to open an IAD and highlight the rows that are described in the table
		# use the same colorsfor the regions specified
		find_best_matching_IAD(top_features)

		# lastly we can look at frames in the video corresponding to those IADs
		find_video_frames()


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
