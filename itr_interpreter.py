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




def f_importances(coef, names, count=5):
	
	# convert Scipy matrix to one dimensional vector
	imp = coef.toarray()[0]

	# sorted into ascending order so I need to reverse 
	imp,names = zip(*sorted(zip(imp,names)))
	imp = imp[::-1]
	names = names[::-1]

	#print(imp.shape)

	top, bot = imp[:count], imp[count:]#np.stack((imp[:top], imp[top:]))
	top_n, bot_n = names[:count], names[count:]#np.stack((names[:top], names[top:]))



	print(top, type(top))


	# place into chart
	plt.barh(range(count), top, align='center')
	plt.yticks(range(count), top_n)
	#plt.savefig("test.png")
			
		

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	#restore model
	depth = 4

	save_file = os.path.join(save_name, str(dataset_id), dataset_type)
	filename = save_file.replace('/', '_')+'_'+str(depth)
	tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

	coef = tcg.clf.coef_
	names = tcg.tfidf.get_feature_names()

	'''
	Coef has shape of n_classes, n_features due to one vs rest setting of linear SVM learner
	Each coef shows the significance of a particular feature towards that classifcation
	'''
	print(coef.shape)

	#select the first class only 
	f_importances(abs(coef[0]), names)
	#f_importances(abs(coef[0]), names)



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
