from sets import Set
import os, sys, math
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

import matplotlib
matplotlib.use('Agg')

import cv2, time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import VotingClassifier


import matplotlib


from itr_sklearn import ITR_Extractor


from itertools import product
from string import ascii_lowercase
#https://buhrmann.github.io/tfidf-analysis.html

import eli5
from eli5.lime import TextExplainer
from eli5.sklearn import PermutationImportance
from sklearn.pipeline import Pipeline, make_pipeline

import pydot
import colorsys

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble='\usepackage{color}')

import matplotlib.pyplot as plt

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
'''
def generate_top_bottom_table(tcg, label, csv_contents, count=10, out="feature_importance.png"):

	files = [ex for ex in csv_contents if ex["label"] == label]
	for i, ex in enumerate(files):
		#tcg.add_file_to_eval_corpus(ex["txt_path"], label, ex["label_name"])
		tcg.evalcorpus.append(tcg.parse_txt_file(ex["txt_path"]))
		tcg.evallabels.append(label)


	#data = tcg.tfidf.transform(tcg.evalcorpus)

	t_s = time.time()

	te = TextExplainer(random_state=42)
	pipe = make_pipeline(tcg.tfidf, tcg.clf)
	te.fit(tcg.evalcorpus[0], pipe.predict_proba)
	out = te.show_weights(target_names=range(13))
	print(out)
	print(out.data)
	"""
	perm = PermutationImportance(tcg.clf).fit(data.toarray(), tcg.evallabels)
	out = eli5.show_weights(perm, feature_names=tcg.tfidf.get_feature_names())
	print(out.data)
	"""
	print("Elapsed Time:", time.time() - t_s)

	return None, None
'''


# Assign unique colors to each ITR
# go through the ITR extraction processes to determine the specific event relationships

# have a dictionary
'''
adict = {"event name": {[start:end] : ["color 1", "color2"], [start2:end2]: ["color2", "color3"]} }

'''

# Generate the IAD using black to fill in the actions

# go backthrough and 











def generate_top_bottom_table(tcg, label, count=10, out="feature_importance.png", title=""):
	# get the top and bottom most features and save them in a plotable figure

	# convert Scipy matrix to one dimensional vector
	#importance = tcg.clf.coef_[label].toarray()[0]
	feature_names = tcg.tfidf.get_feature_names()


	n_classes = len(tcg.clf.classes_)

	sort_matrix = np.zeros(  ( n_classes, tcg.clf.coef_.shape[0] )  )

	k = 0
	for i in range(n_classes):
		for j in range(i + 1, n_classes):
			sort_matrix[i][k] += 1
			sort_matrix[j][k] -= 1
			k+=1

	#print("coef:", tcg.clf.coef_.shape)
	#print("sort:", sort_matrix.shape)

	importance = np.dot(tcg.clf.coef_.toarray().T, sort_matrix.T)
	#print("out:", importance.shape)	
	importance = importance[:, label]















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



		itr_colors = {}
		label_colors = []

		c_i = 0
		for i, itr in enumerate(top_n):
			itr_colors[itr] = np.linspace(0, 255, num=len(top_n), dtype=np.uint8)[i]
			rgb_color = colorsys.hsv_to_rgb(itr_colors[itr]/256.0, 1.0, 1.0)


			label_colors.append(rgb_color)

		label_colors += [(0.0,0.0,0.0)]*count
		#print("label_colors")
		#print(label_colors)

		colors = ['b']*count + ['r']*count

		# place into chart
		
		#label = r"This is \textbf{line 1}"

		names = [ r"\textcolor[rgb]{0,0,1}{"+itr+"}" 
					# "\textcolor[hsv]{"+str(itr_colors[itr])+",1,1}{"+itr+"}" 
						if itr in itr_colors else itr for itr in names   ]

		#print("names")
		#print(names)

		plt.figure(figsize=(3,3))

		plt.barh(range(count*2), data, align='center', color = colors)

		#plt.xticks(np.arange(np.min(bot), np.max(top)))
		plt.yticks(range(count*2), names)
		plt.gca().invert_yaxis()
		plt.title(title)
		plt.tight_layout()
	else:
		colors = ['b']*len(importance[importance > 0]) + ['r']*len(importance[importance < 0])

		# place into chart
		plt.barh(range(len(importance)), importance, align='center', color = colors)
		#plt.yticks(range(len(feature_names)), feature_names)

	#plt.show()
	plt.savefig(out)

	return top_n, itr_colors


def find_best_matching_IAD(tcg, label, top_features, itr_colors, csv_contents, out_name='iad.png'):

	# CHOOSE THE FILE TO DEMONSTRATE WITH

	# find the IAD that best matches the given IADs and color it and save fig
	tcg.evalcorpus = []

	files = [ex for ex in csv_contents if ex["label"] == label]
	
	# files list of files with the same label
	for i, ex in enumerate(files):
		tcg.add_file_to_eval_corpus(ex["txt_path"], ex["label"], ex["label_name"])

	data = tcg.tfidf.transform(tcg.evalcorpus)
	prob = tcg.clf.decision_function(data)

	pred = tcg.clf.predict(data)

	#print("prob.shape:", prob.shape)
	#print(pred)
	#print(tcg.evallabels)


	# select the greatest decision function in favor of the class
	top = np.argmax(prob[:, label], axis =0)
	#print(prob[top], files[top]["iad_path"])


	#for x, y in zip(tcg.evallabels, pred):
	#	print(x,y)
	#print("metrics:", metrics.accuracy_score(pred, tcg.evallabels))

	
	#print("itr_colors")
	#print(itr_colors)

	#SETUP WHICH EVENTS ARE COLORED
	
	events = sorted( tcg.read_file(files[top]["txt_path"]) )

	event_colors = {}

	num_features = 128#len(events)
	max_window = 0

	for i in range(len(events)):

		if(int(events[i].end) > max_window):
			max_window = int(events[i].end)

		j = i+1
		while(j < len(events) and events[j].name != events[i].name):

			e1 = events[i]
			e2 = events[j]
			itr_name = e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)#tcg.all_itrs(events[i], events[j], 0)

			itr = "{0}-{1}-{2}".format(e1.name, itr_name, e2.name)

			if(itr in itr_colors):

				

				if e1.name not in event_colors:
					event_colors[ e1.name ] = {}
				if( (e1.start, e1.end) not in event_colors[ e1.name ]):
					event_colors[ e1.name ][(e1.start, e1.end)] = []

				event_colors[ e1.name ][(e1.start, e1.end)].append(itr_colors[itr])

				if e2.name not in event_colors:
					event_colors[ e2.name ] = {}
				if( (e2.start, e2.end) not in event_colors[ e2.name ]):
					event_colors[ e2.name ][(e2.start, e2.end)] = []

				event_colors[ e2.name ][(e2.start, e2.end)].append(itr_colors[itr])


			j+=1

	#print("event_colors")
	#print(event_colors)


	# MAKE THE PICTURE

	#num_features = 128 #get from the num used features
	#max_window = 256 
	iad = np.ones((num_features, max_window), np.float32)#np.array(np.ones((num_features, max_window)), dtype=np.uint8) * 255

	# change the colors space to HSV for simplicity 
	iad = cv2.cvtColor(iad,cv2.COLOR_GRAY2BGR)
	iad = cv2.cvtColor(iad,cv2.COLOR_BGR2HSV)

	action_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]


	for i, e in enumerate(events):

		if e.name in event_colors:
			timing_pair = (int(e.start),int(e.end))
			if timing_pair in event_colors[e.name]:

				colors = event_colors[e.name][timing_pair]
				for idx in range(int(e.start), int(e.end)):
					iad[action_labels.index(e.name) , idx, 0] = colors[idx % len(colors)]
					iad[action_labels.index(e.name) , idx, 1]  = 1
			#else:
				#iad[action_labels.index(e.name) , int(e.start):int(e.end), 2]  = 0
		else:
			iad[action_labels.index(e.name) , int(e.start):int(e.end), 2]  = 0


	#print("after iad[0,0]:", iad[0,0])

	iad = cv2.cvtColor(iad,cv2.COLOR_HSV2BGR)

	#print("format iad[0,0]:", iad[0,0])

	iad *= 255
	iad = iad[3:]
	#print("incr iad[0,0]:", iad[0,0])

	iad = iad.astype(np.uint8)

	scale = 4
	iad = cv2.resize(iad, (iad.shape[1]*scale, iad.shape[0]*scale), interpolation=cv2.INTER_NEAREST)


	#print("type iad[0,0]:", iad[0,0])

	cv2.imwrite(out_name, iad)
	#cv2.imshow('img', canvas)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()


	#https://www.youtube.com/watch?v=AAC-1wySMLM

	

	return None
	return iad, frame_durations

def find_video_frames():
	# create a figure that highlights the frames in the iad
	return 0





def make_graph(top_features, itr_colors, name="graph.png"):

	

	gfile = open('mygraph.dot', 'w')

	header = "digraph A {\nrankdir = LR;\n"
	gfile.write(header)

	nodes = ""
	edges = ""

	events = Set()
	for itr in top_features:
		itr_s = itr.split('-')
		events.add(itr_s[0])
		events.add(itr_s[2])

		c = itr_colors[itr]/360.0
		#print("itr_colors[itr]:", c, itr_colors[itr])

		edges += '{0} -> {1} [label="{2}" color="{3} 1.0 1.0" ]\n'.format(itr_s[0], itr_s[2], itr_s[1], round(c, 3))

	for e in events:
		nodes += 'node [shape=circle,style=filled] {0}\n'.format(e)
	gfile.write(nodes)
	
	
	gfile.write(edges+"}\n")

	gfile.close()

	#print("pydot.graph_from_dot_file('mygraph.dot'):", pydot.graph_from_dot_file('mygraph.dot'))

	#(graph,) = pydot.graph_from_dot_file('mygraph.dot')
	#graph.write_png('graph.png')

	from subprocess import check_call
	check_call(['dot','-Tpng','mygraph.dot','-o',name])
	
def combine_images(features="", iad = "", graph="", out_name ="" ):

	feature_img = cv2.imread(features)
	graph_img = cv2.imread(graph)
	iad_img = cv2.imread(iad)
	
	#print("feature:", feature_img.shape)
	#print("graph:", graph_img.shape)

	#resize feature and graph to be the same height
	if(feature_img.shape[0] > graph_img.shape[0]):
		scale = feature_img.shape[0]/float(graph_img.shape[0])
		graph_img = cv2.resize(graph_img, (int(graph_img.shape[1]*scale), feature_img.shape[0]))

	else:
		scale = graph_img.shape[0]/float(feature_img.shape[0])
		feature_img = cv2.resize(feature_img, (int(feature_img.shape[1]*scale), graph_img.shape[0]))

	#print("feature2:", feature_img.shape)
	#print("graph2:", graph_img.shape)



	fg_img = np.concatenate((feature_img, graph_img), axis = 1)

	#cv2.imwrite("fg_img.png", fg_img)
	#resize IAD to be the same scale as the width
	#print("border:", fg_img.shape[1]-iad_img.shape[1])
	iad_img = cv2.copyMakeBorder(
		iad_img,
		top=0,
		bottom=0,
		left=0,
		right=fg_img.shape[1]-iad_img.shape[1],
		borderType=cv2.BORDER_CONSTANT,
		value=[255,255,255]
	)

	#combine images together
	combined = np.concatenate((fg_img, iad_img), axis = 0)
	cv2.imwrite(out_name, combined)

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	dir_root = os.path.join("pics", "bm")
	

	for depth in range(5):

		dir_name = os.path.join(dir_root, str(depth))

		if(not os.path.exists(dir_name)):
			os.makedirs(dir_name)

		#open files
		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		for ex in csv_contents:
			ex['iad_path'] = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id), ex['label_name'], ex['example_id']+"_"+str(depth)+".npz")
			ex['txt_path'] = os.path.join(dataset_dir, "txt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')
		csv_contents = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]

		# open saved model
		save_file = os.path.join(save_name, str(dataset_id), dataset_type)
		filename = save_file.replace('/', '_')+'_'+str(depth)
		tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

		for label in range(num_classes):

			label_name = [ex for ex in csv_contents if ex['label'] == label][0]['label_name']
			title = label_name.upper()+", Depth "+str(depth)

			print(title)
			feat_name = os.path.join(dir_name, label_name+'_feat_'+str(depth)+'.png')
			graph_name = os.path.join(dir_name, label_name+'_graph_'+str(depth)+'.png')
			iad_name = os.path.join(dir_name, label_name+'_iad_'+str(depth)+'.png')

			combined_name = os.path.join(dir_name, label_name+'_'+str(depth)+'.png')





			# generate a plot that shows the top 5 and bottom five features for each label.
			top_features, colors = generate_top_bottom_table(tcg, label, count=5, out=feat_name, title=title)

			# from there we need to open an IAD and highlight the rows that are described in the table
			# use the same colorsfor the regions specified

			make_graph(top_features, colors, name=graph_name)

			find_best_matching_IAD(tcg, label, top_features, colors, csv_contents, out_name=iad_name)

			# lastly we can look at frames in the video corresponding to those IADs
			#find_video_frames()


			combine_images(
				features=feat_name, 
				iad=iad_name,
				graph=graph_name,
				out_name=combined_name )

			os.remove(feat_name)
			os.remove(iad_name)
			os.remove(graph_name)

			print('----------------')


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
