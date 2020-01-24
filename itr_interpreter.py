from sets import Set
import os, sys, math, colorsys
import numpy as np

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

import matplotlib
matplotlib.use('Agg')

import cv2, time

from sklearn import metrics
from itr_sklearn import ITR_Extractor

from itertools import product
from string import ascii_lowercase

from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble='\usepackage{color}')

import matplotlib.pyplot as plt

event_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]

def e_to_idx(e):
	# convert aaa to 0
	return event_labels.index(e)

def itr_to_idx(itr):
	# convert aaa-eq-aab to 0-eq-1
	event_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]
	itr_s = itr.split('-')
	return "{0}-{1}-{2}".format(e_to_idx(itr_s[0]), itr_s[1], e_to_idx(itr_s[2]))
	

def generate_top_bottom_table(tcg, label, count=10, out="feature_importance.png", title=""):
	# get the top and bottom most features and save them in a plotable figure

	# convert Scipy matrix to one dimensional vector
	feature_names = tcg.tfidf.get_feature_names()

	# determine the importance of each ITR according to each class	
	n_classes = len(tcg.clf.classes_)
	sort_matrix = np.zeros(  ( n_classes, tcg.clf.coef_.shape[0] )  )
	
	k = 0
	for i in range(n_classes):
		for j in range(i + 1, n_classes):
			sort_matrix[i][k] += 1
			sort_matrix[j][k] -= 1
			k+=1

	importance = np.dot(tcg.clf.coef_.toarray().T, sort_matrix.T)

	# get only the importance ranks for the specific label
	importance = importance[:, label]

	# place features in descending order
	importance, feature_names = zip(*sorted(zip(importance,feature_names)))
	importance, feature_names = np.array(importance), np.array(feature_names)
	importance, feature_names = importance[::-1], feature_names[::-1]

	assert count > 0, "Count value must be greater than 0"

	# get the most and least important ITRs and their importance value
	top, bot = importance[:count], importance[-count:]
	data = top#np.concatenate((top, bot))

	top_n, bot_n = feature_names[:count], feature_names[-count:]
	names = top_n#np.concatenate((top_n, bot_n))

	names = [itr_to_idx(itr) for itr in names]


	#define ITR-coloring scheme
	itr_colors = {}
	bar_colors = []
	for i, itr in enumerate(top_n):
		itr_colors[itr] = np.linspace(0, 255, num=len(top_n), dtype=np.uint8)[i]
		bar_colors.append(colorsys.hsv_to_rgb((itr_colors[itr]/360.0), 1.0, 1.0))

	# place into chart
	'''
	names = [ r"\textcolor[rgb]{0,0,1}{"+itr_to_idx(itr)+"}" 
				# "\textcolor[hsv]{"+str(itr_colors[itr])+",1,1}{"+itr+"}" 
					if itr in itr_colors else itr_to_idx(itr) for itr in names   ]
	'''
	# define plot
	plt.figure(figsize=(2,4))
	#bar_colors = ['b']*count + ['r']*count
	#plt.barh(range(count*2), data, align='center', color=bar_colors)
	plt.barh(range(count), data, align='center', color=bar_colors)


	#plt.xticks(np.arange(np.min(bot), np.max(top)))
	#plt.yticks(range(count*2), names)
	plt.yticks(range(count), names)
	plt.gca().invert_yaxis()
	plt.title(title)
	plt.tight_layout()

	#plt.show()
	plt.savefig(out)
	plt.close()

	return top_n, itr_colors



def find_best_matching_IAD(tcg, label, top_features, itr_colors, csv_contents, out_name='iad.png'):
	'''Find the best IAD match for the label and color the IAD according to the ITRs therein.'''

	# CHOOSE THE BEST FILE TO DEMONSTRATE WITH

	# find the IAD that best matches the given IADs and color it and save fig
	files = [ex for ex in csv_contents if ex["label"] == label]
	
	# files list of files with the same label
	tcg.evalcorpus = []
	for i, ex in enumerate(files):
		tcg.add_file_to_eval_corpus(ex["txt_path"], ex["label"], ex["label_name"])

	data = tcg.tfidf.transform(tcg.evalcorpus)
	prob = tcg.clf.decision_function(data)

	pred = tcg.clf.predict(data)

	# select the greatest decision function in favor of the class
	top = np.argmax(prob[:, label], axis =0)



	#SETUP WHICH EVENTS ARE COLORED
	'''This entire section is used to define the "event_colors" dictionary.
	The dictionary is structured as follows. Since each feature can express multiple events
	we want to separate those events by their start and stop times (the tuples). The individual
	events can be related to one or more ITRs hence the list of colors. Each color represents 
	one ITR with another event:

		event_colors = {
			"feature_name" : { 
				(tuple with start and stop times of an event) : [list of colors for event] 
			}	
		}


	'''

	num_features = 0
	event_idx = {}
	for itr in top_features:
		itr_s = itr.split('-')

		e1, e2 = itr_s[0], itr_s[2]
		
		if(e1 not in event_idx):
			event_idx[e1] = num_features
			num_features += 1

		if(e2 not in event_idx):
			event_idx[e2] = num_features
			num_features += 1

	event_colors = {}


	max_window = 0

	events = sorted( tcg.read_file(files[top]["txt_path"]) )
	event_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]

	for i in range(len(events)):

		#update the size of the IAD
		if(int(events[i].end) > max_window):
			max_window = int(events[i].end)
		#if(event_labels.index(events[i].name) > num_features):
		#	num_features = event_labels.index(events[i].name)

		j = i+1
		while(j < len(events) and events[j].name != events[i].name):

			e1 = events[i]
			e2 = events[j]

			#get the ITRs in the IAD
			itr_name = e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)
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


	# MAKE THE PICTURE


	print("max_window:", max_window)

	# an empty numpy array
	iad = np.ones((num_features, max_window), np.float32)

	# change the colors space to HSV for simplicity 
	iad = cv2.cvtColor(iad,cv2.COLOR_GRAY2BGR)
	iad = cv2.cvtColor(iad,cv2.COLOR_BGR2HSV)

	for i, e in enumerate(events):

		if e.name in event_colors:
			timing_pair = (int(e.start),int(e.end))
			if timing_pair in event_colors[e.name]:

				colors = event_colors[e.name][timing_pair]
				for idx in range(int(e.start), int(e.end)):
					iad[event_idx[e.name] , idx, 0] = colors[idx % len(colors)]
					iad[event_idx[e.name] , idx, 1] = 1
			#else:
				#iad[event_labels.index(e.name) , int(e.start):int(e.end), 2]  = 0
		#else:
		#	iad[event_labels.index(e.name) , int(e.start):int(e.end), 2]  = 0

	
	# trim the front of the IAD
	iad = iad[:, 3:]

	salient_frames = iad[:, :, 1]
	salient_frames = np.sum(salient_frames, axis=0)
	salient_frames = np.where(salient_frames == np.max(salient_frames))[0]
	print(salient_frames)

	cluster_medians = []

	s = 0
	for i in range(1, len(salient_frames)):
		if(salient_frames[i] - salient_frames[i-1] > 1):
			cluster_medians.append(int(np.mean(salient_frames[s: i-1])))
			s = i

	cluster_medians.append(int(np.mean(salient_frames[s: i-1])))
	salient_frames = cluster_medians#np.where(salient_frames == np.max(salient_frames))

	#print("salient_frames.shape:", salient_frames.shape)

	# put back into the BGR colorspace for display/save
	iad = cv2.cvtColor(iad,cv2.COLOR_HSV2BGR)


	# format the IAD as a uint8
	iad *= 255
	iad = iad.astype(np.uint8)

	#resize the IAD
	scale = 30
	iad = cv2.resize(iad, (iad.shape[1]*scale, iad.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

	#Write some Text

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 1
	fontColor              = (0,0,0)
	lineType               = 2

	# add border to IAD until it is the same length as the feature-graph image
	
	t, l = 50, 100
	iad = cv2.copyMakeBorder(
		iad,
		top=t,
		bottom=0,
		left=l,
		right=0,
		borderType=cv2.BORDER_CONSTANT,
		value=[255,255,255]
	)

	#feature labels
	for i, e in enumerate(event_idx.keys()):
		cv2.putText(iad,str(e_to_idx(e)), 
			(10,i*scale+scale+t-5), 
			font, 
			fontScale,
			fontColor,
			lineType)

	# salient points
	for i, e in enumerate(salient_frames):
		cv2.putText(iad,str(i), 
			(e*scale+l+5, t-15), 
			font, 
			fontScale,
			fontColor,
			lineType)
		cv2.circle(iad, (e*scale+l+scale/2, t-10 -scale/2), int(scale*.75), fontColor, lineType)

	#cv2.imshow("iad", iad)
	cv2.imwrite(out_name, iad)

	return files[top], salient_frames

def make_graph(top_features, itr_colors, name="graph.png"):
	'''Make a graph from the top features, the edges indicate the ITRs being represented.'''
	
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

		# pydot HSV goes up to 360 rather than 256
		c = itr_colors[itr]/360.0

		edges += '{0} -> {1} [label="{2}" color="{3} 1.0 1.0" ]\n'.format(e_to_idx(itr_s[0]), e_to_idx(itr_s[2]), itr_s[1], round(c, 3))

	# add nodes to the dot file
	for e in events:
		nodes += 'node [shape=circle,style=filled] {0}\n'.format(e_to_idx( e ))
	gfile.write(nodes)
	
	# add the edges to the dot files
	gfile.write(edges+"}\n")
	gfile.close()

	# generate image from dot file
	from subprocess import check_call
	check_call(['dot','-Tpng','mygraph.dot','-o',name])

def find_video_frames(dataset_dir, file_ex, salient_frames, depth, out_name="frames.png"):
	# create a figure that highlights the frames in the iad

	max_window_scale = [2, 2, 2, 4, 8]
	img_files = [os.path.join(dataset_dir, 'frames', file_ex['label_name'], file_ex['example_id'], 'image_'+str(frame_num*max_window_scale[depth]+3).zfill(5)+'.jpg') for frame_num in salient_frames]

	print(img_files)

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 2
	fontColor              = (0,0,0)
	lineType               = 4
	
	big_img = cv2.imread(img_files[0])

	radius = 50
	pos = (big_img.shape[1]/2 -35 ,60)
	txt_pos = (pos[0]-20, pos[1]+15)
	i = 0

	
	cv2.circle(big_img, pos, radius, (255,255,255), -1)
	cv2.circle(big_img, pos, radius, fontColor, lineType)
	cv2.putText(big_img,str(i), 
		txt_pos, 
		font, 
		fontScale,
		fontColor,
		lineType)
	i+=1
	

	for file in img_files[1:]:
		img = cv2.imread(file)
		cv2.circle(img, pos, radius, (255,255,255), -1)
		cv2.circle(img, pos, radius, fontColor, lineType, )
		cv2.putText(img,str(i), 
			txt_pos, 
			font, 
			fontScale,
			fontColor,
			lineType)
		i+=1

		img = cv2.copyMakeBorder(
			img,
			top=0,
			bottom=0,
			left=10,
			right=0,
			borderType=cv2.BORDER_CONSTANT,
			value=[0,0,0]
		)

		big_img = np.concatenate((big_img, img), axis=1)

	cv2.imwrite(out_name, big_img)

	return 0
	
def combine_images(features="", iad = "", graph="", out_name ="" ):
	'''Combine several images together into a single image'''

	feature_img = cv2.imread(features)
	graph_img = cv2.imread(graph)
	iad_img = cv2.imread(iad)

	
	#resize feature and graph to be the same height
	if(feature_img.shape[0] > graph_img.shape[0]):
		#scale = feature_img.shape[0]/float(graph_img.shape[0])
		#graph_img = cv2.resize(graph_img, (int(graph_img.shape[1]*scale), feature_img.shape[0]))
		graph_img = cv2.copyMakeBorder(
			graph_img,
			top=0,
			bottom=feature_img.shape[0]-graph_img.shape[0],
			left=0,
			right=0,
			borderType=cv2.BORDER_CONSTANT,
			value=[255,255,255]
		)

	else:
		#scale = graph_img.shape[0]/float(feature_img.shape[0])
		#feature_img = cv2.resize(feature_img, (int(feature_img.shape[1]*scale), graph_img.shape[0]))
		feature_img = cv2.copyMakeBorder(
			feature_img,
			top=0,
			bottom=graph_img.shape[0]-feature_img.shape[0],
			left=0,
			right=0,
			borderType=cv2.BORDER_CONSTANT,
			value=[255,255,255]
		)
	
	# combine feature and graph together
	fg_img = np.concatenate((feature_img, graph_img), axis = 1)
	
	'''
	# add border to IAD until it is the same length as the feature-graph image
	iad_img = cv2.copyMakeBorder(
		iad_img,
		top=0,
		bottom=fg_img.shape[0]-iad_img.shape[0],
		left=0,
		right=0,
		borderType=cv2.BORDER_CONSTANT,
		value=[255,255,255]
	)
	'''

	# add border to IAD until it is the same length as the feature-graph image
	
	if(fg_img.shape[0] > iad_img.shape[0]):
		scale = fg_img.shape[0]/float(iad_img.shape[0])
		iad_img = cv2.resize(iad_img, (int(iad_img.shape[1]*scale), fg_img.shape[0]), interpolation=cv2.INTER_NEAREST)

	else:
		scale = iad_img.shape[0]/float(fg_img.shape[0])
		fg_img = cv2.resize(fg_img, (int(fg_img.shape[1]*scale), iad_img.shape[0]))

	print("fg_img.shape[0]-iad_img.shape[0]", fg_img.shape, iad_img.shape)

	#combine all images together
	combined = np.concatenate((fg_img, iad_img), axis = 1)
	cv2.imwrite(out_name, combined)



def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name):

	dir_root = os.path.join("pics", save_name)
	

	for depth in range(1):#5):

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

		for label in range(1):#num_classes):

			label_name = [ex for ex in csv_contents if ex['label'] == label][0]['label_name']
			title = label_name.upper().replace('_', ' ')+", Depth "+str(depth)

			print(title)
			feat_name = os.path.join(dir_name, label_name+'_feat_'+str(depth)+'.png')
			graph_name = os.path.join(dir_name, label_name+'_graph_'+str(depth)+'.png')
			iad_name = os.path.join(dir_name, label_name+'_iad_'+str(depth)+'.png')
			frames_name = os.path.join(dir_name, label_name+'_frames_'+str(depth)+'.png')

			combined_name = os.path.join(dir_name, label_name+'_'+str(depth)+'.png')





			# generate a plot that shows the top 5 and bottom five features for each label.
			top_features, colors = generate_top_bottom_table(tcg, label, count=5, out=feat_name, title=title)

			# from there we need to open an IAD and highlight the rows that are described in the table
			# use the same colorsfor the regions specified

			make_graph(top_features, colors, name=graph_name)

			file_ex, salient_frames = find_best_matching_IAD(tcg, label, top_features, colors, csv_contents, out_name=iad_name)
			find_video_frames(dataset_dir, file_ex, salient_frames, depth, out_name=frames_name)

			# lastly we can look at frames in the video corresponding to those IADs
			#find_video_frames()

			
			combine_images(
				features=feat_name, 
				iad=iad_name,
				graph=graph_name,
				out_name=combined_name )
			
			#os.remove(feat_name)
			#os.remove(iad_name)
			#os.remove(graph_name)

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
