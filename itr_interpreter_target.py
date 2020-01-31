from sets import Set
import os, sys, math, colorsys
import numpy as np
import tensorflow as tf

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv
import tf_utils
import rank_i3d as model
from feature_rank_utils import get_top_n_feature_indexes, get_top_n_feature_indexes_combined

import matplotlib
matplotlib.use('Agg')

import cv2, time

from sklearn import metrics
from itr_sklearn import ITR_Extractor

from itertools import product
from string import ascii_lowercase

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

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
	feature_names = tcg.tfidf.get_feature_names()

	# place features in descending order
	importance, feature_names = zip(*sorted(zip(importance,feature_names)))
	importance, feature_names = np.array(importance), np.array(feature_names)
	importance, feature_names = importance[::-1], feature_names[::-1]

	idx = [i for i in range(len(feature_names)) if feature_names[i].find('-d-') == -1 and 
	feature_names[i].find('-eq-') == -1 and feature_names[i].find('-b-') == -1]
	#importance, feature_names = importance[idx], feature_names[idx]

	assert count > 0, "Count value must be greater than 0"

	# get the most and least important ITRs and their importance value
	top, bot = importance[:count], importance[-count:]
	data = top#np.concatenate((top, bot))

	top_n, bot_n = feature_names[:count], feature_names[-count:]
	names = top_n#np.concatenate((top_n, bot_n))

	#define ITR-coloring scheme
	itr_colors = {}
	bar_colors = []
	for i, itr in enumerate(top_n):
		itr_colors[itr] = np.linspace(0, 255, num=len(top_n), dtype=np.uint8)[i]
		bar_colors.append(colorsys.hsv_to_rgb((itr_colors[itr]/360.0), 1.0, 1.0))

	event_colors = {}
	for itr in top_n:
		itr_s = itr.split('-')
		if(e_to_idx(itr_s[0]) not in event_colors):
			event_colors[e_to_idx(itr_s[0])] = 0
		if(e_to_idx(itr_s[2]) not in event_colors):
			event_colors[e_to_idx(itr_s[2])] = 0
	for i, k in enumerate(event_colors.keys()):
		event_colors[k] = np.linspace(0, 255, num=len(event_colors), dtype=np.uint8)[i]

	# define plot
	plt.figure(figsize=(2,4))
	plt.barh(range(count), data, align='center', color=bar_colors)

	plt.yticks(range(count), [itr_to_idx(itr) for itr in names], fontsize=15)
	plt.gca().invert_yaxis()
	plt.title(title)
	plt.tight_layout()

	plt.savefig(out)
	plt.close()

	return top_n, itr_colors, event_colors



def find_best_matching_IAD(tcg, label, top_features, itr_colors, csv_contents, out_name='iad.png'):
	'''Find the best IAD match for the label and color the IAD according to the ITRs therein.'''

	# CHOOSE THE BEST FILE TO DEMONSTRATE WITH

	# find the IAD that best matches the given IADs and color it and save fig
	files = [ex for ex in csv_contents if ex["label"] == label]
	
	
	top = -1
	itr_count = 0

	# find the most ITRs
	for f in range(len(files)):
		print(f, len(files))
		eg_count = 0

		events = sorted( tcg.read_file(files[f]["txt_path"]) )
		event_labels = [''.join(i) for i in product(ascii_lowercase, repeat = 3)]

		for i in range(len(events)):

			j = i+1
			while(j < len(events) and events[j].name != events[i].name):
				e1 = events[i]
				e2 = events[j]

				#get the ITRs in the IAD
				itr_name = e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)
				itr = "{0}-{1}-{2}".format(e1.name, itr_name, e2.name)

				if(itr in itr_colors):	
					eg_count+=1
				j+=1

		if(eg_count > itr_count):
			top = f 
			itr_count = eg_count

	print(files[top]["txt_path"])
	'''
	# find best performing examples

	# files list of files with the same label
	tcg.evalcorpus = []
	for i, ex in enumerate(files):
		tcg.add_file_to_eval_corpus(ex["txt_path"], ex["label"], ex["label_name"])

	data = tcg.tfidf.transform(tcg.evalcorpus)
	prob = tcg.clf.decision_function(data)

	pred = tcg.clf.predict(data)

	# select the greatest decision function in favor of the class
	top = np.argmax(prob[:, label], axis =0)
	'''


	#SETUP WHICH EVENTS ARE COLORED
	'''
	This entire section is used to define the "event_colors" dictionary.
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

	event_list = Set()

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

				event_list.add(e1.name) 
				event_list.add(e2.name) 		

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

	event_list = sorted(list(event_list))

	for i, e in enumerate(events):

		if e.name in event_colors:
			timing_pair = (int(e.start),int(e.end))
			if timing_pair in event_colors[e.name]:

				colors = event_colors[e.name][timing_pair]
				for idx in range(int(e.start), int(e.end)):

					iad_idx = event_list.index(e.name)

					iad[ iad_idx , idx, 0] = colors[idx % len(colors)]
					iad[ iad_idx , idx, 1] = 1
			
	# trim the front of the IAD
	iad = iad[:, 3:]

	'''

	salient_frames = iad[:, :, 1]
	salient_frames = np.sum(salient_frames, axis=0)
	salient_frames = np.where(salient_frames == np.max(salient_frames))[0]
	#print(salient_frames)

	if(len(salient_frames) > 0):

		cluster_medians = []

		s = 0
		cluster_medians.append(salient_frames[0])
		for i in range(1, len(salient_frames)):
			if(salient_frames[i] - salient_frames[i-1] > 1):
				
				cluster_medians.append(salient_frames[i-1])
		cluster_medians.append(salient_frames[-1])
	
		salient_frames = cluster_medians
	else:
		salient_frames = []
	'''

	from sklearn.cluster import MeanShift, estimate_bandwidth

	salient_frames = Set()
	events = sorted( tcg.read_file(files[top]["txt_path"]) )

	print("len(events):", len(events))
	print("itr_colors:", itr_colors.keys())

	for i in range(len(events)):

		j = i+1
		while(j < len(events) and events[j].name != events[i].name):

			e1 = events[i]
			e2 = events[j]

			#get the ITRs in the IAD
			itr_name = e1.get_itr_from_time(e1.start, e1.end, e2.start, e2.end)
			itr = "{0}-{1}-{2}".format(e1.name, itr_name, e2.name)

			if(itr in itr_colors):				

				salient_frames.add(e1.start)
				salient_frames.add(e1.start+(e1.end-1-e1.start)/2)
				salient_frames.add(e1.end-1)

				salient_frames.add(e2.start)
				salient_frames.add(e2.start+(e2.end-1-e2.start)/2)
				salient_frames.add(e2.end-1)

				print("f1:", e_to_idx(e1.name), e1.start, e1.end)
				print("f2:", e_to_idx(e2.name), e2.start, e2.end)

			j+=1

	print("len(salient_frames):", len(salient_frames))

	salient_frames = sorted(list(salient_frames))
	salient_frames = [int(x-3) for x in salient_frames]

	X = np.array(zip(np.array(salient_frames),np.zeros(len(np.array(salient_frames)))), dtype=np.int)
	
	if (len(X) > 0):
		bandwidth = estimate_bandwidth(X, quantile=0.15)
	else:
		return files[top], []

	if (bandwidth > 0):
		#clustering
		
		ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
		ms.fit(X)
		labels = ms.labels_
		cluster_centers = ms.cluster_centers_

		labels_unique = np.unique(labels)
		n_clusters_ = len(labels_unique)

		print("Clusters")
		for k in range(n_clusters_):
			my_members = labels == k
			print "cluster {0}: {1}, {2}".format(k, X[my_members, 0]#
				, np.mean(X[my_members, 0]))

		salient_frames = [int(np.mean(X[labels == k, 0])) for k in range(n_clusters_) if len(X[labels == k, 0]) != 0]

		salient_frames = sorted(list(salient_frames))
	

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
	
	t, l = 110, 100
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
			(e*scale+l-5, t-40), 
			font, 
			2,
			fontColor,
			8)

		pos = (e*scale+l+scale/2, t-10 -scale/2)
		x, y = pos[0], pos[1]+25
		w, h = 50, 50

		pts = np.array([[x,y],[x-w,y-.5*h],[x-w,y-2*h],[x+w,y-2*h],[x+w,y-.5*h]], np.int32)
		pts = pts.reshape((-1,1,2))

		#cv2.fillConvexPoly(iad,pts,(255,255,255))
		cv2.polylines(iad,[pts],True,(0,0,0), 4)
		#cv2.circle(iad, (e*scale+l+scale/2, t-10 -scale/2), int(scale*.75), fontColor, lineType)

	#cv2.imshow("iad", iad)
	cv2.imwrite(out_name, iad)

	return files[top], salient_frames

def make_graph(top_features, itr_colors, save_name, event_colors, name="graph.png"):
	'''Make a graph from the top features, the edges indicate the ITRs being represented.'''
	dot_name = save_name+'_mygraph.dot'

	gfile = open(dot_name, 'w')

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

		#edges += '{0} -> {1} [label="{2}" color="{3} 1.0 1.0" ]\n'.format(e_to_idx(itr_s[0]), e_to_idx(itr_s[2]), itr_s[1], round(c, 3))
		edges += '{0} -> {1} [label="{2}" ]\n'.format(e_to_idx(itr_s[0]), e_to_idx(itr_s[2]), itr_s[1])


	# add nodes to the dot file

	for e in events:
		nodes += 'node [shape=circle,style=filled,color="{0} 1.0 1.0"] {1}\n'.format(round(event_colors[ e_to_idx(e) ]/360.0, 3),e_to_idx( e ))
	gfile.write(nodes)
	
	# add the edges to the dot files
	gfile.write(edges+"}\n")
	gfile.close()

	# generate image from dot file
	from subprocess import check_call
	check_call(['dot','-Tpng',dot_name,'-o',name])

def determine_feature_ids(dataset_type, dataset_dir, dataset_id, save_name, top_features, depth):
	feature_retain_count = 128

	datatset_type_list = []
	if(dataset_type=="frames" or dataset_type=="both"):
		datatset_type_list.append("frames")
	if(dataset_type=="flow" or dataset_type=="both"):
		datatset_type_list.append("flow")

	#setup feature_rank_parser
	frame_ranking_file = os.path.join( dataset_dir, 'iad_frames_'+str(dataset_id), "feature_ranks_"+str(dataset_id)+".npz") 
	flow_ranking_file = os.path.join( dataset_dir, 'iad_flow_'+str(dataset_id), "feature_ranks_"+str(dataset_id)+".npz") 

	pruning_indexes = {}
	if(dataset_type=="frames"):
		assert os.path.exists(frame_ranking_file), "Cannot locate Feature Ranking file: "+ frame_ranking_file
		pruning_indexes["frames"] = get_top_n_feature_indexes(frame_ranking_file, feature_retain_count)
	elif(dataset_type=="flow"):
		assert os.path.exists(flow_ranking_file), "Cannot locate Feature Ranking file: "+ flow_ranking_file
		pruning_indexes["flow"] = get_top_n_feature_indexes(flow_ranking_file, feature_retain_count)
	elif(dataset_type=="both"):
		assert os.path.exists(frame_ranking_file), "Cannot locate Feature Ranking file: "+ frame_ranking_file
		assert os.path.exists(flow_ranking_file), "Cannot locate Feature Ranking file: "+ flow_ranking_file

		if(save_name == "ucf"):
			if(dataset_id == 1):
				# UCF 1 -> waiting to finish training
				weight_ranking = [[0.177901,0.334655,0.437483,0.801745,0.916997],[0.299762,0.409992,0.519958,0.7917,0.911182]]
			if(dataset_id == 2):
				# UCF 2* - > training
				weight_ranking = [[0.120804,0.24663,0.31483,0.674861,0.842982],[0.208829,0.256675,0.354216,0.557494,0.6894]]
			if(dataset_id == 3):
				# UCF 3* - > training
				weight_ranking = [[0.100449,0.204335,0.270949,0.528152,0.726408],[0.161248,0.187946,0.21438,0.378007,0.473169]]
		if(save_name == "hmdb"):
			if(dataset_id == 1):
				# HMDB 1* - > training
				weight_ranking = [[0.085621,0.201307,0.21634,0.527451,0.675817],[0.190196,0.233333,0.263399,0.296732,0.332026]]
			if(dataset_id == 2):
				# HMDB 2* - > finished
				weight_ranking = [[0.075163,0.145752,0.169281,0.365359,0.566013],[0.131373,0.184314,0.205882,0.282353,0.443791]]
			if(dataset_id == 3):
				# HMDB 3* - > finished
				weight_ranking = [[0.054248,0.127451,0.14183,0.25098,0.462092],[0.10915,0.145098,0.137255,0.138562,0.231373]]
		if(save_name == "bm"):
			if(dataset_id == 1):
				# BLOCKMOVING 1* - > finished
				weight_ranking = [[0.921875,0.9296875,0.9296875,0.6171875,0.578125],[0.828125,0.898438,0.945313,0.945313,0.953125]]
			if(dataset_id == 2):
				# BLOCKMOVING 2* - > finished
				weight_ranking = [[0.765625,0.8125,0.859375,0.671875,0.695313],[0.742188,0.835938,0.898438,0.875,0.835938]]
			if(dataset_id == 3):	
				# BLOCKMOVING 3* - > finished
				weight_ranking = [[0.671875,0.679688,0.742188,0.609375,0.539063],[0.703125,0.71875,0.84375,0.75,0.742188]]
				
		pruning_indexes = get_top_n_feature_indexes_combined(frame_ranking_file, flow_ranking_file, feature_retain_count, weight_ranking)

	# get the top events in the top ITRs
	events = Set()
	for itr in top_features:
		itr_s = itr.split('-')
		events.add(e_to_idx(itr_s[0]))
		events.add(e_to_idx(itr_s[2]))

	print("num events:", len(events))

	# figure out based on pruning how those top events line up with the pruning values
	#print('pruning_indexes["frames"]:', pruning_indexes["frames"][depth])

	feature_dict = {}
	for e in events:
		feature_dict[e] = pruning_indexes["frames"][depth][e]

	return feature_dict

def adjust_gamma(image, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([
      ((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)])
   return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def visualize_example(tcg, ex, sess, input_placeholder, activation_map, feature_dict, depth, event_colors, min_max_vals, salient_frames, itr_colors, out_name="frames.png"):
	
	if(len(salient_frames) == 0):
		return

	isRGB=True

	# process files
	file = ex['raw_path']
	label = ex['label']

	raw_data, length_ratio = model.read_file(file, input_placeholder, isRGB)

	# generate activation map from model
	am = sess.run(activation_map, feed_dict={input_placeholder: raw_data})[depth]

	print(ex['label_name'], "depth:", depth)

	print("activation map.shape:", am.shape)

	
	for row in range(am.shape[-1]):
		if(min_max_vals["max"][depth][row] - min_max_vals["min"][depth][row] == 0):
			am[..., row] = np.zeros_like(am[row])
		else:
			am[..., row] = (am[..., row] - min_max_vals["min"][depth][row]) / (min_max_vals["max"][depth][row] - min_max_vals["min"][depth][row])
	am[am< 0] = 0  

	am *= 255
	scaled_am = am.astype(np.uint8)



	video_length = len(salient_frames)#am.shape[1]
	img_w, img_h = raw_data.shape[2], raw_data.shape[3]

	#separate
	#background = Image.new('RGBA',(img_w*(video_length+1), img_h*len(feature_dict)), (255, 255, 255, 255))

	'''


	draw = ImageDraw.Draw(background)
	font = ImageFont.truetype("arial.ttf", 60)

	for i, e in enumerate(feature_dict.keys()):
		
		draw.text((img_w/2, (i * img_h)+(img_h/2)- 20),str(e),(0,0,0),font=font)
	'''


	#combined
	background = Image.new('RGBA',(img_w*video_length, img_h), (255, 255, 255, 255))
	bg_w, bg_h = background.size

	event_times = [e for e in sorted( tcg.read_file(ex["txt_path"]) ) if e_to_idx( e.name ) in feature_dict]



	for f_idx, frame in enumerate(salient_frames):
	#for frame in range(video_length):
		#f_idx = frame	
		#print("f_idx:", f_idx, frame)

		max_window_scale = [2, 2, 2, 4, 8]

		src = np.copy(raw_data[0, frame*max_window_scale[depth]])
		src = src.astype(np.uint8)
		#src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
		#src = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
		#src[..., 2] = 125
		#src = cv2.cvtColor(src,cv2.COLOR_HSV2BGR)

		#src = adjust_gamma(src, gamma=1.5)

		#b_channel, g_channel, r_channel = cv2.split(src)
		#src = Image.fromarray(cv2.merge((r_channel, g_channel, b_channel, np.ones_like(b_channel)*255 )))
		
		stack = []

		if(54 not in feature_dict):
			return

		for e in [54]:#feature_dict:
			#get spatial info from activation map
			
			
			alpha_channel = am[ 0, frame, ..., feature_dict[e]]


			color_base = np.ones_like(alpha_channel) * 255
			color_base = cv2.cvtColor(color_base,cv2.COLOR_GRAY2BGR)
			color_base = cv2.cvtColor(color_base,cv2.COLOR_BGR2HSV)
			color_base[..., 0] = event_colors[e]
			color_base[..., 1] = 1
			color_base = cv2.cvtColor(color_base,cv2.COLOR_HSV2BGR)
			
			b_channel, g_channel, r_channel = cv2.split(color_base)
			overlay = cv2.merge((r_channel, g_channel, b_channel, alpha_channel))
			
			overlay = cv2.resize( overlay,  (224, 224), interpolation=cv2.INTER_NEAREST)
			overlay = Image.fromarray(overlay.astype(np.uint8))

			stack.append(overlay)
			

			max_point = np.unravel_index(np.argmax(alpha_channel, axis=None), alpha_channel.shape)

			if(f_idx == 5 and e == 54):
				print(alpha_channel)
				print(np.max(alpha_channel))
				print(max_point)
				print(alpha_channel[max_point])

			#alpha_channel = am[ 0, frame, ..., feature_dict[e]]
			#print("max_point:", max_point, alpha_channel.shape)

			scale = src.shape[0]/alpha_channel.shape[0]
			max_point = np.array(list(max_point)) * scale
			#print("max_point:", max_point)

			r = 10
			c = colorsys.hsv_to_rgb((event_colors[e]/360.0), 1.0, 1.0)
			c = tuple([int(255*x) for x in list(c)])

			#print("src:", src.shape)
			#print("max_point:", max_point)
			#print("radius:", r)
			#print("color:", c)

			#print(e)
			#print([e_n.name for e_n in event_times])

			for e_n in event_times:
				if(e_to_idx( e_n.name ) == e):
					if(e_n.start < frame and e_n.end > frame):
						src = cv2.circle(src, tuple(max_point), r, c, 3)

		b_channel, g_channel, r_channel = cv2.split(src)
		src = Image.fromarray(cv2.merge((b_channel, g_channel, r_channel, np.ones_like(b_channel)*255 )))
		
		#print("src:", src[0,0])
		#src = Image.fromarray(src)

		

		for i, s in enumerate(stack):
			
			#combined
			src = Image.alpha_composite(src, s)
			#background.paste(src,(frame * img_w, 0))
			#background.paste(src,((f_idx) * img_w, 0))

			#separate
			#out = Image.alpha_composite(src, s)
			#background.paste(out,(frame * img_w, i * img_h))
			#background.paste(out,((f_idx+1) * img_w, i * img_h))

			#background.paste(src,((f_idx+1) * img_w, 0))

		#src.save(out_name, "PNG")
		background.paste(src,((f_idx) * img_w, 0))


		#src.save("viz_spat.png", "PNG")
		
	background.resize((background.width/2,background.height/2))
	background.save(out_name, "PNG")


def find_video_frames(dataset_dir, file_ex, salient_frames, depth, out_name="frames.png"):
	# create a figure that highlights the frames in the iad

	#salient_frames += [34, 12]

	max_window_scale = [2, 2, 2, 4, 8]
	img_files = [os.path.join(dataset_dir, 'frames', file_ex['label_name'], file_ex['example_id'], 'image_'+str(frame_num*max_window_scale[depth]+3).zfill(5)+'.jpg') for frame_num in salient_frames]

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	fontScale              = 4
	fontColor              = (0,0,0)
	lineType               = 8

	tall_img = []
	
	
	big_img = cv2.imread(img_files[0])
	big_img = cv2.resize(big_img, (1920, 1080))

	radius = 100
	pos = (big_img.shape[1]/2 -35 ,110)
	txt_pos = (pos[0]-40, pos[1]+30)
	i = 0

	
	#cv2.square(big_img, pos, radius, (255,255,255), -1)
	#cv2.square(big_img, pos, radius, fontColor, lineType)

	x, y = pos[0], pos[1]+100
	w, h = 100, 100

	pts = np.array([[x,y],[x-w,y-.5*h],[x-w,y-2*h],[x+w,y-2*h],[x+w,y-.5*h]], np.int32)
	pts = pts.reshape((-1,1,2))

	cv2.fillConvexPoly(big_img,pts,(255,255,255))
	cv2.polylines(big_img,[pts],True,(0,0,0), lineType)

	cv2.putText(big_img,str(i), 
		txt_pos, 
		font, 
		fontScale,
		fontColor,
		16)
	i+=1
	

	for file in img_files[1:]:

		img = cv2.imread(file)
		img = cv2.resize(img, (1920, 1080))
		#cv2.circle(img, pos, radius, (255,255,255), -1)
		#cv2.circle(img, pos, radius, fontColor, lineType, )
		cv2.fillConvexPoly(img,pts,(255,255,255))
		cv2.polylines(img,[pts],True,(0,0,0), lineType)

		cv2.putText(img,str(i), 
			txt_pos, 
			font, 
			fontScale,
			fontColor,
			16)
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

def main(dataset_dir, csv_filename, dataset_type, dataset_id, num_classes, save_name, model_filename):

	dir_root = os.path.join("pics", save_name)


	isRGB=True
	batch_size=1
	pad_length=256

	# define placeholder
	input_placeholder = model.get_input_placeholder(isRGB, batch_size, num_frames=pad_length)
	
	# define model
	activation_map, _, saver = model.load_model(input_placeholder, isRGB)
	#print("rank3", rankings[0].get_shape())
	
	#collapse the spatial dimensions of the activation map
	#for layer in range(len(activation_map)):
	#	activation_map[layer] = tf.argmax(activation_map[layer], axis = (2,3))
	min_max_file = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id), "min_maxes.npz")
	f = np.load(min_max_file, allow_pickle=True)
	min_max_vals = {"max": f["max"],"min": f["min"]}


	with tf.Session() as sess:

		# Restore model
		sess.run(tf.global_variables_initializer())
		tf_utils.restore_model(sess, saver, model_filename)

		# prevent further modification to the graph
		sess.graph.finalize()
	

		for depth in range(2,3):#5):

			dir_name = os.path.join(dir_root, str(depth))

			if(not os.path.exists(dir_name)):
				os.makedirs(dir_name)

			#open files
			try:
				csv_contents = read_csv(csv_filename)
			except:
				print("ERROR: Cannot open CSV file: "+ csv_filename)

			for ex in csv_contents:
				ex['raw_path'] = os.path.join(dataset_dir, 'frames', ex['label_name'], ex['example_id'])
				ex['iad_path'] = os.path.join(dataset_dir, 'iad_'+dataset_type+'_'+str(dataset_id), ex['label_name'], ex['example_id']+"_"+str(depth)+".npz")
				ex['txt_path'] = os.path.join(dataset_dir, "txt_"+dataset_type+"_"+str(dataset_id), str(depth), ex['label_name'], ex['example_id']+'_'+str(depth)+'.txt')
			csv_contents = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id and ex['dataset_id'] != 0]
			#csv_contents = [ex for ex in csv_contents if ex['dataset_id'] == 0]


			# open saved model
			save_file = os.path.join(save_name, str(dataset_id), dataset_type)
			filename = save_file.replace('/', '_')+'_'+str(depth)
			tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

			label_targets = ["equals", "starts"]
			label_ids = [ex['label'] for ex in csv_contents if ex['label_name'] in label_targets]

			label_ids = list(Set(label_ids))

			print(label_ids)

			for label in label_ids:#num_classes):

				label_name = [ex for ex in csv_contents if ex['label'] == label][0]['label_name']


				# generate a plot that shows the top 5 and bottom five features for each label.
				feat_name = os.path.join(dir_name, label_name+'_feat_'+str(depth)+'.png')
				title = label_name.upper().replace('_', ' ')+", Depth "+str(depth)
				top_features, colors, event_colors = generate_top_bottom_table(tcg, label, count=5, out=feat_name, title=title)

				graph_name = os.path.join(dir_name, label_name+'_graph_'+str(depth)+'.png')
				make_graph(top_features, colors, save_name, event_colors, name=graph_name)

				feature_dict = determine_feature_ids(dataset_type, dataset_dir, dataset_id, save_name, top_features, depth)

				iad_name = os.path.join(dir_name, label_name+'_iad_'+str(depth)+'.png')
				file_ex, salient_frames = find_best_matching_IAD(tcg, label, top_features, colors, csv_contents, out_name=iad_name)
				
				frames_name = os.path.join(dir_name, label_name+'_frames_'+str(depth)+'.png')

				'''
				if(len(salient_frames) > 0):
					find_video_frames(dataset_dir, file_ex, salient_frames, depth, out_name=frames_name)

				print('----------------')
				'''

				visualize_example(tcg, file_ex, sess, input_placeholder, activation_map, feature_dict, depth, event_colors, min_max_vals, salient_frames, colors, out_name=frames_name)

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

	parser.add_argument('save_name', default="", help='what to save the model as')

	parser.add_argument('model_filename', default="", help='I3D model')
	


	FLAGS = parser.parse_args()

	#i=2

	#for i in range(5):


	main(FLAGS.dataset_dir, 
		FLAGS.csv_filename,
		FLAGS.dataset_type,
		FLAGS.dataset_id,
		FLAGS.num_classes,
		FLAGS.save_name,
		FLAGS.model_filename,
		)
