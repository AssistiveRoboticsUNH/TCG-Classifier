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
			
	# trim the front of the IAD
	iad = iad[:, 3:]

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
		#c = itr_colors[itr]/360.0

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

def visualize_example(ex, sess, input_placeholder, activation_map, feature_dict, depth, event_colors, min_max_vals):
	
	isRGB=True

	# process files
	file = ex['raw_path']
	label = ex['label']

	raw_data, length_ratio = model.read_file(file, input_placeholder, isRGB)

	# generate activation map from model
	am = sess.run(activation_map, feed_dict={input_placeholder: raw_data})[depth]

	print("activation map.shape:", am.shape)

	for row in range(am.shape[-1]):
		if(min_max_vals["max"][depth][row] - min_max_vals["min"][depth][row] == 0):
			am[..., row] = np.zeros_like(am[row])
		else:
			am[..., row] = (am[..., row] - min_max_vals["min"][depth][row]) / (min_max_vals["max"][depth][row] - min_max_vals["min"][depth][row])

	am *= 255
	scaled_am = am.astype(np.uint8)

	video_length = 1

	for frame in range(video_length):
		src = np.copy(raw_data[0, frame])
		src = src.astype(np.uint8)

		stack = []

		for e in feature_dict:
			#get spatial info from activation map
			feature_frame = scaled_am[ ..., feature_dict[e]]

			# color overlay according to feature
			overlay = cv2.cvtColor(feature_frame,cv2.COLOR_GRAY2BGR)
			overlay = cv2.cvtColor(overlay,cv2.COLOR_BGR2HSV)
			overlay[..., 0] = event_colors[e]
			#ovl[..., 0] = 1

			overlay = cv2.resize( overlay,  (224, 224), interpolation=cv2.INTER_NEAREST)
			#overlay = (src + overlay)/2

			overlay = cv2.cvtColor(overlay,cv2.COLOR_HSV2BGR)

			stack.append(overlay)


		alpha = 0.5
		print("src.shape:", src.shape)
		for s in stack:
			print("s.shape:", s.shape)
		cv2.addWeighted(src, alpha, s[0], 1 - alpha, 0, src)
		cv2.addWeighted(src, alpha, s[1], 1 - alpha, 0, src)


	cv2.imwrite("viz_spat.png", cv2.cvtColor(src, cv2.COLOR_RGB2BGR))






	'''
	print("am.shape:", am.shape)
	
	important_am = []
	for e in feature_dict:
		print(e, feature_dict[e])
		print(feature_dict[e])
		important_am.append( am[ ..., feature_dict[e] ] )


	print("am.shape:")
	print(important_am[0].shape)

	#num_frames = len(important_am[0][:,1])

	#max_window_scale = [2, 2, 2, 4, 8]
	length = 10

	print("raw_data.shape:", raw_data.shape)
	src = raw_data[0, 0]
	src = src.astype(np.uint8)

	feat_map = important_am[0][0][0]
	feat_map -= np.min(feat_map)
	feat_map /= np.max(feat_map)
	feat_map *= 255
	feat_map = feat_map.astype(np.uint8)

	ovl = cv2.resize( feat_map,  (224, 224), interpolation=cv2.INTER_NEAREST)

	ovl = cv2.cvtColor(ovl,cv2.COLOR_GRAY2BGR)
	ovl = cv2.cvtColor(ovl,cv2.COLOR_BGR2HSV)
	ovl[..., 0] = event_colors[]

	print("ovl:", ovl)
	ovl[..., 1:3] = 0
	print("ovl.shape:", ovl.shape)
	
	alpha=0.5
	cv2.addWeighted(src, alpha, ovl, 1 - alpha, 0, ovl)
	'''
	#cv2.imwrite("viz_spat.png", cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR))
	
		


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
	

		for depth in range(4,5):

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

			# open saved model
			save_file = os.path.join(save_name, str(dataset_id), dataset_type)
			filename = save_file.replace('/', '_')+'_'+str(depth)
			tcg = ITR_Extractor(num_classes, os.path.join(save_file, filename))

			for label in range(1):#num_classes):

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
				
				'''
				if(len(salient_frames) > 0):
					frames_name = os.path.join(dir_name, label_name+'_frames_'+str(depth)+'.png')
					find_video_frames(dataset_dir, file_ex, salient_frames, depth, out_name=frames_name)

				print('----------------')
				'''

				visualize_example(file_ex, sess, input_placeholder, activation_map, feature_dict, depth, event_colors, min_max_vals)

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
