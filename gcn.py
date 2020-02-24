import tensorflow as tf 





from sets import Set
import os, sys, math, time
import numpy as np
from collections import Counter

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

import matplotlib
import matplotlib.pyplot as plt

from itr_sklearn import ITR_Extractor

import torch


def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 10


		tcg = ITR_Extractor(num_classes)		
		
		#open files
		try:
			csv_contents = read_csv(csv_filename)
		except:
			print("ERROR: Cannot open CSV file: "+ csv_filename)

		path = 'b_path_{0}'.format(layer)
		for ex in csv_contents:
			ex[path] = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), '{0}_{1}.b'.format(ex['example_id'], layer))

		train_data = [ex for ex in csv_contents if ex['dataset_id'] >= dataset_id]
		test_data  = [ex for ex in csv_contents if ex['dataset_id'] == 0]

		train_data = [ex for ex in train_data if ex['label'] < num_classes]
		test_data = [ex for ex in train_data if ex['label'] < num_classes]

		
		save_dir = os.path.join(dataset_dir, 'svm_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id))
		if (not os.path.exists(save_dir)):
			os.makedirs(save_dir)

		# TRAIN
		for i, ex in enumerate(train_data):
			if(i%1000 == 0):
				print("adding data...{0}/{1}".format(i, len(train_data)))
			tcg.add_file_to_corpus(ex[path], ex['label'])
		print("fitting model...")
		print("len(tcg.corpus):", len(tcg.corpus))

		batch_size = 4

		data_in = np.array(tcg.tfidf.fit_transform(tcg.corpus).toarray())
		data_label = np.array(tcg.labels)

		trainloader = torch.utils.data.DataLoader(zip(data_in, data_label), batch_size=batch_size,
										  shuffle=True, num_workers=2)


		# CLASSIFY 
		for ex in test_data:
			tcg.add_file_to_eval_corpus(ex[path], ex['label'], ex['label_name'])

		eval_in = np.array(tcg.tfidf.transform(tcg.evalcorpus).toarray())
		eval_label = np.array(tcg.evallabels)

		testloader = torch.utils.data.DataLoader(zip(eval_in, eval_label), batch_size=batch_size,
										 shuffle=False, num_workers=2)


		
		print("data_in.shape", data_in.shape)
		print("data_label.shape", data_label.shape)

		#model
		import torch.nn as nn
		class Net(nn.Module):
			def __init__(self, input_size, num_classes):
				super(Net, self).__init__()
				self.dense = nn.Linear(input_size, num_classes)				

			def forward(self, x):
				return self.dense(x).double()

		net = Net(data_in.shape[1], num_classes)

		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

		t_s = time.time()
		for epoch in range(2):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				#print(inputs.shape, labels.shape)
				#print(inputs.dtype, labels.dtype)



				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs.float())
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
				if i % 2000 == 1999:    # print every 2000 mini-batches
					print('[%d, %5d] loss: %.3f' %
						  (epoch + 1, i + 1, running_loss / 2000))
					running_loss = 0.0


		print("train elapsed:", time.time()-t_s)
			

		print("evaluating model...")
		t_s = time.time()
		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				images, labels = data
				outputs = net(images.float())
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		print("test elapsed:", time.time()-t_s)

		print('Accuracy of the network on the 10000 test images: %d %%' % (
			100 * correct / total))

		# if model accuracy is good then replace the old model with new save data
		#if(cur_accuracy > max_accuracy):
		#	tcg.save_model(os.path.join(save_dir, "model"))
		#	max_accuracy = cur_accuracy

		#print("Training layer: {:d}, iter: {:d}/{:d}, acc:{:0.4f}, max_acc: {:0.4f}".format(layer, iteration, repeat, cur_accuracy, max_accuracy))
		

if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Generate IADs from input files')
	#required command line args
	parser.add_argument('model_type', help='the type of model to use', choices=['i3d', 'trn', 'tsm'])

	parser.add_argument('dataset_dir', help='the directory where the dataset is located')
	parser.add_argument('csv_filename', help='a csv file denoting the files in the dataset')
	parser.add_argument('dataset_type', help='the dataset type', choices=['frames', 'flow', 'both'])
	parser.add_argument('dataset_id', type=int, help='a csv file denoting the files in the dataset')
	parser.add_argument('num_classes', type=int, help='the number of classes in the dataset')

	parser.add_argument('--num_procs', type=int, default=1, help='number of process to split IAD generation over')
	parser.add_argument('--repeat', type=int, default=1, help='number of times to repeat training the model')


	FLAGS = parser.parse_args()

	if(FLAGS.model_type == 'i3d'):
		from gi3d_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'trn'):
		from trn_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT
	if(FLAGS.model_type == 'tsm'):
		from tsm_wrapper import DEPTH_SIZE, CNN_FEATURE_COUNT


	for layer in range(2):#DEPTH_SIZE):
		main(FLAGS.model_type,
			FLAGS.dataset_dir, 
			FLAGS.csv_filename,
			FLAGS.dataset_type,
			FLAGS.dataset_id,
			layer,
			FLAGS.num_classes,
			FLAGS.repeat
			)
	
	
