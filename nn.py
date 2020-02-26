import tensorflow as tf 





from sets import Set
import os, sys, math, time
import numpy as np

sys.path.append("../IAD-Generator/iad-generation/")
from csv_utils import read_csv

from itr_sklearn import ITR_Extractor

import torch


def main(model_type, dataset_dir, csv_filename, dataset_type, dataset_id, layer, num_classes, repeat=1, parse_data=True):

	max_accuracy = 0

	for iteration in range(repeat):
		print("Processing depth: {:d}, iter: {:d}/{:d}".format(layer, iteration, repeat))
	
		num_classes = 10


		tcg = ITR_Extractor(num_classes, num_procs=4)		
		
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

		train_filename = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'train_{0}_{1}.npz'.format(ex['example_id'], layer))
		test_filename  = os.path.join(dataset_dir, 'b_{0}_{1}_{2}'.format(model_type, dataset_type, dataset_id), 'test{0}_{1}.npz'.format(ex['example_id'], layer))

		parse_data = not os.path.exists(train_filename)

		if(parse_data):
			# TRAIN
			in_files = [ex[path] for ex in train_data]
			in_labels = [ex['label'] for ex in train_data]

			print("adding train data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))


			data_in = np.array(tcg.tfidf.fit_transform(tcg.corpus).toarray())
			data_label = np.array(tcg.labels)

			np.savez_compressed(train_filename, data=data_in, label=data_label)


			in_files = [ex[path] for ex in test_data]
			in_labels = [ex['label'] for ex in test_data]
			print("adding eval data...{0}".format(len(train_data)))
			t_s = time.time()
			tcg.add_files_to_eval_corpus(in_files, in_labels)
			print("data added. time: {0}".format(time.time() - t_s))

			eval_in = np.array(tcg.tfidf.transform(tcg.evalcorpus).toarray())
			eval_label = np.array(tcg.evallabels)

			np.savez_compressed(test_filename, data=eval_in, label=eval_label)

		else:
			f = np.load(train_filename)
			data_in, data_label = f["data"], f["label"]
			f = np.load(test_filename)
			eval_in, eval_label = f["data"], f["label"]


		
		print("fitting model...")
		print("data_in.shape", data_in.shape)
		print("data_label.shape", data_label.shape)


		batch_size = 10

		trainloader = torch.utils.data.DataLoader(zip(data_in, data_label), batch_size=batch_size,
										  shuffle=True, num_workers=2)

		testloader = torch.utils.data.DataLoader(zip(eval_in, eval_label), batch_size=batch_size,
										 shuffle=False, num_workers=2)


		

		if torch.cuda.is_available():
		    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
		    print("Running on the GPU")
		else:
		    device = torch.device("cpu")
		    print("Running on the CPU")


		#model
		import torch.nn as nn
		class Net(nn.Module):
			def __init__(self, input_size, num_classes):
				super(Net, self).__init__()
				n_hidden= 1024

				self.dense1 = nn.Linear(input_size, n_hidden)
				self.dense2 = nn.Linear(n_hidden, num_classes)				

			def forward(self, x):
				x = self.dense1(x)#.double()
				return self.dense2(x)#.double()

		net = Net(data_in.shape[1], num_classes).to(device)




		import torch.optim as optim

		criterion = nn.CrossEntropyLoss()
		optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

		t_s = time.time()
		for epoch in range(5):  # loop over the dataset multiple times

			running_loss = 0.0
			for i, data in enumerate(trainloader, 0):
				# get the inputs; data is a list of [inputs, labels]
				inputs, labels = data

				inputs = inputs.to(device).float()
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(inputs)
				loss = criterion(outputs, labels)
				loss.backward()
				optimizer.step()

				# print statistics
				running_loss += loss.item()
			print('[%d, %5d] loss: %.3f' % (epoch + 1, len(trainloader), running_loss / 2000))
			running_loss = 0.0


		print("train elapsed:", time.time()-t_s)
			

		print("evaluating model...")
		t_s = time.time()
		correct = 0
		total = 0
		with torch.no_grad():
			for data in testloader:
				inputs, labels = data

				inputs = inputs.to(device).float()
				labels = labels.to(device)

				outputs = net(inputs)

				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				for p, l in zip(predicted, labels):
					print(p, l)
				print(correct, total)
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
	parser.add_argument('--parse_data', type=bool, default=True, help='whether to parse the data again or load from file')


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
	
	
