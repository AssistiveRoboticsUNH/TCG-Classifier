import matplotlib
matplotlib.use('Agg')

a = """(0, array([0]))
(0, array([0]))
(0, array([0]))
(0, array([0]))
(0, array([0]))
(0, array([0]))
(0, array([0]))
(0, array([0]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(1, array([1]))
(2, array([2]))
(2, array([2]))
(2, array([2]))
(2, array([2]))
(2, array([2]))
(2, array([2]))
(2, array([2]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(3, array([3]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(4, array([4]))
(5, array([5]))
(5, array([5]))
(5, array([5]))
(5, array([5]))
(5, array([5]))
(5, array([5]))
(5, array([5]))
(6, array([6]))
(4, array([6]))
(6, array([6]))
(6, array([6]))
(4, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(6, array([6]))
(4, array([6]))
(6, array([6]))
"""

a = a.replace('(', '') \
	.replace(')', '') \
	.replace('array', '') \
	.replace('[', '') \
	.replace(']', '') \
	.split('\n')[:-1]



a = [x.split(',') for x in a]
print(a)



import numpy as np

a = np.array([[int(x[0]), int(x[1])] for x in a])
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix


# import some data to play with


# Split the data into a training set and a test set


# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results


# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
				  ("Normalized confusion matrix", 'true')]

labels = [
"add_milk",
"add_sugar", 
"add_tea_bag", 
"add_water",
"nothing",
"stir",
"toggle_on_off"]



def plot_confusion_matrix(cm,
						  target_names,
						  title='Confusion matrix',
						  cmap=None,
						  normalize=True):
	"""
	given a sklearn confusion matrix (cm), make a nice plot

	Arguments
	---------
	cm:           confusion matrix from sklearn.metrics.confusion_matrix

	target_names: given classification classes such as [0, 1, 2]
				  the class names, for example: ['high', 'medium', 'low']

	title:        the text to display at the top of the matrix

	cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
				  see http://matplotlib.org/examples/color/colormaps_reference.html
				  plt.get_cmap('jet') or plt.cm.Blues

	normalize:    If False, plot the raw numbers
				  If True, plot the proportions

	Usage
	-----
	plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
															  # sklearn.metrics.confusion_matrix
						  normalize    = True,                # show proportions
						  target_names = y_labels_vals,       # list of names of the classes
						  title        = best_estimator_name) # title of graph

	Citiation
	---------
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

	"""
	

	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

	accuracy = np.trace(cm) / float(np.sum(cm))
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	plt.savefig("tea_making_cm.png")




cm = confusion_matrix(y_true=a[1], y_pred=a[0])

plot_confusion_matrix(cm, labels)



