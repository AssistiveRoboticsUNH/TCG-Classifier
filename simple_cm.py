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
from sklearn.metrics import plot_confusion_matrix
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


confusion_matrix(a[1], a[0], labels)

for title, normalize in titles_options:


    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.savefig("tea_making_cm.png")


