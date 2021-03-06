import sys
from types import ModuleType
from multiprocessing.sharedctypes import Array
from ctypes import c_double
import numpy as np

import scipy

temp_module_name = '__hogwildsgd__temp__'


class SharedWeights:
    """ Class to create a temporary module with the gradient function inside
        to allow multiprocessing to work for async weight updates.
    """
    
    def __init__(self, size_w, num_classes):

        print("size_w:", size_w)
        print("num_classes:", num_classes)

        comb = scipy.special.comb(num_classes, 2)
        self.num_classes = num_classes

        coef_shared = Array(c_double, 
                (np.random.normal(size=(comb, size_w)) * 1./np.sqrt(size_w)).flat,
                lock=False)

        w = np.frombuffer(coef_shared)
        w = w.reshape((comb, len(w))) 
        self.w = w

    '''
    def __init__(self, size_w):

        print("size_w:", size_w)



        coef_shared = Array(c_double, 
                (np.random.normal(size=(size_w,1)) * 1./np.sqrt(size_w)).flat,
                lock=False)

        w = np.frombuffer(coef_shared)
        w = w.reshape((len(w),1)) 
        self.w = w

        #print("init Shared Weights")
    '''
    def __enter__(self, *args):
        # Make temporary module to store shared weights
        print("enter Shared Weights")

        mod = ModuleType(temp_module_name)
        mod.__dict__['w'] =  self.w

        #print("mod.__name__:", mod.__name__)
        sys.modules[mod.__name__] = mod    
        self.mod = mod    
        return self
    
    def __exit__(self, *args):
        # Clean up temporary module
        print("exit Shared Weights")
        del sys.modules[self.mod.__name__]         


def mse_gradient_step(X, y, learning_rate, shared_w):
    """ Gradient for mean squared error loss function. """
    #print("shared_w2:", shared_w)
    #for k in sorted(shared_w.__dict__.keys()):
    #    print(k)

    #w = sys.modules[temp_module_name].__dict__['w']
    w = shared_w.w

    # Calculate gradient
    err = y.reshape((len(y),1))-np.dot(X,w)
    grad = -2.*np.dot(np.transpose(X),err)/ X.shape[0]

    for index in np.where(abs(grad) > .01)[0]:
         w[index] -= learning_rate*grad[index,0]

def hinge_gradient_step(X, y, learning_rate, shared_w, thresh = 1.0):
    """ Gradient for mean squared error loss function. """

    #w = sys.modules[temp_module_name].__dict__['w']
    w = shared_w.w

    #print("np.dot(X,w):", np.dot(X,w))
    #print("y:", y)

    z = np.dot(X,w) * y
    #z = np.dot(y,X*w) 
    #print("z:", z)
    if z < thresh:
        grad = -y*X#-np.dot(np.transpose(X), y)#-y

        #print("indexes:", np.where(abs(grad) > .01))
        for index in np.where(abs(grad) > .01)[1]:
            #  print("index:", index)

            w[index] -= learning_rate*grad[0, index]
        #print("post grad:", w)
        #print("np.dot(X,w):", np.dot(X,w))

    #print('--------')

