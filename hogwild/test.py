import unittest
from hogwildsgd import HogWildRegressor
from hogwild import HogWildClassifier
import scipy.sparse
import numpy as np
from sklearn.datasets import make_classification


'''
class TestHogwild(unittest.TestCase):

    def test_work(self):
        X = scipy.sparse.random(20000,10, density=.2).toarray() # Guarantees sparse grad updates
        real_w = np.random.uniform(0,1,size=(10,1))
        y = np.dot(X,real_w)


        hw = HogWildRegressor(n_jobs = 4, 
                              n_epochs = 5,
                              batch_size = 1, 
                              chunk_size = 32,
                              learning_rate = .001,
                              generator=None,
                              verbose=2)
        hw = hw.fit(X,y)


        y_hat = hw.predict(X)
        y = y.reshape((len(y),))
        score = np.mean(abs(y-y_hat))
        self.assertTrue(score < .005) 
'''
class TestHogwild(unittest.TestCase):

    def test_work(self):
        #X, y = make_classification(n_samples=20000, n_features=5, n_classes=2)
        #y[ y == 0] = -1
        X = np.array([[-1,-1], [1,-1], [-1,1], [1,1]])
        y = np.array([1,1,-1,-1])


        print("X:", X)
        print("y:", y)



        hw = HogWildClassifier(n_jobs = 4, 
                              n_epochs = 5,
                              batch_size = 1, 
                              chunk_size = 1,#32,
                              learning_rate = .1,#.001,
                              generator=None,
                              verbose=2)
        hw = hw.fit(X,y)


        y_hat = hw.predict(X)
        print("y:", y)
        print("y_hat:", y_hat)

        from sklearn.metrics import accuracy_score
        y = y.reshape((len(y),))
        score = accuracy_score(y_true =y, y_pred=y_hat)#np.mean(abs(y-y_hat))

        print("score:", score)

        self.assertTrue(score < .005) 


if __name__ == '__main__':
    unittest.main()