import logging
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import DotProduct, RBF

class SVCT(BaseEstimator):
    """
    2-phase SVC for SASSE ERA5 polygon classification
    """

    def __init__(self, args1={'kernel': 'rbf', 'probability': True}, args2={'kernel': DotProduct(), 'probability': True}, verbose=False, bayesian_prediction=False):
        """
        ...
        """
        self.model1 = SVC(**args1)
        self.model2 = SVC(**args2)

        self.verbose=verbose
        self.bayesian_prediction = bayesian_prediction

    def fit(self, X, y):
        """
        Empty function, logical model doesn't need any training
        """

        X1 = X
        y1 = y.copy()
        y1[(y1>0)] = 1

        X2 = X[(y > 0)]
        y2 = y[(y > 0)]

        if self.verbose:
            logging.info('Fitting model 1...')
        self.model1.fit(X1, y1)

        if self.verbose:
            logging.info('Fitting model 2...')
        self.model2.fit(X2, y2)

        return self


    def predict(self, X):
        """
        Predict
        """

        #y_pred_proba = self.predict_proba(X)
        #return np.argmax(y_pred_proba, axis=1)

        # Alternative, more straight forward method
        y1_ = self.model1.predict(X)
        X2_ = X[(y1_>0)]
        y2_ = self.model2.predict(X2_)
        y1_[(y1_ > 0)] = y2_

        return y1_

    def predict_proba(self, X):
        """
        Predict with probabilities
        """

        yp1_ = self.model1.predict_proba(X)
        yp2_ = self.model2.predict_proba(X)

        y_pred_proba = np.zeros((len(X), 3))

        # If use bayesian, predict classes 1 and 2 as bayesian probability,
        # else pick probablities directly from model outputs

        if self.bayesian_prediction:
            # TODO use likelihoods
            y_pred_proba[:,0] = yp1_[:,0]
            y_pred_proba[:,1] = self.bayes(yp1_[:,1], yp2_[:,0], (1-yp1_[_:,0]))
            y_pred_proba[:,2] = self.bayes(yp1_[:,1], yp2_[:,1], (1-yp1_[_:,0]))
        else:
            y_pred_proba[:,0] = yp1_[:,0]
            y_pred_proba[:,1] = np.where(yp1_[:,1] >=.5, yp2_[:,0], yp1_[:,1])
            y_pred_proba[:,2] = np.where(yp1_[:,1] >=.5, yp2_[:,1], yp1_[:,1])

        return y_pred_proba

    def bayes(self, p_a, p_b_given_a, p_b_given_not_a):
        # calculate P(not A)
        not_a = 1 - p_a
        # calculate P(B)
        p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
        # calculate P(A|B)
        p_a_given_b = (p_b_given_a * p_a) / p_b
        return p_a_given_b
