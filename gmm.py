import os
import sys

import numpy as np
import pandas as pd

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class GMM():
    def __init__(self, n_components=2, num_iters=200, init='kmeans'):
        self.num_means = n_components
        self.max_iters = num_iters

        self.theta = {}
        self._init = init

    def _para_init(self, X, method="kmeans"):  # Kmeans for now, allow more choices later
        ''' Initializes theta according to the method
        Args:
            X : n x d
            method : str
        '''
        theta = {
            'num_dims': X.shape[1],
            'a': np.array([1 / self.num_means] * self.num_means),
            'mu': np.zeros(self.num_means),
            'cov': np.array([np.diag(np.ones(X.shape[1]))] * self.num_means),
        }
        if method == "kmeans":
            # update self.theta
            km = KMeans(n_clusters=self.num_means)
            km.fit(X)
            keys = np.array([x**2 + y**2 for x, y in km.cluster_centers_])
            order = keys.argsort()
            theta['a'] = (np.unique(km.labels_, return_counts=True)[
                          1] / len(X))[order]
            theta['mu'] = km.cluster_centers_[order]
        elif method == "random":
            means = np.random.rand(self.num_means, theta['num_dims']) * 20 - 10
            keys = np.array([x**2 + y**2 for x, y in means])
            order = keys.argsort()
            theta['mu'] = means[order]
        # theta = self._update_cdf_major(X, theta)
        self.theta = theta

    def _get_prob(self, Z, theta):
        ''' Returns h (probability of each data point belonging to a certain cluster)
        Args:
            Z : n x d data input
            theta :
                a : M x 1
                mu : M x d
                cov : M x d x d
        Returns:
            probability : M x n
        '''
        n, d = Z.shape
        # gaussians: M x n
        gaussians = np.array([[mvn.pdf(Z[i], mean=theta['mu'][k], cov=theta['cov'][k],
                                       allow_singular=True) for i in range(n)] for k in range(self.num_means)])
        numerator = (theta['a'] * gaussians.T).T
        denominator = np.sum(numerator, axis=0)
        probability = numerator / denominator
        return probability

    def _cov_matrix(self, X, mu, prob_col):  # 1 x d x d
        ''' Returns the covariance matrix for Z on prob_col and mu
        Args:
            X : n x d data input
            mu : 1 x d mean for a certain cluster
            prob_col: probability that each point in Z belongs to this certain cluster
        Returns:
            cov : d x d
        '''
        # Z : nxd, mu : dx1 (bc numpy), Z - mu : nxd, prob_col : nx1
        mat = (np.sqrt(prob_col) * (X - mu).T).T
        return mat.T @ mat

    # EM for general gaussian mixture
    def _GM_EM_iter(self, X, theta):  # keep a general GM-EM-iter implementation,
        ''' Steps theta on X
        Args:
            X : n x d data input
            theta :
                a : M x 1
                mu : M x d
                cov : M x d x d
        Returns:
            new_theta
        '''
        # plt.scatter(X[:, 0], X[:, 1], c=self.predict(X, theta))
        # plt.show()
        n = X.shape[0]
        prob = self._get_prob(X, theta)  # M x n
        # print('prob', prob)
        prob_mass = np.sum(prob, axis=1)  # 1 x M
        a = prob_mass / n  # h is M by N
        # removed denominator since it sums to 1
        mus = ((prob @ X).T / prob_mass).T
        sigmas = (np.array([self._cov_matrix(X, mus[k], prob[k])
                            for k in range(self.num_means)]).T / prob_mass).T  # M x d x d
        new_theta = {
            'num_dims': X.shape[1],
            'a': a,
            'mu': mus,
            'cov': sigmas,
        }
        # new_theta = self._update_cdf_major(X, new_theta)
        return new_theta

    def _gmm_likelihood(self, X, theta):
        n, d = X.shape
        return np.sum([
            np.log(np.sum([
                theta['a'][k] *
                mvn.pdf(X[i],
                        mean=theta['mu'][k],
                        cov=theta['cov'][k])
                for k in range(self.num_means)
            ])) for i in range(n)
        ])

    def fit(self, X, eps=1e-4, max_iters=1e3, init=None):
        if not init is None:
            self._para_init(X, method=init)
        else:
            self._para_init(X, method=self._init)  # initialize theta on kmeans
        max_likelihood = float('-inf')
        prev_likelihood = max_likelihood
        theta_new = self.theta.copy()
        # theta_new = self._update_cdf_major(X, theta_new)
        for i in range(self.max_iters):
            # print('=== ITER', i, '===')
            theta_new = self._GM_EM_iter(X, theta_new)
            # print('theta new', theta_new)
            # find pseudolikelihood
            # if likelihood is greater than previous max
            #     set self.theta to current theta
            likelihood = self._gmm_likelihood(X, theta_new)
            # print('likelihood', likelihood)
            # print("MONOTONIC:", likelihood > max_likelihood)
            if likelihood > max_likelihood:
                self.theta = theta_new.copy()
                max_likelihood = likelihood
            else:
                # print('NONMONOTONIC!!!!!')
                break
            # try different stopping criteria
            if np.abs(likelihood - prev_likelihood)/np.abs(prev_likelihood) < eps:
                break
            prev_likelihood = likelihood

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X, theta=None):
        if theta is None:
            theta = self.theta
        h = self._get_prob(X, theta)
        preds = np.argmax(h.T, axis=1)
        return preds
