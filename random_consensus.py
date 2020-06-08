import random
from abc import ABC, abstractmethod

import numpy as np
import scipy.special


class Model(ABC):
    @abstractmethod
    def fit(self, pts):
        ...

    @abstractmethod
    def error(self, data):
        ...

    @abstractmethod
    def predict(self, data):
        ...

    @staticmethod
    @abstractmethod
    def get_complexity():
        ...


def RANSAC(data, modelType, tolerance, prob_inlier, min_ratio_correct_model):
    m = modelType.get_complexity()
    best_num_inliers = 0
    n = data.shape[0]
    max_times = int(np.ceil(1 / (prob_inlier ** m)))
    inliers = []
    for _ in range(max_times):
        pts = data[random.sample(range(n), m)]
        model = modelType()
        model.fit(pts)
        error = model.error(data)
        num_inliers = (error < tolerance).sum()
        if num_inliers / n > min_ratio_correct_model:
            inliers = data[error < tolerance]
            break
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            inliers = data[error < tolerance]

    model = modelType()
    model.fit(inliers)

    return model


def PROSAC(data, quality, modelType, tolerance, beta, phi=0.05, eta=0.05):
    indexes = np.argsort(quality)
    data = data[indexes[::-1]]

    N = data.shape[0]
    m = modelType.get_complexity()
    t = 0
    n = m
    n_star = N
    Tn = 1
    Tn_prime = 1
    while True:
        # 1. Choice of the hypothesis generation set
        t += 1
        if t == Tn_prime and n < n_star:
            Tn_1 = Tn * (n + 1) / (n + 1 - m)
            Tn_prime = Tn_prime + np.ceil(Tn_1 - Tn)
            Tn = Tn_1
            n = min(n + 1, N)
        # 2. Semi-random sample M of size m
        if t < Tn_prime:
            pts_idx = [n] + random.sample(range(n - 1), m - 1)
        else:
            pts_idx = random.sample(range(n), m)
        sample = data[pts_idx]
        # 3. Model parameter estimation
        model = modelType()
        model.fit(sample)

        # 4. Model verification
        error = model.error(data)
        num_inliers = (error < tolerance).sum()

        # 4.1 Non random solution
        def P(i):
            return (beta ** (i - m)) * ((1 - beta) ** (n - i + m)) * scipy.special.binom(n - m, i - m)

        def Imin_value(j):
            return sum(P(i) for i in range(j, n + 1))

        imin = 0
        for j in range(m, N - m):
            value = Imin_value(j)
            if value < phi:
                imin = j
                break

        non_random = num_inliers > imin

        # 4.2 Maximality
        Pin = scipy.special.binom(num_inliers, m) / scipy.special.binom(n_star, m)
        maximality = (1 - Pin) ** t <= eta

        if non_random and maximality:
            break

    return model
