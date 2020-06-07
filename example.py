import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from random_consensus import Model, PROSAC, RANSAC


class LinearModel(Model):

    def __init__(self):
        self.m = 0
        self.b = 0

    def fit(self, pts):
        xs = pts[:, 0]
        ys = pts[:, 1]
        self.m, self.b = stats.linregress(xs, ys)[:2]

    def predict(self, x):
        return self.m * x + self.b

    def error(self, data):
        prediction = self.predict(data[:, 0])
        true_value = data[:, 1]
        return np.sqrt(np.square(true_value - prediction))

    @staticmethod
    def get_complexity():
        return 2


def generate_data(f, n, outliers):
    noise = 2
    data = np.zeros((n + outliers, 2))
    x_range = [0, 10]
    y_range = sorted([f(x_range[0]), f(x_range[1])])
    quality = []
    for i in range(n):
        x = np.random.uniform(*x_range)
        data[i] = [x, f(x) + np.random.randn() * noise]
        quality.append(1 / abs(np.random.randn()))
    for i in range(outliers):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        data[n + i] = [x, y]
        quality.append(1 / abs(np.random.randn() * 2 + 5))
    return data, quality, x_range, y_range


if __name__ == '__main__':
    m = -4
    b = 10


    def f(x):
        return m * x + b


    inliers = 100
    outliers = 100
    data, quality, x_range, y_range = generate_data(f, inliers, outliers)

    x_min, x_max = x_range
    y_min, y_max = y_range

    tolerance = 3

    # Base from data ranges and tolerance, a point has beta probability of being supported by an incorrect model
    data_area = (x_max - x_min) * (y_max - y_min)
    max_line_length = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
    tolerance_area = max_line_length * tolerance

    beta = (tolerance_area / data_area) * 1.1  # add 10% to be more pessimist

    model_prosac = PROSAC(data, quality, LinearModel, tolerance, beta=beta)
    # The value can be hard to estimate, it is better to underestimate the value
    prob_inlier = inliers * 0.5 / (inliers + outliers)
    model_ransac = RANSAC(data, LinearModel, tolerance, prob_inlier=prob_inlier, min_ratio_correct_model=0.8)

    plt.plot(range(10), [model_prosac.predict(x) for x in range(10)], c='magenta', label='PROSAC')
    plt.plot(range(10), [model_ransac.predict(x) for x in range(10)], c='r', label='RANSAC')
    plt.scatter(data[:, 0], data[:, 1], label='Data')
    plt.legend()
    plt.show()

    print(f'Real: y = {m}x + {b}')
    print(f'PROSAC: y = {model_prosac.m}x + {model_prosac.b}')
    print(f'RANSAC: y = {model_ransac.m}x + {model_ransac.b}')
