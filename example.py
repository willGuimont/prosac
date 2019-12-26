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
    x_range = 0, 10
    y_range = f(x_range[0]), f(x_range[1])
    distances = []
    for i in range(n):
        x = np.random.uniform(*x_range)
        data[i] = [x, f(x) + np.random.randn() * noise]
        distances.append(abs(np.random.randn()))
    for i in range(outliers):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        data[n + i] = [x, y]
        distances.append(abs(np.random.randn() * 2 + 5))
    return data, distances, x_range, y_range


if __name__ == '__main__':
    m = -4
    b = 10


    def f(x):
        return m * x + b


    inliers = 100
    outliers = 100
    data, distances, x_range, y_range = generate_data(f, inliers, outliers)

    tolerance = 10

    model_prosac = PROSAC(data, distances, LinearModel, tolerance, beta=0.10)
    model_ransac = RANSAC(data, LinearModel, tolerance, inliers * 0.75 / (inliers + outliers), 0.8)

    plt.plot(range(10), [model_prosac.predict(x) for x in range(10)], c='magenta', label='PROSAC')
    plt.plot(range(10), [model_ransac.predict(x) for x in range(10)], c='r', label='RANSAC')
    plt.scatter(data[:, 0], data[:, 1], label='Data')
    plt.legend()
    plt.show()

    print(f'Real: y = {m}x + {b}')
    print(f'PROSAC: y = {model_prosac.m}x + {model_prosac.b}')
    print(f'RANSAC: y = {model_ransac.m}x + {model_ransac.b}')
