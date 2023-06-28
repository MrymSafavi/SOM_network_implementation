import copy
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from neuron import Neuron


def show_neurons(neuron, typ):
    tsne_w = TSNE(n_components=2, perplexity=8)
    tsne_weights = tsne_w.fit_transform(neuron)
    plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], s=100, c='red', label='Neuron')
    plt.title(f'Random Neurons Position type {typ}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


class SOM:

    def __init__(self, neuron, decay_lr, data, typ):
        self.neurons = neuron
        self.dlr = decay_lr
        self.weights = self.initial_neuron_layer(data, typ)

    def initial_neuron_layer(self, data, typ):
        initial_weight = self.initial_basic_neuron_layer(data)
        show_neurons(initial_weight, typ)

        if typ == 1 or typ == 2:
            return initial_weight
        return self.neuron_layer_type3(initial_weight)

    def initial_basic_neuron_layer(self, data):
        random_indices = np.random.choice(len(data), size=self.neurons, replace=False)
        W = data[random_indices, :]
        return W

    def neuron_layer_type3(self, weights):
        list_ner = []
        for feature_vec in weights:
            w = Neuron(feature_vec)
            list_ner.append(w)
        W = np.array([[list_ner[0], list_ner[1], list_ner[2]], [list_ner[3], list_ner[4], list_ner[5]],
                      [list_ner[6], list_ner[7], list_ner[8]], [None, list_ner[9], None]])
        return W

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=1)
        J = np.argmin(distances)
        return J

    def find_bmu_type3(self, x):
        min_distance = sys.maxsize
        min_i = None
        min_j = None
        for i in range(len(self.weights)):
            for j in range(len(self.weights[0])):
                if (i == 3 and j == 0) or (i == 3 and j == 2):
                    continue
                distance = np.linalg.norm(self.weights[i][j].fv - x)
                if distance < min_distance:
                    min_distance = distance
                    min_i = i
                    min_j = j
        return min_i, min_j

    def find_theta(self, bmu, lr, diagonal):
        dist = np.linalg.norm(np.array(bmu) - np.array(bmu + diagonal))
        theta = np.exp(-dist ** 2 / (2 * lr ** 2))
        return theta

    def update_weights_type1(self, x, bmu, lr):

        self.weights[bmu] += lr * (x - self.weights[bmu])

        theta_1bfr = self.find_theta(bmu, lr, -1)
        theta_1aft = self.find_theta(bmu, lr, 1)

        if bmu == 0:
            self.weights[bmu + 1] += theta_1aft * lr * (x - self.weights[bmu + 1])
        elif bmu == self.neurons - 1:
            self.weights[bmu - 1] += theta_1bfr * lr * (x - self.weights[bmu - 1])
        else:
            self.weights[bmu + 1] += theta_1aft * lr * (x - self.weights[bmu + 1])
            self.weights[bmu - 1] += theta_1bfr * lr * (x - self.weights[bmu - 1])
        return self.weights

    def update_weights_type2(self, x, bmu, lr):
        self.weights[bmu] += lr * (x - self.weights[bmu])

        if bmu < 3:
            for i in range(3):
                theta = self.find_theta(bmu, lr, i + 1)
                self.weights[bmu + i + 1] += theta * lr * (x - self.weights[bmu + i + 1])

        elif bmu >= self.neurons - 3:
            for i in range(3):
                theta = self.find_theta(bmu, lr, -1 * (i + 1))
                self.weights[bmu - (i + 1)] += theta * lr * (x - self.weights[bmu - (i + 1)])

        else:
            for i in range(3):
                theta_aft = self.find_theta(bmu, lr, (i + 1))
                self.weights[bmu + (i + 1)] += theta_aft * lr * (x - self.weights[bmu + (i + 1)])

                theta_bfr = self.find_theta(bmu, lr, -1 * (i + 1))
                self.weights[bmu - (i + 1)] += theta_bfr * lr * (x - self.weights[bmu - (i + 1)])
        return self.weights

    def update_weights_type3(self, x, bmu_i, bmu_j, lr):

        if bmu_i == 0 and bmu_j == 0:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)


        elif bmu_i == 0 and bmu_j == 1:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)
            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

        elif bmu_i == 0 and bmu_j == 2:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)
            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

        elif bmu_i == 1 and bmu_j == 0:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

        elif bmu_i == 1 and bmu_j == 1:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

        elif bmu_i == 1 and bmu_j == 2:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

        elif bmu_i == 2 and bmu_j == 0:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

        elif bmu_i == 2 and bmu_j == 1:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)

            theta_r = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j + 1]))
            self.weights[bmu_i][bmu_j + 1].fv += theta_r * lr * (x - self.weights[bmu_i][bmu_j + 1].fv)

            theta_b = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i + 1, bmu_j]))
            self.weights[bmu_i + 1][bmu_j].fv += theta_b * lr * (x - self.weights[bmu_i + 1][bmu_j].fv)

            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

        elif bmu_i == 2 and bmu_j == 2:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)
            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

            theta_l = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i, bmu_j - 1]))
            self.weights[bmu_i][bmu_j - 1].fv += theta_l * lr * (x - self.weights[bmu_i][bmu_j - 1].fv)

        elif bmu_i == 3 and bmu_j == 1:
            self.weights[bmu_i][bmu_j].fv += lr * (x - self.weights[bmu_i][bmu_j].fv)
            theta_t = self.find_theta(np.array([bmu_i, bmu_j]), lr, np.array([bmu_i - 1, bmu_j]))
            self.weights[bmu_i - 1][bmu_j].fv += theta_t * lr * (x - self.weights[bmu_i - 1][bmu_j].fv)

    def train(self, data, n_epochs, typ, learning_rate):
        lr_0 = copy.deepcopy(learning_rate)
        for epoch in range(n_epochs):
            for i, x in enumerate(data):
                if typ == 1:
                    bmu = self.find_bmu(x)
                    self.update_weights_type1(x, bmu, learning_rate)
                if typ == 2:
                    bmu = self.find_bmu(x)
                    self.update_weights_type2(x, bmu, learning_rate)
                if typ == 3:
                    bmu_i, bmu_j = self.find_bmu_type3(x)
                    self.update_weights_type3(x, bmu_i, bmu_j, learning_rate)

            learning_rate = lr_0 * np.exp(-epoch * self.dlr)

