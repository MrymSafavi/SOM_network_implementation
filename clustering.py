import numpy as np


def variance(num_of_classes):
    return np.var(num_of_classes)


class Clustering:
    def __init__(self, data, neurons):
        self.data = data
        self.centroids = neurons

    def apply_clustering(self):
        cluster = np.zeros(self.data.shape[0])
        for i, row in enumerate(self.data):
            dist = np.linalg.norm(self.centroids - row, axis=1)
            idx = np.argmin(dist)
            cluster[i] = idx

        return cluster

    def print_cluster(self, labels):
        cluster = self.apply_clustering()

        cluster1 = []
        cluster2 = []
        cluster3 = []
        cluster4 = []
        cluster5 = []
        cluster6 = []
        cluster7 = []
        cluster8 = []
        cluster9 = []
        cluster10 = []

        for idx, c in enumerate(cluster):
            if c == 0:
                cluster1.append(labels[idx])
            elif c == 1:
                cluster2.append(labels[idx])
            elif c == 2:
                cluster3.append(labels[idx])
            elif c == 3:
                cluster4.append(labels[idx])
            elif c == 4:
                cluster5.append(labels[idx])
            elif c == 5:
                cluster6.append(labels[idx])
            elif c == 6:
                cluster7.append(labels[idx])
            elif c == 7:
                cluster8.append(labels[idx])
            elif c == 8:
                cluster9.append(labels[idx])
            elif c == 9:
                cluster10.append(labels[idx])
        print('----------------Number of data from each class in each cluster:----------------')
        num_of_class = []
        print('cluster1:')
        for i in range(10):
            c1 = cluster1.count(i)
            print(f"class {i}: {c1}")
            num_of_class.append(c1)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster2:')
        for i in range(10):
            c2 = cluster2.count(i)
            print(f"class {i}: {c2}")
            num_of_class.append(c2)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster3:')
        for i in range(10):
            c3 = cluster3.count(i)
            print(f"class {i}: {c3}")
            num_of_class.append(c3)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster4:')
        for i in range(10):
            c4 = cluster4.count(i)
            print(f"class {i}: {c4}")
            num_of_class.append(c4)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster5:')
        for i in range(10):
            c5 = cluster5.count(i)
            print(f"class {i}: {c5}")
            num_of_class.append(c5)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster6:')
        for i in range(10):
            c6 = cluster6.count(i)
            print(f"class {i}: {c6}")
            num_of_class.append(c6)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster7:')
        for i in range(10):
            c7 = cluster7.count(i)
            print(f"class {i}: {c7}")
            num_of_class.append(c7)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster8:')
        for i in range(10):
            c8 = cluster8.count(i)
            print(f"class {i}: {c8}")
            num_of_class.append(c8)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster9:')
        for i in range(10):
            c9 = cluster9.count(i)
            print(f"class {i}: {c9}")
            num_of_class.append(c9)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
        print()
        print('cluster10:')
        for i in range(10):
            c10 = cluster10.count(i)
            print(f"class {i}: {c10}")
            num_of_class.append(c10)
        print(f"variance: {variance(num_of_class)}")
        num_of_class.clear()
