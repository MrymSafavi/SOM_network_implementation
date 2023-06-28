import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets
from torchvision.models import resnet34

from clustering import Clustering
from som import SOM

np.random.seed(0)

DECAY_LR = 0.1
NEPOCHS = 20
DATA_SIZE = None
FEATURE_SIZE = None
NEURONS = 10
DATA = None


def extract_feature(cifar10):
    resnet = resnet34(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    modules = list(resnet.children())[:-1]
    resnet = torch.nn.Sequential(*modules)
    return resnet(cifar10).numpy()


def convert_neuron_layer(weight):
    temp_weight = []
    for i in range(len(weight)):
        for j in range(len(weight[0])):
            if (i == 3 and j == 0) or (i == 3 and j == 2):
                continue
            temp_weight.append(weight[i][j].fv)

    new_weight = np.array(temp_weight).reshape(NEURONS, FEATURE_SIZE)
    return new_weight


def show_neurons(tsne_weights, typ):
    plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], s=100, c='red', label='Neuron')
    # plt.plot(tsne_weights[:, 0], tsne_weights[:, 1], lw=1, c='r', marker='o', ms=4, label='Neurons')
    plt.title(f'Neurons Position type {typ + 1}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def show_data():
    tsne_d = TSNE(n_components=2, perplexity=30)
    tsne_datas = tsne_d.fit_transform(DATA)
    plt.scatter(tsne_datas[:, 0], tsne_datas[:, 1], s=100, c='blue', label='Data')
    plt.title(f'Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def show_all_data(tsne_weights, tsne_datas, typ):
    plt.scatter(tsne_datas[:, 0], tsne_datas[:, 1], label='Data')
    plt.scatter(tsne_weights[:, 0], tsne_weights[:, 1], s=100, c='red', label='Neurons')
    plt.title(f'Data and Neurons Position type {typ + 1}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.show()


def main():
    global DATA_SIZE
    global FEATURE_SIZE
    global DATA

    tsne = TSNE(n_components=2, perplexity=7)
    # features, y_train, x_test, y_test = load_or_extract_features()
    # DATA = features
    # DATA_SIZE = features.shape[0]
    # FEATURE_SIZE = features.shape[1]

    print("Data extracting...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10('data', train=True,
                                  download=True, transform=transform)

    x_train = []
    y_train = []
    for img, label in train_data:
        x_train.append(img)
        y_train.append(label)

    x_train = torch.stack(x_train)
    y_train = np.array(y_train)
    print("Features of data extracting...")
    features = extract_feature(x_train)
    DATA = features.reshape(features.shape[0], features.shape[1])
    DATA_SIZE = features.shape[0]
    FEATURE_SIZE = features.shape[1]

    print("SOM started ...")
    for typ in range(1):
        typ = 2
        print(f"type: {typ + 1}")
        som = SOM(NEURONS, DECAY_LR, DATA, typ + 1)
        som.train(DATA, NEPOCHS, typ + 1, 0.5)
        if typ == 0 or typ == 1:
            clustering = Clustering(DATA, som.weights)
            clustering.print_cluster(y_train)
            tsne_weights = tsne.fit_transform(som.weights)
            tsne_datas = tsne.fit_transform(DATA)
            show_neurons(tsne_weights, typ)
            print(f"Showing data and Neuron positions...")
            show_all_data(tsne_weights, tsne_datas, typ)
        else:
            weights_type3 = convert_neuron_layer(som.weights)
            clustering = Clustering(DATA, weights_type3)
            clustering.print_cluster(y_train)
            tsne_weights = tsne.fit_transform(weights_type3)
            tsne_datas = tsne.fit_transform(DATA)
            show_neurons(tsne_weights, typ)
            print(f"Showing data and Neuron positions...")
            show_all_data(tsne_weights, tsne_datas, typ)
        print(f"<<<----------------- End Of type {typ + 1} ----------------->>>")


if __name__ == '__main__':
    main()
