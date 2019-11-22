import numpy as np
from sklearn.cluster import KMeans
from scipy.linalg import svd
from sklearn.preprocessing import OneHotEncoder


class HEMAP:
    def __init__(self, src_data, tar_data, args):
        # src_data and tar_data types: numpy
        self.src_feature = src_data[:, :-1]
        self.tar_feature = tar_data[:, :-1]
        self.src_label = src_data[:, -1]
        self.tar_label = tar_data[:, -1]
        self.beta = args.beta
        self.theta = args.theta
        self.topk = args.topk

    @staticmethod
    def one_hot(x):
        x = x.reshape([-1, 1])
        encoder = OneHotEncoder(categories='auto')
        encoder.fit(x)
        onehot_labels = encoder.transform(x).toarray()
        return onehot_labels

    def generate_partition_matrix(self):
        src_neigh = KMeans(n_clusters=2)
        src_neigh.fit(self.src_feature)
        src_partition = src_neigh.predict(self.src_feature)

        tar_neigh = KMeans(n_clusters=2)
        tar_neigh.fit(self.tar_feature)
        tar_partition = tar_neigh.predict(self.tar_feature)

        src_partition_onehot = self.one_hot(src_partition)
        tar_partition_onehot = self.one_hot(tar_partition)

        return src_partition_onehot, tar_partition_onehot

    @property
    def construct_a_matrix(self):
        c_s, c_t = self.generate_partition_matrix()
        a_1 = 2 * self.theta ** 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature)) + \
              self.beta ** 2 / 2 * np.matmul(self.src_feature, np.transpose(self.src_feature)) + \
              (1 - self.theta) * (self.beta + 2 * self.theta) * np.matmul(c_t, np.transpose(c_t))
        a_2 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
                                        + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        a_3 = self.beta * self.theta * (np.matmul(self.tar_feature, np.transpose(self.tar_feature))
                                        + np.matmul(self.src_feature, np.transpose(self.src_feature)))
        a_4 = 2 * self.theta ** 2 * np.matmul(self.src_feature, np.transpose(self.src_feature)) + \
              self.beta ** 2 / 2 * np.matmul(self.tar_feature, np.transpose(self.tar_feature)) + \
              (1 - self.theta) * (self.beta + 2 * self.theta) * np.matmul(c_s, np.transpose(c_s))
        a_upper = np.concatenate((a_1, a_2), axis=1)
        a_lower = np.concatenate((a_3, a_4), axis=1)
        a = np.concatenate((a_upper, a_lower), axis=0)
        return a

    def calculate_eigenvalue(self):
        a = self.construct_a_matrix
        U, _, _ = svd(a)
        return U[:, :self.topk]

    def make_projected_data(self):
        U = self.calculate_eigenvalue()
        bt = U[:len(U)//2]
        bs = U[len(U)//2:]
        return bt, bs
