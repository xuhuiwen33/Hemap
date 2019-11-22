import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class HEMAP:
    def __init__(self, src_data, tar_data, beta, theta):
        # src_data and tar_data types: numpy
        self.src_data = src_data
        self.tar_data = tar_data
        self.beta = beta
        self.theta = theta

    def generate_partition_matrix(self):
        return 0, 1

    def construct_a_matrix(self):
        c_s, c_t = self.generate_partition_matrix()
        a_1 = 2 * self.theta ** 2 * self.tar_data * np.transpose(self.tar_data) + \
              self.beta ** 2 / 2 * self.src_data * np.transpose(self.src_data) + \
              (1 - self.theta) * (self.beta + 2 * self.theta) * c_t * np.transpose(c_t)
        a_2 = self.beta * self.theta * (self.tar_data * np.transpose(self.tar_data)
                                        + self.src_data * np.transpose(self.src_data))
        a_3 = self.beta * self.theta * (self.tar_data * np.transpose(self.tar_data)
                                        + self.src_data * np.transpose(self.src_data))
        a_4 = 2 * self.theta ** 2 * self.src_data * np.transpose(self.src_data) + \
              self.beta ** 2 / 2 * self.tar_data * np.transpose(self.tar_data) + \
              (1 - self.theta) * (self.beta + 2 * self.theta) * c_s * np.transpose(c_s)
        a_upper = np.concatenate((a_1, a_2), axis=1)
        a_lower = np.concatenate((a_3, a_4), axis=1)
        a = np.concatenate((a_upper, a_lower), axis=0)
        return a

    def calculate_eigenvalue(self):
        a = self.construct_a_matrix()


    def make_projected_data(self):
        pass