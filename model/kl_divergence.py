import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class KLDivergence:
    def __init__(self, src_projected, tar_projected, args):
        self.args = args
        length = len(src_projected)
        # self.src and self.tar: [feature_1, ..., feature_n, label, s/r] if source: s/r=0, else: s/r=1
        self.src = np.concatenate((src_projected, np.zeros([length, 1])), axis=1)
        self.tar = np.concatenate((tar_projected, np.ones([length, 1])), axis=1)
        self.mixed = self.mix_data()
        self.centroids, self.c_labels = self.kmeansclustering()
        # self.visualize()
        self.total = self.adaptive_clustering()
        self.reduce_kl()

    def mix_data(self):
        mixed = np.concatenate((self.src, self.tar), axis=0)
        np.random.shuffle(mixed)
        return mixed

    def adaptive_clustering(self):
        for i in range(self.args.kclusters):
            while True:
                kmeans = KMeans(n_clusters=self.args.kclusters, random_state=0).fit(self.mixed[:, :-2])
                centroids = kmeans.cluster_centers_
                c_labels = kmeans.labels_

                # feature, label, source/target, cluster_label
                total = np.concatenate((self.mixed, c_labels.reshape([-1, 1])), axis=1)

                c_i = total[total[:, -1] == i]
                c_src = c_i[c_i[:, -2] == 0]
                c_tar = c_i[c_i[:, -2] == 1]

                if c_src.shape[0] > 2 or c_tar.shape[0] > 2:
                    break

            src_in_c = all(elem in c_i for elem in c_src)
            tar_in_c = all(elem in c_i for elem in c_tar)

            src_feature_mean = np.mean(c_src[:, :-3], axis=0)
            tar_feature_mean = np.mean(c_tar[:, :-3], axis=0)

            if src_in_c and tar_in_c and np.abs(src_feature_mean - tar_feature_mean).all() > 0:
                while True:
                    kmeans = KMeans(n_clusters=2).fit(c_i[:, :-3])
                    labels = kmeans.labels_

                    count_nonzero = np.count_nonzero(labels == 0)

                    if 1 < count_nonzero and count_nonzero < len(c_i) - 1  :
                        break

                labels[labels == 1] = labels[labels == 1] + self.args.kclusters - 1 + i
                total[total[:, -1] == i] = labels.reshape([-1, 1])

        print(total[:, -1])

        return total

    def reduce_kl(self):
        total = self.total
        print(total[:, -1])

    def visualize(self):
        # feature, label, source/target, cluster_label
        total = np.concatenate((self.mixed, self.c_labels.reshape([-1, 1])), axis=1)

        color = ['gray', 'brown', 'orange', 'olive', 'green', 'blue', 'cyan', 'purple', 'pink', 'red']

        for i in range(self.args.kclusters):
            tmp_total = total[total[:, -1] == i]
            tmp_src = tmp_total[tmp_total[:, -2] == 0]
            tmp_tar = tmp_total[tmp_total[:, -2] == 1]

            if len(tmp_src) < 2 or len(tmp_tar) < 2:
                continue

            src = TSNE(n_components=2).fit(tmp_src[:, :-3])
            tar = TSNE(n_components=2).fit(tmp_tar[:, :-3])
            src_embedding = src.embedding_
            tar_embedding = tar.embedding_

            plt.scatter(src_embedding[:, 0], src_embedding[:, 1], marker='.', c=color[i])
            plt.scatter(tar_embedding[:, 0], tar_embedding[:, 1], marker='+', c=color[i])
        plt.show()
