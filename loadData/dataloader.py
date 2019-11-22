import pickle
import torch.utils.data as data
import torch
import numpy as np


def load_data(file_path):
    """
    Load data
    :param file_path: file path
    :return: (train dataframe, test dataframe, number of features)
    """
    with open(file_path, 'rb') as f:
        saved = pickle.load(f)
        train_df = saved["train_df"]
        test_df = saved["test_df"]
        num_features = saved["num_features"]

        len = train_df.shape[0]
        train = np.array(train_df)[:int(len*0.8)]
        val = np.array(train_df)[int(len*0.8):]
        test = np.array(test_df)

    return train, val, test, num_features


def k_fold_train_val(train, k=3, i=0):
    """
    seperate train and validation data
    :param train_df: train dataframe
    :param k: k-fold cross validation
    :param i: ith validation set
    :return: (train data, validation data)
    """
    # i-th validation for k-fold crossvalidation
    pos = train[:, :][train[:, -1] == 1.0]
    neg = train[:, :][train[:, -1] == 0.0]

    pos_v_size = pos.shape[0] // k  # v : validation
    neg_v_size = neg.shape[0] // k
    v_pos = pos[pos_v_size * i: pos_v_size * (i + 1)]
    v_neg = neg[neg_v_size * i: neg_v_size * (i + 1)]
    v = np.concatenate([v_neg, v_pos])
    print(v)
    train = train[~train.index.isin(v.index)]

    # print(pos_df.shape)
    # print(neg_df.shape)
    # print(v_pos_df.shape)
    # print(v_neg_df.shape)
    # print(v_df.shape)
    # print(train_df.shape)
    return train, val


class Dataset(data.Dataset):
    def __init__(self, df, num_feature):
        """
        Make a dataset
        :param df: dataframe
        :param num_feature: number of features
        :param if_onehot: if True, the label is one-hot vector, otherwise the label consists of 0/1
        """
        _df = df
        self.features = np.array(_df.iloc[:, :num_feature])
        self.label = np.array(_df.iloc[:, -1])
        self.n_data = len(self.label)

    def __getitem__(self, idx):
        """
        Call the item
        :param idx: index
        :return: (data, label)
        """
        features = torch.tensor(self.features[idx].astype(np.float64))
        label = torch.tensor(self.label[idx].astype(np.float64))

        return features, label

    def __len__(self):
        """
        :return: the number of data
        """
        return self.n_data