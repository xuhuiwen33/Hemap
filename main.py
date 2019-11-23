import sys
sys.path.append('../')
sys.path.append('../loadData')

import argparse
import os
import torch.utils.data
import click
import numpy as np
from loadData.dataloader import load_data, k_fold_train_val, Dataset
from utils.util import mkdir
from model.transformation import HEMAP
from model.kl_divergence import KLDivergence


def process_parser():
    """
    Processing parser
    :return: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', default=999, type=int, help='')
    parser.add_argument('-l', '--log', default=6, type=int, help='')
    parser.add_argument('-s', '--source', default='german', type=str, help='')
    parser.add_argument('-t', '--target', default='australian', type=str, help='')
    parser.add_argument('--lr', default=0.001, type=float, help='')
    parser.add_argument('--epoch', default=10, type=int, help='')
    parser.add_argument('--batch', default=32, type=int, help='')
    parser.add_argument('--frac', default=1.0, type=float, help='')
    parser.add_argument('--optimizer', default='adam', type=str, help='')
    parser.add_argument('--beta', type=float, default=1.0, help='')
    parser.add_argument('--theta', type=float, default=0.5, help='')
    parser.add_argument('--finetuning', default=False, type=str, help='')
    # SRC_TRANS, SRC_TRANS_AE, TRANS_ALL, TRANS_ALL_AE, SRC_TRANS_LD, SRC_TRANS_LD_AE, TRANS_ALL_LD, TRANS_ALL_LD_AE
    parser.add_argument('--model', default='SRC_TRANS_LD_AE', type=str, help='')
    parser.add_argument('--k', default=5, type=int, help='')
    parser.add_argument('--count', default=5, type=int, help='')
    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--topk', default=50, type=int, help='')
    parser.add_argument('--kclusters', default=10, type=int, help='')
    return parser


if __name__ == '__main__':
    parser = process_parser()
    args = parser.parse_args()
    print(args)

    # Save hyperparameters
    save_path = './saved'
    mkdir(save_path)
    mkdir(os.path.join(save_path, 'results'))
    with open(os.path.join(save_path + '/results', 'hyperparameter.txt'), 'a+') as f:
        f.write(str(args)+'\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:\t', DEVICE)

    # Hyperparameters
    SRC_NAME = args.source
    TAR_NAME = args.target
    BATCH_SIZE = args.batch
    FRAC = args.frac
    BETA = args.beta
    THETA = args.theta
    MODEL_NAME = args.model
    K = args.k  # K-fold cross validation

    # Always use all source!
    src_file_path = "./data/" + SRC_NAME + "/" + SRC_NAME + "_" + str(
        int((1.0 + 0.0000001) * 100)) + ".pickle"
    tar_file_path = "./data/" + TAR_NAME + "/" + TAR_NAME + "_" + str(
        int((FRAC + 0.0000001) * 100)) + ".pickle"

    src_train, src_val, src_test, SRC_DIM = load_data(src_file_path)
    tar_train, tar_val, tar_test, TAR_DIM = load_data(tar_file_path)

    val_losses = []
    min_val_loss = 99999999

    # print("SRC_TRAIN: ", src_train.shape)
    # print("SRC_VALID: ", src_val.shape)
    # print("SRC_TEST: ", src_test.shape)
    # print("TAR_TRAIN: ", tar_train.shape)
    # print("TAR_VALID: ", tar_val.shape)
    # print("TAR_TEST: ", tar_test.shape)

    # # Adjust batch size
    # if BATCH_SIZE > tar_train_df.shape[0]:
    #     BATCH_SIZE = tar_train_df.shape[0] // 2
    #
    # src_train_dataset = Dataset(src_train_df, SRC_DIM)
    # src_valid_dataset = Dataset(src_valid_df, SRC_DIM)
    # src_test_dataset = Dataset(src_test_df, SRC_DIM)
    # tar_train_dataset = Dataset(tar_train_df, TAR_DIM)
    # tar_valid_dataset = Dataset(tar_valid_df, TAR_DIM)
    # tar_test_dataset = Dataset(tar_test_df, TAR_DIM)
    #
    # source_train = torch.utils.data.DataLoader(dataset=src_train_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True, drop_last=False,
    #                                            num_workers=1)
    # source_valid = torch.utils.data.DataLoader(dataset=src_valid_dataset,
    #                                            batch_size=len(
    #                                                src_valid_dataset),
    #                                            shuffle=True, drop_last=False,
    #                                            num_workers=1)
    # src_test = torch.utils.data.DataLoader(dataset=src_test_dataset,
    #                                        batch_size=len(src_test_dataset),
    #                                        shuffle=True, drop_last=False,
    #                                        num_workers=1)
    # target_train = torch.utils.data.DataLoader(dataset=tar_train_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True, drop_last=False,
    #                                            num_workers=1)
    # target_valid = torch.utils.data.DataLoader(dataset=tar_valid_dataset,
    #                                            batch_size=len(
    #                                                tar_valid_dataset),
    #                                            shuffle=True, drop_last=False,
    #                                            num_workers=1)
    # tar_test = torch.utils.data.DataLoader(dataset=tar_test_dataset,
    #                                        batch_size=len(tar_test_dataset),
    #                                        shuffle=True, drop_last=False,
    #                                        num_workers=1)


    # Random sampling
    rand_index = np.random.randint(len(src_train), size=len(tar_train))
    src_train = src_train[rand_index]

    hemap = HEMAP(src_train, tar_train, args)
    src_projected, tar_projected = hemap.make_projected_data()

    kl_divergence = KLDivergence(src_projected, tar_projected, args)
