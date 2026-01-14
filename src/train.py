#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import random
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

sys.path.insert(0, '../')
from data import bpe_dataset
# from src.data import bpe_dataset

from data.mol_bpe import Tokenizer
from pl_models import PSVAEModel
from utils.logger import print_log
from utils.nn_utils import VAEEarlyStopping
from utils.nn_utils import common_config, predictor_config, encoder_config
from utils.nn_utils import ps_vae_config
import torch

print(torch.cuda.is_available())


#  定义一个可以设置随机种子的函数
def setup_seed(seed):
    """
       seed在深度学习代码中叫随机种子，设置seed的目的是由于深度学习网络模型中初始的权值参数通常都是初始化成随机数，
       而使用梯度下降法最终得到的局部最优解对于初始位置点的选择很敏感，设置了seed就相当于规定了初始的随机值。
       即产生随机种子意味着每次运行实验，产生的随机数都是相同的。为将模型在初始化过程中所用到的“随机数”全部固定下来，
       以保证每次重新训练模型需要初始化模型参数的时候能够得到相同的初始化参数，从而达到稳定复现训练结果的目的。
    """
    torch.manual_seed(seed)  # 为CPU设置种子，保证每次的随机初始化都是相同的，从而保证结果可以复现
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子（适用于多GPU时），作用同上
    np.random.seed(seed)  # 保证生成的每个随机数组都是一样的
    # 改变随机生成器的种子，传入的数值用于指定随机数生成时所用算法开始时所选定的整数值，如果使用相同的seed()值，则每次生成的随机数都相同；如果不设置值，每次生成的随机数会不同
    random.seed(seed)
    # 将这个flag设置为TRUE的话，每次返回的卷积算法是确定的，即默认算法，可以提升训练速度，会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
    # 如果结合前置torch中设置随机数seed一定的话，可以保证每次运行网络相同输入得到的输出是相同的。
    torch.backends.cudnn.deterministic = True
    pl.utilities.seed.seed_everything(seed=seed)  # 设置所有的随机种子，包括PyTorch、NumPy和Python的随机种子，可以帮助我们在训练模型时保持结果的一致性。


#  设置随机数种子
setup_seed(2021)


def train(model, train_loader, valid_loader, test_loader, args):
    checkpoint_callback = ModelCheckpoint(
        monitor=args.monitor,  # monitor:需要监视的指标
    )
    # 提前停止（early stopping）是将一部分训练集作为验证集（validation set）。
    # 当验证集的性能越来越差时或者性能不再提升，则立即停止对该模型的训练。这被称为提前停止。
    print_log('Using vae kl warmup early stopping strategy')
    anneal_step = args.kl_warmup + (args.max_beta // args.step_beta - 1) * args.kl_anneal_iter
    early_stop_callback = VAEEarlyStopping(
        finish_anneal_step=anneal_step,
        monitor=args.monitor,
        patience=args.patience
    )
    trainer_config = {
        'gpus': args.gpus,
        'max_epochs': args.epochs,
        'default_root_dir': args.save_dir,
        'callbacks': [checkpoint_callback, early_stop_callback],
        'gradient_clip_val': args.grad_clip
    }
    if len(args.gpus.split(',')) > 1:
        trainer_config['accelerator'] = 'dp'
    trainer = pl.Trainer(**trainer_config)
    # 训练模型
    trainer.fit(model, train_loader, valid_loader)
    # test
    trainer.test(model, dataloaders=test_loader)
    # 改动处：test_dataloaders=test_loader


def parse():
    """parse command"""
    parser = argparse.ArgumentParser(description='training overall model for molecule generation')
    parser.add_argument('--train_set', type=str, required=True, help='path of training dataset')
    parser.add_argument('--valid_set', type=str, required=True, help='path of validation dataset')
    parser.add_argument('--test_set', type=str, required=True, help='path of test dataset')
    parser.add_argument('--vocab', type=str, required=True, help='path of vocabulary (.pkl) or bpe vocab(.txt)')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--save_dir', type=str, required=True, help='path to store the model')
    parser.add_argument('--batch_size', type=int, default=64, help='size of mini-batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, required=True,
                        help='balancing reconstruct loss and predictor loss')
    # vae training
    parser.add_argument('--beta', type=float, default=0.001,
                        help='balancing kl loss and other loss')
    parser.add_argument('--step_beta', type=float, default=0.0005,
                        help='value of beta increasing step')
    parser.add_argument('--max_beta', type=float, default=0.005)
    parser.add_argument('--kl_warmup', type=int, default=2000,
                        help='Within these steps beta is set to 0')
    parser.add_argument('--kl_anneal_iter', type=int, default=1000)

    parser.add_argument('--num_workers', type=int, default=4, help='number of cpus to load data')
    parser.add_argument('--gpus', default=None, help='gpus to use')
    parser.add_argument('--epochs', type=int, default=20, help='max epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping number of epochs')
    parser.add_argument('--grad_clip', type=float, default=0,
                        help='clip large gradient to prevent gradient boom')
    parser.add_argument('--monitor', type=str, default='val_loss',
                        help='Value to monitor in early stopping')

    # model parameters
    parser.add_argument('--props', type=str, nargs='+', choices=['qed', 'sa', 'logp', 'gsk3b', 'jnk3'],
                        help='properties to predict')
    parser.add_argument('--predictor_hidden_dim', type=int, default=200,
                        help='hidden dim of predictor (MLP)')
    parser.add_argument('--node_hidden_dim', type=int, default=100,
                        help='dim of node hidden embedding in encoder and decoder')
    parser.add_argument('--graph_embedding_dim', type=int, default=200,
                        help='dim of graph embedding by encoder and also condition for ae decoders')
    parser.add_argument('--latent_dim', type=int, default=56,
                        help='dim of latent z for vae decoders')
    # ps-vae decoder only
    parser.add_argument('--max_pos', type=int, default=50,
                        help='Max number of pieces')
    parser.add_argument('--atom_embedding_dim', type=int, default=50,
                        help='Embedding dim for a single atom')
    parser.add_argument('--piece_embedding_dim', type=int, default=100,
                        help='Embedding dim for piece')
    parser.add_argument('--pos_embedding_dim', type=int, default=50,
                        help='Position embedding of piece')
    parser.add_argument('--piece_hidden_dim', type=int, default=200,
                        help='Hidden dim for rnn used in piece generation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()
    print_log(args)  # 打印参数列表
    print_log('loading data ...')
    tokenizer = Tokenizer(args.vocab)
    vocab = tokenizer.chem_vocab
    train_loader = bpe_dataset.get_dataloader(args.train_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=args.shuffle, num_workers=args.num_workers)
    valid_loader = bpe_dataset.get_dataloader(args.valid_set, tokenizer, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    test_loader = bpe_dataset.get_dataloader(args.test_set, tokenizer, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)

    # 配置和创建模型
    print('creating model ...')
    config = {**common_config(args), **encoder_config(args, vocab), **predictor_config(args)}
    # 编码器的配置也已更新,将分词器加入
    config.update(ps_vae_config(args, tokenizer))
    model = PSVAEModel(config, tokenizer)
    print_log(f'config: {config}')
    print(model)

    # 从头开始训练
    print('start training')
    train(model, train_loader, valid_loader, test_loader, args)
