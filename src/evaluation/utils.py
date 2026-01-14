#!/usr/bin/python
# -*- coding:utf-8 -*-
# import numpy as np
import networkx as nx  # 绘制网络关系图
from rdkit import Chem, DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Contrib.SA_Score import sascorer

sys.path.remove(os.path.join(RDConfig.RDContribDir, 'SA_Score'))


def smiles2molecule(smiles: str):
    """将smiles转换为分子形式"""
    return Chem.MolFromSmiles(smiles)


# 计算分子间的相似度
def similarity(mol1, mol2):
    fps1 = AllChem.GetMorganFingerprint(mol1, 2)  # 返回分子的摩根指纹
    fps2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fps1, fps2)  # 返回两个向量之间的谷本相似度


def num_long_cycles(mol):
    """计算长周期数。
    Args:
      mol: Molecule. A molecule.
    Returns:
      negative cycle length.
    """
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if not cycle_list:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -cycle_length


def get_sa(molecule):
    """返回给定分子的可合成性.
       取值范围从1到10，越低越好（越容易制作）"""
    return sascorer.calculateScore(molecule)


def get_penalized_logp(mol):
    """
    奖励由SA惩罚的log p和长周期组成，如Kusner et al. 2017）中所述。
    根据 250k_rndm_zinc_drugs_clean.smi 数据集的统计数据对分数进行归一化
    :p aram mol： rdkit mol object
    ：return： float
    """
    # 归一化常量，来自 250k_rndm_zinc_drugs_clean.smi 的统计信息
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # 周期分数
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    # return log_p + SA + cycle_score
    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


def get_qed(molecule):
    """给定分子的QED。取值范围从 0 到 1，越高越好"""
    return qed(molecule)


# 评价指标字典
def eval_funcs_dict():
    eval_funcs = {
        'qed': get_qed,
        'sa': get_sa,
        'logp': get_penalized_logp
    }
    return eval_funcs


def get_normalized_property_scores(mol):
    # 使每个属性大致在范围（0， 1） 内，并且所有属性都越高越好
    qed = get_qed(mol)
    sa = get_sa(mol)
    logp = get_penalized_logp(mol)
    return [qed, 1 - sa / 10, (logp + 10) / 13]  # 都是越高越好


def restore_property_scores(normed_props):
    return [normed_props[0], 10 * (1 - normed_props[1]),
            13 * normed_props[2] - 10]


PROP_TH = [0.6, 6.0, 0]  # 改动处：将4.0改成6.0
NORMALIZED_TH = None
PROPS = ['qed', 'sa', 'logp']


def map_prop_to_idx(props):
    global PROPS
    idxs = []
    p2i = {}
    for i, p in enumerate(PROPS):
        p2i[p] = i
    for p in props:
        if p in p2i:
            idxs.append(p2i[p])
        else:
            raise ValueError('Invalid property')
    return sorted(list(set(idxs)))


def overpass_th(prop_vals, prop_idx):
    ori_prop_vals = [0 for _ in PROPS]
    for i, val in zip(prop_idx, prop_vals):
        ori_prop_vals[i] = val
    ori_prop_vals = restore_property_scores(ori_prop_vals)
    for i in prop_idx:
        if ori_prop_vals[i] < PROP_TH[i]:
            return False
    return True


# 采样策略top-k
class TopStack:
    """仅保存 top-k 结果和相应的 """

    def __init__(self, k, cmp):
        # k: capacity, cmp: 二进制比较器，指示 x 是否在 y 之前
        self.k = k
        self.stack = []
        self.cmp = cmp

    def push(self, val, data=None):
        i = len(self.stack) - 1
        while i >= 0:
            if self.cmp(self.stack[i][0], val):
                break
            else:
                i -= 1
        i += 1
        self.stack.insert(i, (val, data))
        if len(self.stack) > self.k:
            self.stack.pop()

    def get_iter(self):
        return iter(self.stack)


if __name__ == '__main__':
    # args = parse()
    # init_stats(args.data, args.cpus)
    eg = 'CN(C)CC[C@@H](c1ccc(Br)cc1)c1ccccn1'
    m = smiles2molecule(eg)
    eval_funcs = eval_funcs_dict()
    for key in eval_funcs:
        f = eval_funcs[key]
        print(f'{key}: {f(m)}')
    print(f'normalized: {get_normalized_property_scores(m)}')
