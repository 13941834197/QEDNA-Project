#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import time
from copy import copy
import argparse
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from rdkit import Chem

sys.path.insert(0, '../')
from src.utils.chem_utils import molecule2smiles, smiles2molecule, get_submol
from src.utils.chem_utils import cnt_atom, MAX_VALENCE, GeneralVocab, bfs_morgan_order_extended_by_admat
from src.utils.logger import print_log

PIECE_CONNECT_NUM = 9  # 最多 5 个连接，添加 1 个填充和 3 个保留


class MolInPiece:
    """初始化:词汇表V被初始化为所有唯一的原子（带有一个节点的子图）"""

    def __init__(self, mol):
        self.mol = mol                                      # mol对象是rdkit的特殊的对象，专门用于保存化学分子的
        self.smi = molecule2smiles(mol)                     # 将分子转换成smiles格式
        self.pieces, self.pieces_smis = {}, {}              # pid是唯一标识，即pieces片段的id（通过所有原子索引初始化）

        # mol.GetAtoms()对原子进行遍历，都保存成atom对象，返回分子中所有原子atom对象组成的列表
        for atom in mol.GetAtoms():
            idx, symbol = atom.GetIdx(), atom.GetSymbol()    # 获取原子索引和原子符号
            self.pieces[idx] = {idx: symbol}                 # pieces存储idx：symbol对
            self.pieces_smis[idx] = symbol                   # pieces_smis存储原子符号
        self.inversed_index = {}                             # 将原子索引分配给PID  反向索引
        for aid in range(mol.GetNumAtoms()):                 # 对pieces进行遍历
            for key in self.pieces:
                piece = self.pieces[key]
                if aid in piece:
                    self.inversed_index[aid] = key
        self.dirty = True
        self.smi2pids = {}                                   # 不公开，记录相邻图形及其 PID

    def get_nei_pieces(self):                                # 得到相邻的片段
        nei_pieces, merge_pids = [], []
        for key in self.pieces:
            piece = self.pieces[key]
            local_nei_pid = []
            for aid in piece:
                atom = self.mol.GetAtomWithIdx(aid)           # 通过索引获取原子
                for nei in atom.GetNeighbors():               # 获取相连的原子
                    nei_idx = nei.GetIdx()
                    if nei_idx in piece or nei_idx > aid:     # 只考虑连接前面的原子
                        continue
                    local_nei_pid.append(self.inversed_index[nei_idx])
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                new_piece = copy(piece)
                new_piece.update(self.pieces[nei_pid])
                nei_pieces.append(new_piece)
                merge_pids.append((key, nei_pid))
        return nei_pieces, merge_pids

    def get_nei_smis(self):  # 得到相邻的smiles
        if self.dirty:
            nei_pieces, merge_pids = self.get_nei_pieces()
            nei_smis, self.smi2pids = [], {}
            for i, piece in enumerate(nei_pieces):
                submol = get_submol(self.mol, piece)
                smi = molecule2smiles(submol)
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis

    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            merge_pids = self.smi2pids[smi]
            for pid1, pid2 in merge_pids:
                if pid1 in self.pieces and pid2 in self.pieces:  # possibly del by former
                    self.pieces[pid1].update(self.pieces[pid2])
                    self.pieces_smis[pid1] = smi
                    for aid in self.pieces[pid2]:
                        self.inversed_index[aid] = pid1
                    del self.pieces[pid2]
                    del self.pieces_smis[pid2]
        self.dirty = True  # revised

    def get_smis_pieces(self):
        # 返回元组列表(smi, idxs)
        res = []
        for pid in self.pieces_smis:
            smi = self.pieces_smis[pid]
            group_dict = self.pieces[pid]
            idxs = list(group_dict.keys())
            res.append((smi, idxs))
        return res


def freq_cnt(mol):
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, mol


def graph_bpe(fname, vocab_len, vocab_path, cpus):
    """
    字节对编码（BPE, Byte Pair Encoder），是一种数据压缩算法，用来在固定大小的词表中实现可变⻓度的子词
    BPE 首先将词分成单个字符，然后依次用另一个字符替换频率最高的一对字符 ，直到循环次数结束。
    """
    # 载入分子
    print_log(f'Loading mols from {fname} ...')
    with open(fname, 'r') as fin:  # 读取文件
        smis = list(map(lambda x: x.strip(), fin.readlines()))  # 对读入的每一行都去除空格和换行符
    # 1.初始化:词汇表V被初始化为所有唯一的原子（带有一个节点的子图）
    mols = [MolInPiece(smiles2molecule(smi)) for smi in smis if '+' not in smi]
    # loop
    selected_smis, details = list(MAX_VALENCE.keys()), {}  # details: <smi: [atom cnt, frequency]
    # 计算单原子频率
    for atom in selected_smis:
        details[atom] = [1, 0]               # 不计算单个原子的频率
    for smi in smis:
        for c in smi:
            if c in details:
                details[c][1] += 1
    # bpe 处理
    add_len = vocab_len - len(selected_smis)
    pool = mp.Pool(cpus)
    for i in tqdm(range(add_len)):
        res_list = pool.map(freq_cnt, mols)  # 每个元素都是（频率，mol）（ 因为mol不会被同步...
        freqs, mols = {}, []
        st = time.time()
        for freq, mol in res_list:
            mols.append(mol)
            for key in freq:
                freqs.setdefault(key, 0)
                freqs[key] += freq[key]
        # 查找要合并的片段
        max_cnt, merge_smi = 0, ''
        for smi in freqs:
            cnt = freqs[smi]
            if cnt > max_cnt:
                max_cnt = cnt
                merge_smi = smi
        # 合并
        for mol in mols:
            mol.merge(merge_smi)
        selected_smis.append(merge_smi)
        details[merge_smi] = [cnt_atom(merge_smi), max_cnt]
    print_log('sorting vocab by atom num')
    selected_smis.sort(key=lambda x: details[x][0], reverse=True)
    pool.close()
    with open(vocab_path, 'w') as fout:
        fout.writelines(list(map(lambda smi: f'{smi}\t{details[smi][0]}\t{details[smi][1]}\n', selected_smis)))
    return selected_smis, details


class Tokenizer:
    """ 分词器 """

    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as fin:             # 读入长字符串并去除空格和换行符
            lines = fin.read().strip().split('\n')
        self.vocab_dict = {}    # 建立一个空词典
        self.idx2piece, self.piece2idx = [], {}
        for line in lines:
            smi, aton_num, freq = line.strip().split('\t')
            self.vocab_dict[smi] = (int(aton_num), int(freq))
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        self.pad, self.end = '<pad>', '<s>'                # 定义填充符和终止符
        for smi in [self.pad, self.end]:
            self.piece2idx[smi] = len(self.idx2piece)
            self.idx2piece.append(smi)
        # 用于细粒度水平（原子级）
        self.atom_pad = '<pad>'
        self.chem_vocab = GeneralVocab(atom_special=[self.atom_pad])

    def tokenize(self, mol, return_idx=False):
        if isinstance(mol, str):              # 判断mol是否是一个已知类型
            mol = smiles2molecule(mol)
        rdkit_mol = mol
        mol = MolInPiece(mol)
        while True:
            nei_smis = mol.get_nei_smis()
            max_freq, merge_smi = -1, ''
            for smi in nei_smis:
                if smi not in self.vocab_dict:
                    continue
                freq = self.vocab_dict[smi][1]
                if freq > max_freq:
                    max_freq, merge_smi = freq, smi
            if max_freq == -1:
                break
            mol.merge(merge_smi)
        res = mol.get_smis_pieces()
        # smi_atom_cnt = { smi: cnt_atom(smi) for smi, _ in res }
        # res.sort(key=lambda x: smi_atom_cnt[x[0]], reverse=True)  # sort by atom num descending

        # sort by extended morgan bfs 排序方式：扩展morgan BFS(广度优先算法)
        # 构造反向索引
        aid2pid = {}
        for pid, piece in enumerate(res):
            _, aids = piece
            for aid in aids:
                aid2pid[aid] = pid
        # 构造相邻矩阵
        ad_mat = [[0 for _ in res] for _ in res]
        for aid in range(rdkit_mol.GetNumAtoms()):
            atom = rdkit_mol.GetAtomWithIdx(aid)
            for nei in atom.GetNeighbors():
                nei_id = nei.GetIdx()
                i, j = aid2pid[aid], aid2pid[nei_id]
                if i != j:
                    ad_mat[i][j] = ad_mat[j][i] = 1
        # order_list, _ = bfs_morgan_order_extended_by_admat(ad_mat)
        # res = [res[i] for i in order_list]
        np.random.shuffle(res)

        res.insert(0, (self.end, []))
        res.append((self.end, []))
        if not return_idx:
            return res
        piece_idxs = [self.piece_to_idx(x[0]) for x in res]
        group_idxs = [x[1] for x in res]
        return piece_idxs, group_idxs

    def idx_to_piece(self, idx):
        return self.idx2piece[idx]

    def piece_to_idx(self, piece):
        return self.piece2idx[piece]

    def pad_idx(self):
        return self.piece2idx[self.pad]

    def end_idx(self):
        return self.piece2idx[self.end]

    def atom_pad_idx(self):
        return self.chem_vocab.atom_to_idx(self.atom_pad)

    def num_piece_type(self):
        return len(self.idx2piece)

    def num_atom_type(self):
        return self.chem_vocab.num_atom_type()

    def __call__(self, mol, return_idx=False):
        return self.tokenize(mol, return_idx)

    def __len__(self):
        return len(self.idx2piece)


def parse():
    # 1.创建解析器： 创建一个 ArgumentParser 对象，ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser = argparse.ArgumentParser(description='Graph bpe')
    # 2.添加参数
    parser.add_argument('--smiles', type=str,
                        default='O=C(Cc1cc(C(F)(F)F)cc(C(F)(F)F)c1)NCC(c1ccccc1Br)N1CCC(N2CCCCC2)CC1',
                        help='The molecule to tokenize (example)')
    parser.add_argument('--data', type=str, required=True, help='Path to molecule corpus')
    parser.add_argument('--vocab_size', type=int, default=500, help='Length of vocab')
    parser.add_argument('--output', type=str, required=True, help='Path to save vocab')
    parser.add_argument('--workers', type=int, default=16, help='Number of cpus to use')
    return parser.parse_args()  # 返回解析参数


if __name__ == '__main__':    # 条件判断语句，当该文件作为脚本被运行，就执行内部代码
    args = parse()            # 获得解析的参数
    graph_bpe(args.data, vocab_len=args.vocab_size, vocab_path=args.output, cpus=args.workers)
    tokenizer = Tokenizer(args.output)
    print(f'Example: {args.smiles}')
    print(tokenizer.tokenize(args.smiles))
