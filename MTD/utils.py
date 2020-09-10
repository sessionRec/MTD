# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import pdb
import random
import math
import pickle
from collections import Counter
import scipy.sparse as sp


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return np.array(us_pois), np.array(us_msks), len_max


def split_validation_v2(all_train_seq, frac=0.1):
    train_len = int(len(all_train_seq) * (1 - frac))
    train_seq = all_train_seq[:train_len]
    valid_seq = all_train_seq[train_len:]

    def generate_seq(seqs):
        set_x, set_y = [], []
        for seq in seqs:
            for i in range(1, len(seq)):
                set_x.append(seq[:-i])
                set_y.append(seq[-i])
        return set_x, set_y

    train_set_x, train_set_y = generate_seq(train_seq)
    valid_set_x, valid_set_y = generate_seq(valid_seq)
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y), train_seq


def split_validation_v1(train_set, frac=0.1):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - frac)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def adj_to_bias(adj, nhood=1):
    nb_graphs = adj.shape[0]
    sizes_g = adj.shape[1]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes_g):
            for j in range(sizes_g):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


class Data():
    def __init__(self, data, all_seqs=None, sub_graph=False, method='sat', sparse=False, shuffle=False):
        self.inputs = np.asarray(data[0])
        self.targets = np.asarray(data[1])
        self.length = len(self.inputs)
        self.shuffle = shuffle
        self.sub_graph = sub_graph
        self.sparse = sparse
        self.method = method

    def construct_remap(self, all_seqs):
        if all_seqs is None:
            return None
        seqs = []
        for i in all_seqs:
            seqs.extend(i)
        cnt = Counter(seqs)
        cc = cnt.most_common()
        nodes = [c[0] for c in cc]
        remap = dict(zip(nodes, range(len(nodes))))
        return remap

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            # self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        remain = self.length % batch_size
        if remain != 0:
            n_batch += 1
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
            slices[-1] = np.arange(self.length - remain, self.length)
        else:
            slices = np.split(np.arange(n_batch * batch_size), n_batch)
        # pdb.set_trace()
        return slices

    def get_slice(self, index):
        if self.method == 'srgnn':
            return self.prepare_data_ggnn(index)
        elif self.method == 'narm':
            return self.prepare_data_narm(self.inputs[index], self.targets[index])
        elif self.method == 'stamp':
            return self.prepare_data_stamp(self.inputs[index], self.targets[index])
        elif self.method == 'sa' or self.method == 'sat' or self.method == 'att':
            return self.prepare_data_sa(self.inputs[index], self.targets[index])
        elif self.method == 'tsa':
            return self.prepare_data_tsa(self.inputs[index], self.targets[index])

    def prepare_data_narm(self, seqs, y, maxlen=19):
        n_samples = len(seqs)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        seq_len = []
        for idx, s in enumerate(seqs):
            t_len = len(s)
            u_len = min(maxlen, t_len)
            seq_len.append(u_len)
            inputs[idx, :u_len] = s[t_len - u_len: t_len]
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, y

    def prepare_data_stamp(self, seqs, y):
        n_samples = len(seqs)
        # seq_len = [len(s) for s in seqs]
        seq_len = [min(len(s), 19) for s in seqs]
        maxlen = max(seq_len)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        for idx, s in enumerate(seqs):
            inputs[idx, :seq_len[idx]] = s[:seq_len[idx]]
            pos[idx, :seq_len[idx]] = range(seq_len[idx], 0, -1)
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, pos, y

    def prepare_data_sa(self, seqs, y):
        n_samples = len(seqs)
        seq_len = [len(s) for s in seqs]
        # seq_len = [min(len(s), 19) for s in seqs]
        maxlen = max(seq_len)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        for idx, s in enumerate(seqs):
            inputs[idx, :seq_len[idx]] = s[-seq_len[idx]:]
            pos[idx, :seq_len[idx]] = range(seq_len[idx], 0, -1)
            # inputs[idx, :u_len] = s[:u_len]
        return inputs, seq_len, pos, y

    def prepare_data_tsa(self, seqs, y, maxlen=19, time_span=256):
        n_samples = len(seqs)
        inputs = np.zeros((n_samples, maxlen)).astype('int64')
        time_matrix = np.zeros((n_samples, maxlen, maxlen)).astype('int64')
        pos = np.zeros((n_samples, maxlen)).astype('int64')
        seq_len = np.zeros((n_samples)).astype('int64')
        for idx, seq in enumerate(seqs):
            c_len = min(maxlen, len(seq))
            seq_len[idx] = c_len
            pos[idx, :c_len] = range(c_len, 0, -1)
            item_seq = [i[0] for i in seq]
            time_seq = [i[1] for i in seq]
            time_scale = 0
            for i in range(len(time_seq) - 1):
                if time_seq[i + 1] - time_seq[i] != 0:
                    time_scale = min(time_seq[i + 1] - time_seq[i], time_scale)
            time_scale = 1 if time_scale == 0 else time_scale
            time_min = min(time_seq)
            time_seq = [int(round((x - time_min) / time_scale) + 1) for x in time_seq]
            inputs[idx, :c_len] = item_seq[-c_len:]
            time_seq = time_seq[-c_len:] + [0] * (max(maxlen - c_len, 0))

            for i in range(maxlen):
                for j in range(maxlen):
                    span = abs(time_seq[i] - time_seq[j])
                    if span > time_span:
                        time_matrix[idx, i, j] = time_span
                    else:
                        time_matrix[idx, i, j] = span
        y = [i[0] for i in y]
        return inputs, pos, time_matrix, seq_len, y

    def prepare_data_ggnn(self, index):
        items, n_node, A_in, A_out, alias_inputs = [], [], [], [], []
        u_inputs, mask, len_max = data_masks(self.inputs[index], [0])
        for u_input in u_inputs:
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)
        for u_input in u_inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)

            A_in.append(u_A_in.transpose())
            A_out.append(u_A_out.transpose())
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        return A_in, A_out, alias_inputs, items, mask, self.targets[index]


class Graph():
    def __init__(self, train_data):
        self.train_data = train_data
        self.G = self.build_graph(train_data)
        self.remap = self.construct_remap()
        self.num_node = len(self.remap)

    def build_graph(self, train_data):
        graph = nx.Graph()
        for seq in train_data:
            for i in range(len(seq) - 1):
                if seq[i] == seq[i + 1]:
                    continue
                if graph.has_edge(seq[i], seq[i + 1]):
                    graph[seq[i]][seq[i + 1]]['weight'] += 1
                else:
                    graph.add_edge(seq[i], seq[i + 1], weight=1)
        # pdb.set_trace()
        for node in graph.nodes():
            unnormal_weight = [graph[node][nbr]['weight'] for nbr in graph.neighbors(node)]
            total = sum(unnormal_weight)
            for nbr in graph.neighbors(node):
                graph[node][nbr]['weight'] /= total
        return graph

    def construct_remap(self):
        nodes = self.G.degree()
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        nodes = [n[0] for n in nodes]
        remap = dict(zip(nodes, range(len(nodes))))
        return remap

    def generate_gat(self):
        adj = np.zeros((self.num_node, self.num_node), dtype=np.float)
        node_list = []
        for nodeid in self.remap.keys():
            neighbors = [self.remap[n] for n in self.G.neighbors(nodeid)]
            adj[self.remap[nodeid], neighbors] = np.ones((len(neighbors), ), dtype=np.float)
            node_list.append(nodeid)
        adj = sp.csr_matrix(adj)
        adj[adj > 0.0] = 1.0
        biases = self.preprocess_adj_bias(adj + sp.eye(adj.shape[0]))
        return biases, np.array(node_list)[np.newaxis]

    def preprocess_adj_bias(self, adj):
        if not sp.isspmatrix_coo(adj):
            adj = adj.tocoo()
        adj = adj.astype(np.float32)
        indices = np.vstack((adj.col, adj.row)).transpose()
        return indices, adj.data, adj.shape
