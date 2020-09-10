# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import tensorflow as tf
from model import MIR4SR
from utils import Data
import pickle
import argparse
import time
import math


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose')
parser.add_argument('--method', type=str, default='sat', help='session encoder method')
parser.add_argument('--model_path', type=str, default='./exp/yoochoose/model/model')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--emb_size', type=int, default=100, help='hidden state size')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--kg', type=int, default=1000)
parser.add_argument('--num_head', type=int, default=1)
parser.add_argument('--num_block', type=int, default=1)
parser.add_argument('--num_gcn', type=int, default=1)
opt = parser.parse_args()

test_data = pickle.load(open('../datasets/' + opt.dataset + '/test.txt', 'rb'))
test_data = Data(test_data, sub_graph=True, method=opt.method, shuffle=False)

if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
    n_node = 37484
elif opt.dataset == 'yoochoose':
    n_node = 17377
elif opt.dataset == 'y14':
    n_node = 30445
elif opt.dataset == 'lastfm2':
    n_node = 40001
elif opt.dataset == 'lastfm':
    n_node = 39164
elif opt.dataset == 'retailrocket' or 'rr' in opt.dataset:
    n_node = 36969

model = MIR4SR(hidden_size=opt.hidden_size,
               emb_size=opt.emb_size,
               n_node=n_node,
               method=opt.method,
               kg=opt.kg,
               num_head=opt.num_head,
               num_block=opt.num_block,
               nonhybrid=opt.nonhybrid,
               num_gcn=opt.num_gcn)

saver = tf.train.Saver()
saver.restore(model.sess, opt.model_path)
slices = test_data.generate_batch(opt.batch_size)
print('start predict:' + time.strftime('%m-%d %H:%M:%S ', time.localtime(time.time())))
hit = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
mrr = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
ndcg = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
# fetches = [model.logits, model.rec_loss, model.top_k, model.s_emb, model.sa_att, model.att]
fetches = [model.logits, model.rec_loss, model.top_k]
test_loss_, ans, sess_emb, atts, sa_atts = [], [], [], [], []
for i, j in zip(slices, np.arange(len(slices))):
    batch_input = test_data.get_slice(i)
    # scores, test_loss, tk, s_emb, sa_att, att = model.run_rec(
    #     fetches, batch_input, is_train=False)
    scores, test_loss, tk = model.run_rec(
        fetches, batch_input, is_train=False)
    test_loss_.append(test_loss)
    ans.append(tk)
    # sess_emb.append(s_emb)
    # sa_atts.append(sa_att)
    # atts.append(att)

    targets = batch_input[-1]
    for score, target in zip(tk, targets):
        for i in [5, 10, 15, 20, 30, 40, 50, 60]:
            hit[i].append(np.isin(target - 1, score[:i]))
            if len(np.where(score[:i] == target - 1)[0]) == 0:
                mrr[i].append(0)
                ndcg[i].append(0)
            else:
                rank = 1 + np.where(score[:i] == target - 1)[0][0]
                mrr[i].append(1 / rank)
                ndcg[i].append(1 / math.log(rank + 1, 2))

print('test sample %d' % len(hit[20]))
for i in [5, 10, 15, 20, 30, 40, 50, 60]:
    hit[i] = np.mean(hit[i]) * 100
    mrr[i] = np.mean(mrr[i]) * 100
    ndcg[i] = np.mean(ndcg[i]) * 100

test_loss = np.mean(test_loss_)
print('test_loss: %.4f' % (test_loss))
print('Recall@5: %.4f, MMR@5: %.4f, NDCG@5: %.4f' % (hit[5], mrr[5], ndcg[5]))
print('Recall@10: %.4f, MMR@10: %.4f, NDCG@10: %.4f' % (hit[10], mrr[10], ndcg[10]))
print('Recall@15: %.4f, MMR@15: %.4f, NDCG@15: %.4f' % (hit[15], mrr[15], ndcg[15]))
print('Recall@20: %.4f, MMR@20: %.4f, NDCG@20: %.4f' % (hit[20], mrr[20], ndcg[20]))
print('Recall@30: %.4f, MMR@30: %.4f, NDCG@30: %.4f' % (hit[30], mrr[30], ndcg[30]))
print('Recall@40: %.4f, MMR@40: %.4f, NDCG@40: %.4f' % (hit[40], mrr[40], ndcg[40]))
print('Recall@50: %.4f, MMR@50: %.4f, NDCG@50: %.4f' % (hit[50], mrr[50], ndcg[50]))
print('Recall@60: %.4f, MMR@60: %.4f, NDCG@60: %.4f' % (hit[60], mrr[60], ndcg[60]))
