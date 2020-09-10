import numpy as np
import pickle
import pdb
from sklearn.manifold import TSNE
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import random
matplotlib.use('TkAgg')


def plot_embedding(data, label, idx):
    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x, type1_y = [], []
    type2_x, type2_y = [], []
    type3_x, type3_y = [], []
    type4_x, type4_y = [], []
    type5_x, type5_y = [], []
    type6_x, type6_y = [], []
    type7_x, type7_y = [], []
    type8_x, type8_y = [], []
    type9_x, type9_y = [], []

    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if label[i] == 5:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])
        if label[i] == 6:
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])
        if label[i] == 8:
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])
        if label[i] == 9:
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])

    type1 = plt.scatter(type1_x, type1_y, s=20, c='r')
    type2 = plt.scatter(type2_x, type2_y, s=20, c='g')
    type3 = plt.scatter(type3_x, type3_y, s=20, c='b')
    type4 = plt.scatter(type4_x, type4_y, s=20, c='k')
    type5 = plt.scatter(type5_x, type5_y, s=20, c='c')
    type6 = plt.scatter(type6_x, type6_y, s=20, c='m')
    type7 = plt.scatter(type7_x, type7_y, s=20, c='y')
    # type8 = plt.scatter(type8_x, type8_y, s=10, c='m')
    # type9 = plt.scatter(type9_x, type9_y, s=10, c='y')
    # plt.legend((type1, type2, type3, type4, type5, type6, type7),
    #            ('N', 'B0', 'B1', 'B2', 'I0', 'I1', 'I2', 'O0'),
    #            loc=(0.97, 0.5))

    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')
    # ax.spines['right'].set_visible(False)  # 去除右边框
    # ax.spines['top'].set_visible(False)    # 去除上边框
    return fig


def plot_2D(data, label, epoch=1):
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, perplexity=40.0, init='pca', random_state=0, n_iter=epoch)  # 使用TSNE对特征降到二维
    result = tsne.fit_transform(data)  # 降维后的数据
    fig = plot_embedding(result, label, epoch)
    fig.subplots_adjust(right=0.8)  # 图例过大，保存figure时无法保存完全，故对此参数进行调整


test_data = pickle.load(open('../datasets/diginetica/test.txt', 'rb'))
cnt = Counter(test_data[1])

srgnn = pickle.load(open('./mexp/srgnn.emb', 'rb'))
srgnn = np.concatenate(srgnn)

lgsr = pickle.load(open('./mexp/lgsr.emb', 'rb'))
lgsr = np.concatenate(lgsr)

stamp = pickle.load(open('./mexp/stamp.emb', 'rb'))
stamp = np.concatenate(stamp)

csrm = pickle.load(open('./mexp/csrm.emb', 'rb'))
csrm = np.concatenate(csrm)

tar_s = pickle.load(open('./mexp/diginetica/srgnn.hit', 'rb'))
tar_l = pickle.load(open('./mexp/diginetica/lgsr.hit', 'rb'))
t_s = np.concatenate(tar_s)
t_l = np.concatenate(tar_l)

item_d = dict()
cc = list(cnt.keys())
for i, j in enumerate(test_data[1]):
    if j in cc:
        if j in item_d:
            item_d[j].append(i)
        else:
            item_d[j] = [i]

ans_s = dict()
ans_l = dict()
for k in item_d:
    ans_s[k] = sum([k - 1 in t_s[i, :20] for i in item_d[k]])
    ans_l[k] = sum([k - 1 in t_l[i, :20] for i in item_d[k]])

# cs = []
# for k in ans_s:
#     if ans_l[k] - ans_s[k] >= 6:
#         cs.append(k)

css = []
for i in cnt:
    # if cnt[i] >= 30 and cnt[i] <= 50:
    if cnt[i] >= 30:
        css.append(i)
print(len(css))
for i in range(10):
    cs = random.sample(css, 6)
    print(cs)
    data_s, data_l, data_t, data_c, label = [], [], [], [],[]
    for i in cs:
        # if cs.index(i) in [1, 4]:
        #     continue
        data_s.append(srgnn[item_d[i], :])
        data_l.append(lgsr[item_d[i], :])
        data_t.append(stamp[item_d[i], :])
        data_c.append(csrm[item_d[i], :])
        label.extend([cs.index(i)] * len(item_d[i]))
    data_s = np.concatenate(data_s)
    data_l = np.concatenate(data_l)
    data_t = np.concatenate(data_t)
    data_t = np.concatenate(data_c)
    label = np.array(label)

    plot_2D(data_s, label, 270)
    plot_2D(data_t, label, 270)
    plot_2D(data_c, label, 270)
    plot_2D(data_l, label, 1000)
    plt.show()
