import pickle
import numpy as np
import pdb


def cal(dataset, cut=5):
    test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))
    narm = pickle.load(open('./exp/' + dataset + '/narm.hit', 'rb'))
    narm = np.concatenate(narm)
    stamp = pickle.load(open('./exp/' + dataset + '/stamp.hit', 'rb'))
    stamp = np.concatenate(stamp)
    srgnn = pickle.load(open('./exp/' + dataset + '/srgnn.hit', 'rb'))
    srgnn = np.concatenate(srgnn)
    csrm = pickle.load(open('./exp/' + dataset + '/csrm.pkl', 'rb'))
    csrm = np.concatenate([cc[0] for cc in csrm])
    if dataset == 'yoochoose':
        sa = pickle.load(open('./exp/' + dataset + '/sa.hit', 'rb'))
        mtd = pickle.load(open('./exp/' + dataset + '/mtd.mrr', 'rb'))
        tsa = pickle.load(open('./exp/' + dataset + '/tsa.mrr', 'rb'))
    else:
        sa = pickle.load(open('./exp/' + dataset + '/sa.hit', 'rb'))
        mtd = pickle.load(open('./exp/' + dataset + '/mtd.hit', 'rb'))
        tsa = pickle.load(open('./exp/' + dataset + '/tsa.hit', 'rb'))
    mtd = np.concatenate(mtd)
    tsa = np.concatenate(tsa)
    sa = np.concatenate(sa)
    # pdb.set_trace()

    def cal_method(method, result):
        hit_short, hit_long, hit_all = [], [], []
        for idx in range(len(test_data[0])):
            if method == 'csrm':
                sucess = test_data[1][idx] in result[idx, :20]
            else:
                sucess = (test_data[1][idx] - 1) in result[idx, :20]
            if len(test_data[0][idx]) <= cut:
                hit_short.append(sucess)
            else:
                hit_long.append(sucess)
            hit_all.append(sucess)
        print('    <=======%s=======>' % method)
        print('short:', len(hit_short), np.mean(hit_short))
        print('long :', len(hit_long), np.mean(hit_long))
        print('all  :', len(hit_all), np.mean(hit_all))

    cal_method('narm', narm)
    cal_method('stamp', stamp)
    cal_method('srgnn', srgnn)
    cal_method('csrm', csrm)
    cal_method('sa', sa)
    cal_method('tsa', tsa)
    cal_method('mtd', mtd)


def cal_interval(dataset):
    test_data = pickle.load(open('../datasets/' + dataset + '/test.txt', 'rb'))
    narm = pickle.load(open('./exp/' + dataset + '/narm.hit', 'rb'))
    narm = np.concatenate(narm)
    stamp = pickle.load(open('./exp/' + dataset + '/stamp.hit', 'rb'))
    stamp = np.concatenate(stamp)
    srgnn = pickle.load(open('./exp/' + dataset + '/srgnn.hit', 'rb'))
    srgnn = np.concatenate(srgnn)
    sa = pickle.load(open('./exp/' + dataset + '/sa.hit', 'rb'))
    sa = np.concatenate(sa)
    csrm = pickle.load(open('./exp/' + dataset + '/csrm.pkl', 'rb'))
    csrm = np.concatenate([cc[0] for cc in csrm])
    if dataset == 'yoochoose':
        mtd = pickle.load(open('./exp/' + dataset + '/mtd.hit', 'rb'))
        tsa = pickle.load(open('./exp/' + dataset + '/tsa.mrr', 'rb'))
    else:
        mtd = pickle.load(open('./exp/' + dataset + '/mtd.hit', 'rb'))
        tsa = pickle.load(open('./exp/' + dataset + '/tsa.hit', 'rb'))
    mtd = np.concatenate(mtd)
    tsa = np.concatenate(tsa)
    # pdb.set_trace()

    # split
    bucket = dict()
    total_click = 0
    for idx in range(len(test_data[0])):
        s_len = len(test_data[0][idx])
        if s_len in bucket:
            bucket[s_len].append(idx)
        else:
            bucket[s_len] = [idx]
        total_click += s_len + 1
    max_len = max(list(bucket.keys()))
    min_len = min(list(bucket.keys()))
    split_num = total_click / 4
    s_idx, temp = 0, 0
    split_list = [0, 0, 0, 0]
    split_len = [0, 0, 0, 0]
    # pdb.set_trace()
    for i in range(min_len, max_len + 1):
        if s_idx > 3:
            break
        temp += (len(bucket[i]) * (i + 1))
        split_list[s_idx] = i
        if split_num - temp < 1000 and s_idx < 3:
            split_len[s_idx] = temp
            temp = 0
            s_idx += 1
    if split_len[-1] == 0:
        split_len[-1] = temp
    # pdb.set_trace()

    def cal_method(method, result):
        hit_1, hit_2, hit_3, hit_4, hit_all = [], [], [], [], []
        mrr_1, mrr_2, mrr_3, mrr_4, mrr_all = [], [], [], [], []
        for idx in range(len(test_data[0])):
            if method == 'csrm':
                sucess = test_data[1][idx] in result[idx, :20]
            else:
                sucess = (test_data[1][idx] - 1) in result[idx, :20]
            if sucess:
                if method == 'csrm':
                    rank = 20 - np.where(result[idx, :20] == test_data[1][idx])[0][0]
                else:
                    rank = 1 + np.where(result[idx, :20] == test_data[1][idx] - 1)[0][0]
                rank = 1 / rank
            else:
                rank = 0
            if len(test_data[0][idx]) <= split_list[0]:
                hit_1.append(sucess)
                mrr_1.append(rank)
            elif len(test_data[0][idx]) <= split_list[1]:
                hit_2.append(sucess)
                mrr_2.append(rank)
            elif len(test_data[0][idx]) <= split_list[2]:
                hit_3.append(sucess)
                mrr_3.append(rank)
            elif len(test_data[0][idx]) <= split_list[3]:
                hit_4.append(sucess)
                mrr_4.append(rank)
            hit_all.append(sucess)
            mrr_all.append(rank)
        print('    <=======%s=======>' % method)
        print('hit_1:', split_list[0], len(hit_1), np.mean(hit_1), split_len[0])
        print('hit_2:', split_list[1], len(hit_2), np.mean(hit_2), split_len[1])
        print('hit_3:', split_list[2], len(hit_3), np.mean(hit_3), split_len[2])
        print('hit_4:', split_list[3], len(hit_4), np.mean(hit_4), split_len[3])
        print('all  :', len(hit_all), np.mean(hit_all))
        print('mrr_1:', split_list[0], len(mrr_1), np.mean(mrr_1), split_len[0])
        print('mrr_2:', split_list[1], len(mrr_2), np.mean(mrr_2), split_len[1])
        print('mrr_3:', split_list[2], len(mrr_3), np.mean(mrr_3), split_len[2])
        print('mrr_4:', split_list[3], len(mrr_4), np.mean(mrr_4), split_len[3])
        print('all  :', len(mrr_all), np.mean(mrr_all))
        # pdb.set_trace()

    cal_method('narm', narm)
    cal_method('stamp', stamp)
    cal_method('sa', sa)
    cal_method('tsa', tsa)
    cal_method('srgnn', srgnn)
    cal_method('csrm', csrm)
    cal_method('mtd', mtd)


if __name__ == '__main__':
    # print('######-yoochooce-######')
    # cal('yoochoose', 6)
    # print('\n######-diginetica-######')
    # cal('diginetica', 5)
    # print('\n######-retailrocket-######')
    # cal('retailrocket', 10)
    print('######-yoochooce-######')
    cal_interval('yoochoose')
    print('\n######-diginetica-######')
    cal_interval('diginetica')
    print('\n######-retailrocket-######')
    cal_interval('retailrocket')
