import argparse
import time
import csv
import pickle
import operator
import datetime
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = './digineica/train-item-views.csv'
elif opt.dataset == 'yoochoose':
    dataset = './yoochoose/yoochoose-clicks.dat'
elif opt.dataset == 'retailrocket':
    dataset = './retailrocket/event1.csv'

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose' or opt.dataset == 'retailrocket':
        reader = csv.DictReader(f, delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')
    sess_clicks = {}
    sess_date = {}
    ctr = 0
    curid = -1
    curdate = None
    for data in reader:
        sessid = data['session_id']
        if curdate and not curid == sessid:
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
            elif opt.dataset == 'retailrocket':
                date = int(curdate)
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            sess_date[curid] = date
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id'], int(time.mktime(time.strptime(data['timestamp'][:19], '%Y-%m-%dT%H:%M:%S')))
        elif opt.dataset == 'retailrocket':
            item = data['item_id'], int(data['timestamp'])
        else:
            item = data['item_id'], int(data['timeframe'])
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        elif opt.dataset == 'retailrocket':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    elif opt.dataset == 'retailrocket':
        date = int(curdate)
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
            # sess_clicks[i] = [c[0] for c in sorted_clicks]
            sess_clicks[i] = sorted_clicks
    sess_date[curid] = date
print("-- Reading data @ %ss" % datetime.datetime.now())
print("length of session clicks: %d" % len(sess_clicks))

# pdb.set_trace()
# Filter out length 1 sessions
filter_len = 1
if opt.dataset == 'retailrocket':
    filter_len = 2
for s in list(sess_clicks):
    if len(sess_clicks[s]) <= filter_len:
        del sess_clicks[s]
        del sess_date[s]
print("after filter out length of %d, length of session clicks: %d" % (filter_len, len(sess_clicks)))

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid, _ in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i[0]] >= 5, curseq))
    if len(filseq) <= filter_len:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq
print("after item<5 , length of session clicks: %d" % len(sess_clicks))

# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
# print(len(tra_sess))    # 186670    # 7966257
# print(len(tes_sess))    # 15979     # 15324
# print(tra_sess[:3])
# print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper
item_dict = {}


# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i, j in seq:
            if i in item_dict:
                outseq += [(item_dict[i], j)]
            else:
                outseq += [(item_ctr, j)]
                item_dict[i] = item_ctr
                item_ctr += 1
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('item_num: %d' % (item_ctr - 1))    # 43098, 37484
    return train_ids, train_dates, train_seqs


# Convert test sessions to sequences, ignoring items that do not appear in training set
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i, j in seq:
            if i in item_dict:
                outseq += [(item_dict[i], j)]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()
print("training: %d" % len(tra_ids))
print("testing: %d" % len(tes_ids))
# pdb.set_trace()


def process_seqs(iseqs, idates):
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for sid, seq, date in zip(range(len(iseqs)), iseqs, idates):
        temp = len(seq)
        if opt.dataset == 'retailrocket':
            temp = len(seq) - 1
        for i in range(1, temp):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [sid]
    return out_seqs, out_dates, labs, ids


tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)
tes = (te_seqs, te_labs)
print("after split,training: %d" % len(tr_seqs))
print("after split,testing: %d" % len(te_seqs))
# print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
# print(te_seqs[:3], te_dates[:3], te_labs[:3])
total_len = 0

for seq in tra_seqs:
    total_len += len(seq)
for seq in tes_seqs:
    total_len += len(seq)
print('avg length: ', total_len / (len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train_time.txt', 'wb'))
    pickle.dump(tes, open('diginetica/test_time.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq_time.txt', 'wb'))
    pickle.dump(tes_seqs, open('diginetica/all_test_seq_time.txt', 'wb'))
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test_time.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test_time.txt', 'wb'))

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train_time.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq_time.txt', 'wb'))
    pickle.dump(tes_seqs, open('yoochoose1_4/all_test_seq_time.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train_time.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq_time.txt', 'wb'))
    pickle.dump(tes_seqs, open('yoochoose1_64/all_test_seq_time.txt', 'wb'))
elif opt.dataset == 'retailrocket':
    if not os.path.exists('retailrocket'):
        os.makedirs('retailrocket')
    pickle.dump(tra, open('retailrocket/train_time.txt', 'wb'))
    pickle.dump(tes, open('retailrocket/test_time.txt', 'wb'))
    pickle.dump(tra_seqs, open('retailrocket/all_train_seq_time.txt', 'wb'))
    pickle.dump(tes_seqs, open('retailrocket/all_test_seq_time.txt', 'wb'))
else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
