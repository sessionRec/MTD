import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()


def drop_time(dataset):
    train_data = pickle.load(open('../datasets/' + dataset + '/train_time.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + dataset + '/test_time.txt', 'rb'))
    all_train_seq = pickle.load(open('../datasets/' + dataset + '/all_train_seq_time.txt', 'rb'))

    tr1 = []
    for i in train_data[0]:
        tr1.append([ii[0] for ii in i])
    tr2 = [i[0] for i in train_data[1]]
    train_data = [tr1, tr2]
    pickle.dump(train_data, open('../datasets/' + dataset + '/train.txt', 'wb'))

    te1 = []
    for i in test_data[0]:
        te1.append([ii[0] for ii in i])
    te2 = [i[0] for i in test_data[1]]
    test_data = [te1, te2]
    pickle.dump(test_data, open('../datasets/' + dataset + '/test.txt', 'wb'))

    seq = []
    for i in all_train_seq:
        seq.append([ii[0] for ii in i])
    pickle.dump(seq, open('../datasets/' + dataset + '/all_train_seq.txt', 'wb'))


if __name__ == '__main__':
    drop_time(opt.dataset)
