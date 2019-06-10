import os
import numpy as np

def generate_list(path, out_path, burst=8):
    path_train = os.path.join(path, 'short')
    path_gt = os.path.join(path, 'long')
    f_train = open(os.path.join(out_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(out_path, 'test_list.txt'), 'w')
    files = os.listdir(path_train)
    files_gt = os.listdir(path_gt)
    files_set = set()

    for file in files:
        files_set.add(file[:5])

    len_file = len(files_set)
    train_id = np.random.permutation(len_file)[:200]

    for index, id in enumerate(files_set):
        t = []
        flag = False
        for i in range(len(files)):
            if id in files[i]:
                t.append(i)
                flag = True
            else:
                flag = False
            if len(t) > 0 and not flag:
                break
        rand = np.random.permutation(len(t))[:burst]
        f_temp = []
        for r in rand:
            f_temp.append(os.path.join('short', files[t[r]]))
        for file in files_gt:
            if id in file:
                f_temp.append(os.path.join('long', file))

                if index in train_id:
                    f_train.write(' '.join(f_temp) + '\n')
                else:
                    f_test.write(' '.join(f_temp) + '\n')
                break
    f_train.close()
    f_test.close()

    print('Process of generating train/test file list is ok!')

if __name__ == '__main__':
    dataset_path = 'G:/BinZhang/DataSets/rnn_fcn/Sony/Sony'
    out_path = './'
    generate_list(dataset_path, out_path)