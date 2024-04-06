import os.path

from PIL import Image
import csv
import pandas as pd

f = open("list_eval_partition.txt")
l = pd.read_csv("beard_labels.csv")

train_csv = open("../causal_data/causal_data/beard/train.csv", 'w', encoding='utf-8', newline="")
test_csv = open("../causal_data/causal_data/beard/test.csv", 'w', encoding='utf-8', newline="")

train_csv_writer = csv.writer(train_csv)
test_csv_writer = csv.writer(test_csv)

line = f.readline()
i = 0

origin_path = "E:\\dataset\\celebA_resize\\"
train_path = "E:\\pycharmproject\\GNNProject2\\causal_data\\causal_data\\beard\\train\\"
test_path = "E:\\pycharmproject\\GNNProject2\\causal_data\\causal_data\\beard\\test\\"

while line:
    array = line.split()
    if not os.path.exists(origin_path + array[0]):
        i = i + 1
        line = f.readline()
        continue

    if array[1] == '0':
        train_csv_writer.writerow([l['img'][i], l['label1'][i], l['label2'][i], l['label3'][i], l['label4'][i]])
        I = Image.open(origin_path + array[0]).convert('RGBA')
        I.save(train_path + str(l['img'][i]) + '_' + str(l['label1'][i]) + '_' + str(l['label2'][i]) + '_'
               + str(l['label3'][i]) + '_' + str(l['label4'][i]) + '.png')
    elif array[1] == '2':
        test_csv_writer.writerow([l['img'][i], l['label1'][i], l['label2'][i], l['label3'][i], l['label4'][i]])
        I = Image.open(origin_path + array[0]).convert('RGBA')
        I.save(test_path + str(l['img'][i][0]) + '_' + str(l['label1'][i]) + '_' + str(l['label2'][i]) + '_'
               + str(l['label3'][i]) + '_' + str(l['label4'][i]) + '.png')

    i = i+1
    line = f.readline()


f.close()
train_csv.close()
test_csv.close()
