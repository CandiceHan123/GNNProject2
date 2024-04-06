import csv
f = open("list_attr_celeba.txt")
c = open("smile_labels.csv", 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(c)

line = f.readline()
line = f.readline()
line = f.readline()

while line:
    array = line.split()
    # gender(male)->21   smile->32  eyes open(Narrow_Eyes：细长的眼睛)->24  mouth open(Mouth_Slightly_Open：微微张开嘴巴)->22
    csv_writer.writerow([array[0], array[21], array[32], array[24], array[22]])
    line = f.readline()

f.close()
c.close()

