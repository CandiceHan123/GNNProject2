import csv
f = open("list_attr_celeba.txt")
c = open("beard_labels.csv", 'w', encoding='utf-8', newline="")
csv_writer = csv.writer(c)

line = f.readline()
line = f.readline()
line = f.readline()

while line:
    array = line.split()
    # gender(male)->21   age->40  bald->5  beard(Mustache：胡子，髭)->23
    csv_writer.writerow([array[0], array[21], array[40], array[5], array[23]])
    line = f.readline()

f.close()
c.close()

