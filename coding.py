import itertools

version1 = input()
version2 = input()

res = 0

for v1, v2 in itertools.zip_longest(version1.split('.'), version2.split('.'), fillvalue=0):
    x, y = int(v1), int(v2)
    if x != y:
        res = 1 if x > y else -1
print(res)
