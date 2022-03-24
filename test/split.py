import os
import random
file_path = '/root/VCTK-Corpus/wav48'
file_list = []
for root, dirs, files in os.walk(file_path, topdown=False):
    for name in files:
        if os.path.splitext(name)[1] == ".wav" or ".mp3" or ".flac":
            file_list.append(os.path.join(root, name))
random.shuffle(file_list)
test_set = file_list[:1000]
train_set = file_list[1000:]
with open('train.csv','w') as t:
    t.writelines(['%s\n' % item for item in train_set])
with open('test.csv','w') as t:
    t.writelines(['%s\n' % item for item in test_set])