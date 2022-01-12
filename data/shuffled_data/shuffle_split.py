import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, default='./all/data_binary.csv',help='input data file')
parser.add_argument('--train_file', type=str, default='binary_train.csv',help='output train file')
parser.add_argument('--dev_file', type=str, default='binary_dev.csv',help='output dev file')

FLAGS, unparsed = parser.parse_known_args()

lines = open(FLAGS.data_file).readlines()
print(len(lines))

lines=lines[1:]
random.shuffle(lines)
split = int(round(len(lines)*0.8))
print(f"Analysis:\n Length source file: {len(lines)}\n Split at line: {split}")

with open (FLAGS.train_file, 'w') as trainingfile:
    trainingfile.write('text,label,\n')
    for line in lines[1:split]:
        trainingfile.write(line)
    
with open (FLAGS.dev_file, 'w') as devfile:
    devfile.write('text,label,\n')
    for line in lines[split:]:
        devfile.write(line)

