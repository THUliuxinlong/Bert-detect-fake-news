# AEDA: An Easier Data Augmentation Technique for Text classification
# Akbar Karimi, Leonardo Rossi, Andrea Prati
# 第二步：AEDA
import random
import  pandas as pd

random.seed(0)

# PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
PUNCTUATIONS = ['.', '!', '?', ';', ':']
# 运行的次数
# NUM_AUGS = [1, 2, 4, 8]
NUM_AUGS = [1]
PUNC_RATIO = 0.3

# Insert punction words into a given sentence with the given ratio "punc_ratio"
def insert_punctuation_marks(sentence, punc_ratio=PUNC_RATIO):
    words = sentence.split(' ')
    new_line = []
    q = random.randint(1, int(punc_ratio * len(words) + 1))
    qs = random.sample(range(0, len(words)), q)

    for j, word in enumerate(words):
        if j in qs:
            new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
            new_line.append(word)
        else:
            new_line.append(word)
    new_line = ' '.join(new_line)
    return new_line

def main(dataset):
    for aug in NUM_AUGS:
        # 用来跳过第一行
        skip_title = True
        data_aug = []
        with open(dataset + '/no_punct_data.csv', 'r') as train_orig:
            for line in train_orig:
                if skip_title:
                    data_aug.append(line)
                    skip_title = False
                    continue

                line1 = line.split(',')
                id = line1[0]
                label = line1[1]
                sentence = line1[2]
                for i in range(aug):
                    sentence_aug = insert_punctuation_marks(sentence)
                    line_aug = ','.join([id, label, sentence_aug])
                    data_aug.append(line_aug)
                data_aug.append(line)

        with open(dataset + '/data_augs_' + str(aug) + '.csv', 'w') as train_orig_plus_augs:
            train_orig_plus_augs.writelines(data_aug)


if __name__ == "__main__":
    dataset = './data'
    main(dataset)