import json
from utils.lcgn_util.text_processing import tokenize_clevr
from collections import defaultdict
import os
import numpy as np


# We collect vocabulary and answers from the (unbalanced) full training set
# question_files = [
#     './imdb/imdb_flickr_train.npy',
#     './imdb/imdb_flickr_val.npy'
# ]
vocab_file = './vocabulary_flickr.txt'
#
#
# vocab_count = defaultdict(int)
# for question_file in question_files:
#     print('loading ' + question_file)
#     f = np.load(question_file, allow_pickle=True)
#     # with open(question_file) as f:
#     #     questions = json.load(f)['refexps']
#     for q in f:
#         # words = tokenize_clevr(q['question'])
#         words = q['question'].split(' ')
#         for w in words:
#             if w == '#9':
#                 print(w)
#                 print(words)
#                 print(q['image_name'])
#                 exit()
#             vocab_count[w] += 1
#
# sorted_vocab = ['<pad>', '<unk>', '<start>', '<end>'] + sorted(vocab_count)
# with open(vocab_file, 'w') as f:
#     for w in sorted_vocab:
#         print(w)
#         f.write(w+'\n')


from utils.utils import *
print("Building Vocabulary")
vocabulary = ['<pad>', '<start>', '<end>']
phases = ['train/', 'val/']
for phase in phases:
    m = 1
    ids = [f[:-4] for f in os.listdir('./processed/sentences/' + phase) if f.endswith('.txt')]
    for img_n, img_id in enumerate(ids):
        sentence_path = './processed/sentences/' + phase + img_id + '.txt'
        corefData = get_sentence_data(sentence_path)
        for description in corefData:
            for word in description['sentence'].lower().split(' '):
                if word not in vocabulary:
                    vocabulary.append(word)
                    m += 1

vocabulary.append('<unk>')
with open(vocab_file, 'w', encoding='utf-8') as f:
    for w in vocabulary:
        print(w)
        f.write(w+'\n')

# word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
# idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}
# with open(data_folder + 'vocabulary.pkl', 'wb') as f:
#     pickle.dump((word2idx, idx2word), f, protocol=pickle.HIGHEST_PROTOCOL)