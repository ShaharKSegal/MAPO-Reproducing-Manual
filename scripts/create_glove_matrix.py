from __future__ import print_function

import os
import string
import re
import json

import numpy as np
import pandas as pd

from string import punctuation

verbose = False
skip_bad_lines = False

glove_file_path = '~/nlp/glove/glove.6B.300d.txt'
questions_tsv_file_paths = [
    '~/nlp/nsm_processed_wtq/raw_input/WikiTableQuestions/data/training.tsv']
glove_processed_dir_path = os.path.join(os.getcwd(), 'glove')
ljson_path = os.path.join(glove_processed_dir_path, 'glove_original_vocab.json')
embd_mat_path = os.path.join(glove_processed_dir_path, 'glove_original_embedding_mat.npy')


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def load_glove_model(glove_file):
    print("loading glove model")
    model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
    return model


def add_word(word, glove_dict, words_set, not_found_set, word_origin):
    possessive_suffix = "'s"

    def remove_suffix(w):
        # remove question mark in the end
        if len(w) > 0 and w[-1] == '?':
            w = w[:-1]
        # remove possible possessive form
        if w.endswith(possessive_suffix):
            w = w[:-len(possessive_suffix)]
        return w

    def add_list(original_word, lst):
        found = False
        if original_word not in lst:
            lst.append(original_word)
        for w in lst:
            if w in glove_dict:
                words_set.add(w)
                found = True
        if not found:
            if raw_word not in not_found_set and verbose:
                print('{}:{} is not in glove model'.format(word_origin, raw_word))
            not_found_set.add(raw_word)

    word = string.lower(word.strip())
    word_lst = [word]
    # word_no_suffix
    word_lst.append(remove_suffix(word))
    # strip punctuation
    word_stripped = string.lstrip(string.rstrip(word, punctuation), punctuation)
    word_lst.append(word_stripped)
    word_lst.append(remove_suffix(word_stripped))

    add_list(word, word_lst)


def load_model_from_nsm(dir_path, json_file, npy_file):
    with open(os.path.join(dir_path, json_file)) as f:
        keys = json.load(f)
    values = np.load(os.path.join(dir_path, npy_file))
    d = {}
    for i in range(len(keys)):
        d[keys[i]] = values[i]
    return d


glove_model = load_glove_model(glove_file_path)
words = set()
not_in_glove = set()

# add punctuation
for p in punctuation:
    if p in glove_model:
        words.add(p)

# add words from question tsv files
for question_file in questions_tsv_file_paths:
    print("processing question file {}".format(question_file))
    df = pd.read_csv(question_file, sep='\t', index_col=0, engine='python', error_bad_lines=skip_bad_lines, quoting=3)
    questions = df['utterance']
    for q_id, question in questions.items():
        raw_words = decontracted(question).split()
        for raw_word in raw_words:
            add_word(raw_word, glove_model, words, not_in_glove, q_id)

print("total words: {}. words not in glove: {}".format(len(words), len(not_in_glove)))

sub_glove_model = []
for word in words:
    sub_glove_model.append((word, glove_model[word]))

if not os.path.exists(glove_processed_dir_path):
    os.mkdir(glove_processed_dir_path)

print("saving word as json to {}".format(ljson_path))
with open(ljson_path, 'w') as outfile:
    json_list = [t[0] for t in sub_glove_model]
    json.dump(json_list, outfile)

print("saving glove embedding matrix as npy to {}".format(embd_mat_path))
arr = np.vstack([t[1] for t in sub_glove_model])
np.save(embd_mat_path, arr)
