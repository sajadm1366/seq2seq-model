import re
from collections import Counter
import numpy as np
import tensorflow as tf

'''
1- Removing punctuations like . , ! $( ) * % @
2- Removing Stop words
3- Lower casing
4- 

'''

def data_gen(data, max_len):
    '''
     build tf.data generator
     inputs:
            d1: source
            d2: length source
            d3: target
            d4: length target
            max_len: length time-step
    '''
    def gen():
        for (s1, l1), (s2, l2) in data():
            yield (s1, l1), (s2, l2)

    return tf.data.Dataset.from_generator(gen, output_signature=(
         (tf.TensorSpec(shape=(max_len,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)),
         (tf.TensorSpec(shape=(max_len,), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32))
         ))



def load_data_gen(dataset_path_name, num_scenteces=190000):
    def gen():
        with open(dataset_path_name , 'r', encoding="utf-8") as f:
             for line in f:
                *txtdata, _ = line.split('\t')
                yield txtdata[0], txtdata[1]
    return gen

def replace_char(sentence, lang):
    sentence = sentence.lower()
    if lang == "eng":
            sentence = re.sub(r'\'m', ' am', sentence)
            sentence = re.sub(r'\'s', ' is', sentence)
            sentence = re.sub(r'\'ll', ' will', sentence)
            sentence = re.sub(r'\'d', ' would', sentence)
            sentence = re.sub(r'\'ve', ' have', sentence)
            sentence = re.sub(r'\'re', ' are', sentence)
            sentence = re.sub(r'don\'t', 'do not', sentence)
            sentence = re.sub(r'didn\'t', 'did not', sentence)
            sentence = re.sub(r'didn\'t', 'did not', sentence)
            sentence = re.sub(r'can\'t', 'can not', sentence)
            sentence = re.sub(r'haven\'t', 'have not', sentence)
            sentence = re.sub(r'[^a-z!?.\s]', '', sentence)
    elif lang == "fra":
            sentence = re.sub(r'\-', ' ', sentence)
            sentence = re.sub(r'[^a-zàâçéèêëîïôûùüÿñæœ!?.\s]', '', sentence)

    sentence = re.sub(r'!$', ' !', sentence)
    sentence = re.sub(r'\.$', ' .', sentence)
    sentence = re.sub(r'\?$', ' ?', sentence)

    sentence = re.split(r'[\s]', sentence)

    sentence = [e for e in sentence if len(e) != 0]

    return sentence



class GetVocab:
    def __init__(self, sentences):
        self.sentences = sentences
        self.min_word_count = 2

    def gen_to_list(self, replace_char):
        source = []
        target = []
        for s, t in self.sentences():
            source.append(replace_char(s, lang='eng'))
            target.append(replace_char(t, lang='fra'))
        return source, target

    def tr_get_vocab(self, replace_char):
        source, target = self.gen_to_list(replace_char)
        vocab_source = self.vocab(source)
        vocab_traget = self.vocab(target)
        return vocab_source, vocab_traget

    def vocab(self, data):
        words = [sp for s in data for sp in s]
        words_freq = Counter(words)

        words_freq = {key: value for (key, value) in words_freq.items() if
                      value > self.min_word_count}  # todo check here

        words_dict = {}
        words_dict['<pad>'] = 1
        words_dict['<eos>'] = 1
        words_dict['<bos>'] = 1
        words_dict['<unk>'] = 1
        words_dict.update(words_freq)

        words_dict = {key: i for i, key in enumerate(words_dict)}
        return words_dict



class ProcessSentence:

    def __init__(self, source, target, max_len):
        self.source = source
        self.target = target
        self.max_len = max_len

    def tokenize(self, sentence, lang):
        sentence = replace_char(sentence, lang)
        if lang == "eng":
            return self.word2token(sentence, self.source, self.max_len)

        elif lang == "fra":
            return self.word2token(sentence, self.target, self.max_len)

    def word2token(self, sentence, vocab, lg):
        sentence = [vocab[word] if vocab.get(word, None) != None else vocab["<unk>"] for word in sentence]
        if len(sentence) >= lg:
            return sentence[:lg - 1] + [vocab["<eos>"]], lg-1
        else:
            padded = [vocab["<pad>"]] * (lg - len(sentence) - 1)
            return sentence + [vocab["<eos>"]] + padded, len(sentence)




class Preprocess:
    def __init__(self, data, vocab_source, vocab_target, max_len):
        self.data = data

        self.process_sentence = ProcessSentence(vocab_source, vocab_target, max_len)

    def transform(self):
        for source, target in self.data():
            source_token, len_src = self.process_sentence.tokenize(source, lang='eng')
            target_token, len_tr = self.process_sentence.tokenize(target, lang='fra')
            yield (source_token, len_src), (target_token, len_tr)
