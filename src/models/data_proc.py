import unicodedata
import re
import torch

endtoken = 1

MAX_LENGTH = 30


def tokenize(l, phrase):
    return [l.word2index[i] for i in phrase.split(' ')]


def tensorize_phrase(l, phrase, device):
    i = tokenize(l, phrase)
    i.append(endtoken)
    return torch.tensor(i, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(couple, i_l ,o_l, device):
    i_t = tensorize_phrase(i_l, couple[0],device)
    o_t = tensorize_phrase(o_l, couple[1],device)
    return (i_t, o_t)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataPreparation:
    def __init__(self, lang1, lang2, reverse=False):
        #self.input_lang, self.output_lang, self.pairs = self.prepare_data(reverse)
        self.lang1=lang1
        self.lang2=lang2
        self.reverse=reverse

    def normalize_string(self, s):
        s = ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"([.!?])", r" \1", s.lower().strip())
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
        
    def read_langs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def filter_pairs(self, pairs):
        return [
            pair for pair in pairs
            if len(pair[0].split(' ')) < self.MAX_LENGTH
               and len(pair[1].split(' ')) < self.MAX_LENGTH
        ]

    def prepare_data(self, reverse=True):
        self.input_lang, self.output_lang, pairs = self.read_langs(self.lang1, self.lang2, reverse)
        print("Form: ", pairs[0])
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words")
        for pair in pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        return self.input_lang, self.output_lang, pairs