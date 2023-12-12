import unicodedata
import re
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10 

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class Preprocessor:
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def filterPair(self, p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

class DataLoader:
    def __init__(self, lang1, lang2, reverse=False):
        self.preprocessor = Preprocessor()
        self.input_lang, self.output_lang, self.pairs = self.prepareData(lang1, lang2, reverse)

    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.preprocessor.filterPairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs

    def readLangs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
            read().strip().split('\n')

        pairs = [[self.preprocessor.normalizeString(s) for s in l.split('\t')] for l in lines]

        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def getRandomPair(self):
        return random.choice(self.pairs)
