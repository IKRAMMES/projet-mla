import unicodedata
import re
import torch

# Define special tokens
SOS_token = 0
EOS_token = 1

# Maximum length of sentences
MAX_LENGTH = 50


def indexesFromSentence(lang, sentence):
    """
    Returns a list of indexes corresponding to the words in the given sentence.

    Parameters:
    - lang (Lang): Language object containing word-to-index mapping.
    - sentence (str): Input sentence.

    Returns:
    - list: List of indexes.
    """
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    """
    Converts a sentence to a PyTorch tensor with EOS token appended.

    Parameters:
    - lang (Lang): Language object containing word-to-index mapping.
    - sentence (str): Input sentence.
    - device: PyTorch device.

    Returns:
    - torch.Tensor: Tensor representation of the sentence.
    """
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair,input_lang,output_lang, device):
    """
    Converts a pair of sentences to PyTorch tensors.

    Parameters:
    - pair (list): Pair of sentences [input_sentence, target_sentence].
    - input_lang (Lang): Language object for the input sentences.
    - output_lang (Lang): Language object for the target sentences.
    - device: PyTorch device.

    Returns:
    - tuple: Tuple of PyTorch tensors (input_tensor, target_tensor).
    """
    input_tensor = tensorFromSentence(input_lang, pair[0],device)
    target_tensor = tensorFromSentence(output_lang, pair[1],device)
    return (input_tensor, target_tensor)

class Lang:
    """
    Language class to manage word-to-index mapping and word counts.
    """
    def __init__(self, name):
        """
        Initializes the Lang object.

        Parameters:
        - name (str): Name of the language.
        """
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        """
        Adds words from a sentence to the language vocabulary.

        Parameters:
        - sentence (str): Input sentence.
        """
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        """
        Adds a word to the language vocabulary.

        Parameters:
        - word (str): Input word.
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataPreparation:
    """
    Class for preparing and processing language data.
    """
    def __init__(self, lang1, lang2, reverse=False):
        """
        Initializes the DataPreparation object.

        Parameters:
        - lang1 (str): Name of the first language.
        - lang2 (str): Name of the second language.
        - reverse (bool): Flag to reverse language pairs.
        """
        self.lang1=lang1
        self.lang2=lang2
        self.reverse=reverse

    def unicode_to_ascii(self, s):
        """
        Converts Unicode characters to ASCII.

        Parameters:
        - s (str): Input string.

        Returns:
        - str: ASCII representation of the input string.
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        """
        Normalizes a string by converting to lowercase, removing accents,
        and adding space before punctuation.

        Parameters:
        - s (str): Input string.

        Returns:
        - str: Normalized string.
        """
        s = self.unicode_to_ascii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def read_langs(self, lang1, lang2, reverse=False):
        """
        Reads language pairs from a file and returns language objects.

        Parameters:
        - lang1 (str): Name of the first language.
        - lang2 (str): Name of the second language.
        - reverse (bool): Flag to reverse language pairs.

        Returns:
        - tuple: Tuple containing input language object, output language object, and pairs of sentences.
        """
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

    def filter_pair(self, p):
        """
        Filters a pair of sentences based on maximum length.

        Parameters:
        - p (list): Pair of sentences [input_sentence, target_sentence].

        Returns:
        - bool: True if both sentences meet the length criteria, False otherwise.
        """
        return len(p[0].split(' ')) < MAX_LENGTH and \
               len(p[1].split(' ')) < MAX_LENGTH

    def filter_pairs(self, pairs):
        """
        Filters a list of pairs based on maximum length.

        Parameters:
        - pairs (list): List of pairs of sentences.

        Returns:
        - list: Filtered list of pairs.
        """
        return [pair for pair in pairs if self.filter_pair(pair)]

    def prepare_data(self, reverse=True):
        """
        Prepares language data, creates language objects, and prints statistics.

        Parameters:
        - reverse (bool): Flag to reverse language pairs.

        Returns:
        - tuple: Tuple containing input language object, output language object, and pairs of sentences.
        """
        self.input_lang, self.output_lang, pairs = self.read_langs(self.lang1, self.lang2, reverse)
        print("Form: ", pairs[0])
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            self.input_lang.add_sentence(pair[0])
            self.output_lang.add_sentence(pair[1])
        print("Counted words:")
        print(self.input_lang.name, self.input_lang.n_words)
        print(self.output_lang.name, self.output_lang.n_words)
        return self.input_lang, self.output_lang, pairs

