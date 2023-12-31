import collections

import nltk
import torch

import matplotlib.pyplot as plt


class BLEUScoreNLTK:
    """
    Class for calculating BLEU score.

    Attributes:
        reference_tensor (torch.Tensor): Reference tensor with dimensions (batch_size, Tx).
        candidate_tensor (torch.Tensor): Candidate tensor with dimensions (batch_size, Ty).
    """

    def __init__(self, reference_tensor, candidate_tensor):
        self.reference_tensor = reference_tensor
        self.candidate_tensor = candidate_tensor
        self.bleu_scores = []

    def calculate_bleu_score(self):
        """
        Calculate the BLEU score between references and candidates for each pair in the same order.
        """

        for reference_sequence, candidate_sequence in zip(self.reference_tensor, self.candidate_tensor):
            reference_list = list(reference_sequence.numpy())
            candidate_list = list(candidate_sequence.numpy())

            # Calculate BLEU score for each pair
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference_list], candidate_list)
            self.bleu_scores.append(BLEUscore)

        return torch.tensor(self.bleu_scores)


class BLEUScore:

    """
    Class for calculating BLEU score.

    Attributes:
        reference_tensor (torch.Tensor): Reference tensor with dimensions (batch_size, Tx).
        candidate_tensor (torch.Tensor): Candidate tensor with dimensions (batch_size, Ty).
    """

    def __init__(self, reference_tensor, candidate_tensor, n=2):
        self.reference_tensor = reference_tensor
        self.candidate_tensor = candidate_tensor
        self.n = n  # number of n-grams
        self.bleu_scores = []

    def calculate_bleu_score(self):
        """
        Calculate the BLEU score between references and candidates for each pair in the same order.
        """
        for reference_sequence, candidate_sequence in zip(self.reference_tensor, self.candidate_tensor):
            reference_list = list(reference_sequence.numpy())
            candidate_list = list(candidate_sequence.numpy())

            # Calculate n-grams for reference and candidate
            reference_ngrams = [tuple(reference_list[i : i + self.n]) for i in range(len(reference_list) - self.n + 1)]
            candidate_ngrams = [tuple(candidate_list[i : i + self.n]) for i in range(len(candidate_list) - self.n + 1)]

            # Count n-grams in the reference
            reference_ngram_counts = collections.Counter(reference_ngrams)

            # Count matching n-grams in the candidate that match the reference
            matching_ngram_counts = sum(min(reference_ngram_counts[ngram], candidate_ngrams.count(ngram)) for ngram in set(candidate_ngrams))

            # Calculate BLEU score
            BLEUscore = matching_ngram_counts / len(candidate_ngrams)

            self.bleu_scores.append(BLEUscore)

        return torch.tensor(self.bleu_scores)
    
def plot_bleu_scores(reference_tensors, candidate_tensors):
    bleu_calculator = BLEUScoreNLTK(reference_tensors, candidate_tensors)
    bleu_scores = bleu_calculator.calculate_bleu_score()

    sentence_lengths = [len(ref) for ref in reference_tensors]

    plt.plot(sentence_lengths, bleu_scores,linestyle='-', label="BLEU Score")
    plt.xlabel("Sentence Length")
    plt.ylabel("BLEU Score")
    plt.legend(loc="lower left")
    plt.show()




reference_tensors = [
    torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
    torch.tensor([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]),
]
candidate_tensors = [
    torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    torch.tensor([13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
    torch.tensor([27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]),
]


plot_bleu_scores(reference_tensors, candidate_tensors)