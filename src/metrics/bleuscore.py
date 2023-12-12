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
    
    def plot_bleu_scores(self, sentence_lengths, plotting_style="-"):

        """
        Plot BLEU scores against sentence lengths.

        Args:
            sentence_lengths (list): List of sentence lengths.
            plotting_style (str): Line style for the plot.

        Returns:
            None
        """

        plt.plot(sentence_lengths, self.bleu_scores, plotting_style, label="BLEU Score")
        plt.xlabel("Sentence Length")
        plt.ylabel("BLEU Score")
        plt.legend(loc="lower left")
        plt.show()

import random

# Number of iterations
num_iterations = 5

# Your training or generation loop
for iteration in range(1, num_iterations + 1):
    # Example: Generate random reference and candidate tensors for the current iteration
    reference_tensor = torch.randint(1, 10, size=(random.randint(3, 8),))  # Random length between 3 and 8
    candidate_tensor = torch.randint(1, 10, size=(random.randint(3, 8),))  # Random length between 3 and 8

    # Instantiate the BLEUScore class for the current iteration
    bleu_calculator = BLEUScore(reference_tensor ,candidate_tensor , n=2)

    # Calculate BLEU scores
    bleu_scores = bleu_calculator.calculate_bleu_score()

    

    # Assuming sentence_lengths is a list of corresponding sentence lengths for the current iteration
    sentence_lengths = [len(reference_tensor), len(candidate_tensor)]

    # Print some information about the current iteration
    print(f"Iteration {iteration}:")
    print(f"Reference Tensor: {reference_tensor.tolist()}")
    print(f"Candidate Tensor: {candidate_tensor.tolist()}")
    print(f"Sentence Lengths: {sentence_lengths}")
    print(f"BLEU Scores: {bleu_scores.tolist()}")

    # Plot BLEU scores against sentence lengths
    bleu_calculator.plot_bleu_scores(sentence_lengths)