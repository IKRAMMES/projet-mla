# projet-mla
This project focuses on optimizing English-to-French machine translation by implementing a neural network based on the innovative approach presented in the paper "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE" by Bahdanau, Cho, and Bengio. The key enhancement involves introducing an attention mechanism into the decoder, allowing for dynamic information retrieval during translation.

----------------------------------------------------------------------------

# Implementation Overview:

The "src" directory encompasses three main subdirectories: "models," "tests," and "metrics." Within the "metrics" folder, we calculate metrics such as BLEU score, negative log-likelihood (NLL), and cross-entropy loss. In the "tests" directory, we conduct comprehensive testing for all modules, ensuring robust functionality. Meanwhile, the "models" directory houses the entire architecture, complete with a Jupyter Notebook file serving as a demonstration.

## Encoder:

- Utilizes an embedding layer for converting word indices into dense vectors.
- Employs a bidirectional recurrent neural network (GRU) for capturing contextual information within the source sequence.


## Decoder with Attention:

- Two versions available: basic and attention-enabled.
- Attention mechanism allows the decoder to focus on specific parts of the source sequence during target sequence generation.


## Training and Inference:

- Training involves pairs of source-target sentences with a loss function based on negative log-softmax divergence.
- During inference, the encoder encodes the source sequence, and the decoder generates the target sequence using attention.


----------------------------------------------------------------------------------

# Data:

## Dataset Exploration:

- Explored various datasets, identified errors in the WMT'14 dataset, and opted for an alternative dataset with over 167,000 sentences for each language.

## Data Processing:

- Normalized text by converting Unicode to ASCII and separating punctuation.
- Filtered sentence pairs based on a defined maximum length.
- Managed vocabulary by assigning indices to words for efficient numerical representation.
- Tokenization facilitated the numerical representation of sequences.
- Organized data in a PyTorch-compatible format for efficient model training.

-----------------------------------------------------------------------
# Bibliographie :
- Neural Machine Translation by jointly learning to align and translate : https ://arxiv.org/abs/1409.0473.
- WMT ’14 contains the following English-French parallel corpora: https://www.statmt.org/wmt14/translation-task.html
- PyTorch Tutorials : https://github.com/pytorch/tutorials/tree/main
- MEDIUM Review — Neural Machine Translation :  https://sh-tsang.medium.com/review-neural-machine-translation-by-jointly-learning-to-align-and-translate-3b381fc032e3


