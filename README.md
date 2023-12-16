# projet-mla
This project focuses on optimizing English-to-French machine translation by implementing a neural network based on the innovative approach presented in the paper "NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE" by Bahdanau, Cho, and Bengio. The key enhancement involves introducing an attention mechanism into the decoder, allowing for dynamic information retrieval during translation.

----------------------------------------------------------------------------

Implementation Overview:
Encoder:

Utilizes an embedding layer for converting word indices into dense vectors.
Employs a bidirectional recurrent neural network (GRU) for capturing contextual information within the source sequence.
Decoder with Attention:

Two versions available: basic and attention-enabled.
Attention mechanism allows the decoder to focus on specific parts of the source sequence during target sequence generation.
Training and Inference:

Training involves pairs of source-target sentences with a loss function based on negative log-softmax divergence.
During inference, the encoder encodes the source sequence, and the decoder generates the target sequence using attention.
Differences from the Paper:

Incorporation of word embeddings in RNNs for improved text processing and semantic understanding.
Decision not to implement beam search to balance decoding complexity and output quality.

----------------------------------------------------------------------------------

Data:
Dataset Exploration:
Explored various datasets, identified errors in the WMT'14 dataset, and opted for an alternative dataset with over 167,000 sentences for each language.

Data Processing:
Normalized text by converting Unicode to ASCII and separating punctuation.
Filtered sentence pairs based on a defined maximum length.
Managed vocabulary by assigning indices to words for efficient numerical representation.
Tokenization facilitated the numerical representation of sequences.
Organized data in a PyTorch-compatible format for efficient model training.


