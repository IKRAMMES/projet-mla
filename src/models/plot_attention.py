import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_attention(attention, sentence, predicted_sentence):
    # Convertir le tensor d'attention en tableau NumPy
    attention = attention.detach().cpu().numpy()

    # Normaliser les valeurs pour une meilleure visualisation
    attention = attention / attention.sum(axis=1, keepdims=True)

    # Split the input sentences into lists of words
    sentence = sentence.split()
    predicted_sentence = predicted_sentence.split()
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))
    
    # Add a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # Décalage de deux éléments à gauche dans la matrice d'attention
    shifted_attention = np.roll(attention, shift=-2, axis=1)

    # Crop the shifted attention matrix to match the lengths of the sentences
    shifted_attention = shifted_attention[:len(predicted_sentence), :len(sentence)]

    # Display the shifted attention matrix as an image with the viridis colormap
    im=ax.matshow(shifted_attention, cmap='viridis', vmin=0.0, vmax=1.0)

    # Define the font properties for axis labels
    fontdict = {'fontsize': 14}

    # Set tick labels and rotation for the x-axis (input sentence)
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    
    # Set tick labels for the y-axis (predicted sentence)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    # Set major locators for ticks on both axes
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # Set axis labels
    ax.set_xlabel('Input sentence')
    ax.set_ylabel('Output sentence')
    
    # Set the title of the plot
    plt.suptitle('Attention weights')

  # Ajouter une barre de couleur
    cbar = plt.colorbar(im)
    cbar.set_label('Attention', color='white')