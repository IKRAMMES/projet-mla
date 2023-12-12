import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_attention(attention, sentence, predicted_sentence):
    # Split the input sentences into lists of words
    sentence = sentence.split()
    predicted_sentence = predicted_sentence.split() + ['[END]']
    
    # Create a new figure
    fig = plt.figure(figsize=(10, 10))
    
    # Add a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # Crop the attention matrix to match the lengths of the sentences
    attention = attention[:len(predicted_sentence), :len(sentence)]

    # Display the attention matrix as an image with the viridis colormap
    ax.matshow(attention, cmap='viridis', vmin=0.0)

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
    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')
    
    # Set the title of the plot
    plt.suptitle('Attention weights')

