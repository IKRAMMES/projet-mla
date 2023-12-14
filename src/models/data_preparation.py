# Function to remove empty lines and keep only the first n lines from two text files ( for quick trainning)
def keep_first_n_lines_and_remove_empty(input_file, output_file, n=2000):
    """"
    Reads the specified number of lines (n) from the input file, removes empty lines,
    and writes the cleaned lines to the output file.

    Parameters:
    - input_file (str): The path to the input text file.
    - output_file (str): The path to the output text file.
    - n (int): The number of lines to keep. Default is 2000.
    """
    # Open the input and output files with proper encoding
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:

        lines_written = 0

        # Read the first n lines from the input file and write them to the output file
        while lines_written < n:
            line = in_file.readline()
            
            # Check if the line is empty and skip it
            if not line.strip():
                continue
            
            out_file.write(line)
            lines_written += 1

# Usage of the function with appropriate file names
# keep_first_n_lines_and_remove_empty('news-commentary-v9.fr-en.en', 'fr2000.txt', n=2000)
# keep_first_n_lines_and_remove_empty('news-commentary-v9.fr-en.fr', 'ang2000.txt', n=2000)


# Function to combine data from two files
def combine_files(file1, file2, output_file):
    """
    Combines the content of two text files line by line, separated by a tabulation,
    and writes the combined lines to an output file.

    Parameters:
    - file1 (str): The path to the first input text file.
    - file2 (str): The path to the second input text file.
    - output_file (str): The path to the output text file.
    """
    # Open the input files and the output file with proper encoding
    with open(file1, 'r', encoding='utf-8') as f1, \
         open(file2, 'r', encoding='utf-8') as f2, \
         open(output_file, 'w', encoding='utf-8') as out_file:

        lines_file1 = f1.readlines()
        lines_file2 = f2.readlines()

        # Check if the number of lines in both files is the same
        if len(lines_file1) != len(lines_file2):
            print("Erreur: Le nombre de lignes dans les fichiers n'est pas le même.")
            return

        # Combine lines using a tabulation between each pair and write to the output file
        for line1, line2 in zip(lines_file1, lines_file2):
            combined_line = f"{line1.strip()}\t{line2.strip()}\n"
            out_file.write(combined_line)

# Usage of the function with appropriate file names
# combine_files('eng2000.txt', 'fra2000.txt', 'fra-eng.txt')


