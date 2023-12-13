

# supprimer les lignes vides des deux fichiers texte

'''
# Garder que les 2000 premieres lignes

def keep_first_n_lines_and_remove_empty(input_file, output_file, n=2000):
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:

        lines_written = 0

        # Lit les premières n lignes du fichier d'entrée et les écrit dans le fichier de sortie
        while lines_written < n:
            line = in_file.readline()
            
            # Vérifie si la ligne est vide et la saute
            if not line.strip():
                continue
            
            out_file.write(line)
            lines_written += 1

# Utilisation de la fonction avec les noms de fichiers appropriés

keep_first_n_lines_and_remove_empty('news-commentary-v9.fr-en.en', 'fr2000.txt', n=2000)
keep_first_n_lines_and_remove_empty('news-commentary-v9.fr-en.fr', 'ang2000.txt', n=2000)
'''

# Fonction pour combiner les deux fichiers des données
def combine_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1, \
         open(file2, 'r', encoding='utf-8') as f2, \
         open(output_file, 'w', encoding='utf-8') as out_file:

        lines_file1 = f1.readlines()
        lines_file2 = f2.readlines()

        # Vérifie si le nombre de lignes dans les deux fichiers est le même
        if len(lines_file1) != len(lines_file2):
            print("Erreur: Le nombre de lignes dans les fichiers n'est pas le même.")
            return

        # Combine les lignes en utilisant une tabulation entre chaque paire
        for line1, line2 in zip(lines_file1, lines_file2):
            combined_line = f"{line1.strip()}\t{line2.strip()}\n"
            out_file.write(combined_line)

# Utilisation de la fonction avec les noms de fichiers appropriés
combine_files('eng2000.txt', 'fra2000.txt', 'fra-eng.txt')


