# Fonction pour lire un fichier et renvoyer une liste de phrases sans sauts de ligne
# supprimer les lignes vides des deux fichiers texte



# Garder que les 4000 premieres lignes

def keep_first_n_lines(input_file, output_file, n=4000):
    with open(input_file, 'r', encoding='utf-8') as in_file, \
         open(output_file, 'w', encoding='utf-8') as out_file:

        # Lit les premières n lignes du fichier d'entrée et les écrit dans le fichier de sortie
        for _ in range(n):
            line = in_file.readline()
            if not line:
                break  # Fin du fichier
            out_file.write(line)

# Utilisation de la fonction avec les noms de fichiers appropriés
keep_first_n_lines('europarl-v7.fr-en.fr', 'francais4000.txt', n=4000)
keep_first_n_lines('europarl-v7.de-en.en', 'anglais4000.txt', n=4000)


def supprimer_lignes_vides(nom_fichier_entree, nom_fichier_sortie):
    with open(nom_fichier_entree, 'r') as fichier_entree:
        lignes = fichier_entree.readlines()

    # Filtrer les lignes vides
    lignes_non_vides = [ligne for ligne in lignes if ligne.strip()]

    with open(nom_fichier_sortie, 'w') as fichier_sortie:
        fichier_sortie.writelines(lignes_non_vides)

nom_fichier_entree = 'europarl-v7.de-en.en'
nom_fichier_sortie = 'outputenglais.txt'

nom_fichier_entreefr = 'europarl-v7.fr-en.fr'
nom_fichier_sortiefr = 'outputfrancais.txt'

supprimer_lignes_vides(nom_fichier_entree, nom_fichier_sortie)
supprimer_lignes_vides(nom_fichier_entreefr, nom_fichier_sortiefr)






def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# Lire les phrases du fichier anglais
english_lines = read_lines('outputenglais.txt')

# Lire les phrases du fichier français
french_lines = read_lines('outputfrancais.txt')

# Concaténer les phrases anglaises et françaises avec une tabulation
concatenated_lines = [f"{english}\t{french}" for english, french in zip(english_lines, french_lines)]

# Écrire les phrases concaténées dans un nouveau fichier
with open('true-eng-fra.txt', 'w', encoding='utf-8') as file_concatenated:
    for line in concatenated_lines:
        file_concatenated.write(line + '\n')



def correspondance_entre_fichiers(fichier1, fichier2, fichier_sortie, nombre_lignes=5000):
    with open(fichier1, 'r', encoding='utf-8') as f1, open(fichier2, 'r', encoding='utf-8') as f2, open(fichier_sortie, 'w', encoding='utf-8') as fs:
        # Lire les premières 5000 lignes de chaque fichier
        lignes_f1 = f1.readlines()[:nombre_lignes]
        lignes_f2 = f2.readlines()[:nombre_lignes]

        # S'assurer que les deux fichiers ont le même nombre de lignes
        if len(lignes_f1) != len(lignes_f2):
            raise ValueError("Les fichiers n'ont pas le même nombre de lignes.")

        # Écrire les lignes correspondantes séparées par un espace constant
        for ligne_f1, ligne_f2 in zip(lignes_f1, lignes_f2):
            ligne_combinee = f"{ligne_f1.strip()} {ligne_f2.strip()}\n"
            fs.write(ligne_combinee)

# Exemple d'utilisation
fichier1 = 'outputenglais.txt'
fichier2 = 'outputfrancais.txt'
fichier_sortie = 'fichier_combine.txt'

correspondance_entre_fichiers(fichier1, fichier2, fichier_sortie, nombre_lignes=5000)
