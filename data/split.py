import random
import math
import sys

def separer_csv_simple(fichier_source: str, 
                       fichier_train: str, 
                       fichier_val: str, 
                       fichier_test: str, 
                       random_seed: int = 42):
    """
    Sépare un fichier CSV en trois (train, val, test) par répartition 
    aléatoire des lignes, sans aucune bibliothèque externe.

    Répartition :
    - 75% : Train
    - 20% : Validation
    - 5%  : Test
    """
    
    print(f"Lancement de la séparation pour '{fichier_source}'...")
    
    # Initialise le générateur aléatoire pour des résultats reproductibles
    random.seed(random_seed)

    try:
        # --- 1. Lecture de toutes les lignes ---
        with open(fichier_source, 'r', encoding='utf-8') as f:
            # On garde l'en-tête (la première ligne) de côté
            # Il sera copié dans les 3 fichiers de sortie
            header = f.readline() 
            
            # On lit TOUTES les autres lignes de données
            data_lines = f.readlines()

        if not data_lines:
            print(f"Erreur : Le fichier '{fichier_source}' contient un en-tête mais pas de données.")
            return

    except FileNotFoundError:
        print(f"Erreur critique : Le fichier source '{fichier_source}' n'a pas été trouvé.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture : {e}", file=sys.stderr)
        return

    # --- 2. Mélange aléatoire des lignes ---
    random.shuffle(data_lines)
    
    # --- 3. Calcul des tailles de répartition ---
    total_count = len(data_lines)
    
    # On utilise math.ceil pour s'assurer que les petits ensembles
    # (test et val) ne soient pas arrondis à zéro si le total est faible.
    test_count = math.ceil(total_count * 0.05)
    val_count = math.ceil(total_count * 0.20)
    
    # Le train prend tout le reste (environ 75%)
    train_count = total_count - test_count - val_count

    # --- 4. Répartition des listes (slicing) ---
    # On prend les lignes dans l'ordre de la liste mélangée
    
    # Les X premières lignes pour le Test (5%)
    lignes_test = data_lines[ : test_count]
    
    # Les Y lignes suivantes pour la Validation (20%)
    lignes_val = data_lines[test_count : test_count + val_count]
    
    # Tout le reste pour l'Entraînement (75%)
    lignes_train = data_lines[test_count + val_count : ]

    # --- 5. Écriture des 3 fichiers ---
    try:
        # Fonction d'aide pour écrire un fichier
        def ecrire_fichier(nom_fichier, lignes):
            with open(nom_fichier, 'w', encoding='utf-8') as f:
                # Écrit l'en-tête en premier
                f.write(header)
                # Écrit les lignes de données
                f.writelines(lignes)

        # Écriture des 3 fichiers
        ecrire_fichier(fichier_train, lignes_train)
        ecrire_fichier(fichier_val, lignes_val)
        ecrire_fichier(fichier_test, lignes_test)
        
        # --- Rapport final ---
        print("\n--- Séparation terminée avec succès ---")
        print(f"Total initial : {total_count} lignes de données")
        print("-" * 30)
        print(f"Entraînement (~75%): {len(lignes_train)} lignes -> '{fichier_train}'")
        print(f"Validation (~20%)  : {len(lignes_val)} lignes -> '{fichier_val}'")
        print(f"Test (~5%)         : {len(lignes_test)} lignes -> '{fichier_test}'")
        print("-" * 30)
        print(f"Total réparti    : {len(lignes_train) + len(lignes_val) + len(lignes_test)} lignes")

    except Exception as e:
        print(f"Une erreur est survenue lors de l'écriture des fichiers : {e}", file=sys.stderr)


# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    
    # 1. Nom du fichier que vous voulez séparer
    # REMPLACEZ "mon_csv_complet.csv" PAR LE NOM DE VOTRE FICHIER
    FICHIER_SOURCE_REEL = "/workspace/Retiniax-Quality-Grding/data/Label_EyeQ_test.csv"

    # --- Optionnel : Création d'un faux fichier pour tester ---
    # (Juste pour que le script puisse tourner si vous n'avez pas le fichier)
    try:
        with open(FICHIER_SOURCE_REEL, 'x', encoding='utf-8') as f: # 'x' = créer (échoue s'il existe)
            print(f"Création d'un fichier de test '{FICHIER_SOURCE_REEL}'...")
            f.write(",image,quality,DR_grade\n") # L'en-tête
            for i in range(16250): # 1000 lignes de données
                f.write(f"{i},image_{i}.jpg,{i%2},{i%5}\n")
    except FileExistsError:
        print(f"Utilisation du fichier existant '{FICHIER_SOURCE_REEL}'.")
    except Exception as e:
        print(f"Impossible de créer le fichier de test : {e}")
    # -----------------------------------------------------------

    # 2. Appelez la fonction de séparation
    separer_csv_simple(
        fichier_source=FICHIER_SOURCE_REEL,
        fichier_train="/workspace/Retiniax-Quality-Grding/data/train_set.csv",
        fichier_val="/workspace/Retiniax-Quality-Grding/data/validation_set.csv",
        fichier_test="/workspace/Retiniax-Quality-Grding/data/test_set.csv"
    )