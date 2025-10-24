import pandas as pd

# --- Paramètres à configurer ---

# Le fichier CSV de prédictions (en entrée)
FICHIER_PREDICTIONS = "/workspace/Retiniax-Quality-Grding/MCF_Net/resultDenseNet121_v3_v1.csv" 

# Le nouveau fichier CSV (en sortie)
FICHIER_SORTIE = "/workspace/Retiniax-Quality-Grding/MCF_Net/results_filtred.csv"

# Les noms des colonnes de proba pour les classes 0 et 1
COLS_CLASSE_0_1 = ['Good', 'Usable']

# Le nom de la colonne pour la classe 2
COL_CLASSE_2 = 'Reject'

# Le seuil pour filtrer. 
# Je mets 0.4 car votre 2ème condition (< 0.4) est plus stricte 
# que la première (< 0.5).
SEUIL_REJECT = 0.5

# --- Fin des paramètres ---


try:
    # Charger le CSV de prédictions, en supposant que la première colonne est un index
    df = pd.read_csv(FICHIER_PREDICTIONS, index_col=0)
except FileNotFoundError:
    print(f"Erreur: Le fichier '{FICHIER_PREDICTIONS}' n'a pas été trouvé.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue : {e}")
    exit()

# 1. Filtrer le DataFrame
# On garde uniquement les lignes où la proba 'Reject' est < SEUIL_REJECT
df_filtre = df[df[COL_CLASSE_2] < SEUIL_REJECT].copy()

# 2. Créer la nouvelle colonne en sommant les probas 0 et 1
# .sum(axis=1) fait la somme par ligne
df_filtre['Proba_Classe_0_ou_1'] = df_filtre[COLS_CLASSE_0_1].sum(axis=1)

# 3. Sélectionner les colonnes à sauvegarder
# On garde le nom de l'image, la nouvelle proba sommée, et la proba de classe 2
colonnes_a_garder = ['image_name', 'Proba_Classe_0_ou_1', COL_CLASSE_2]

# S'assurer que 'image_name' existe, sinon chercher 'image'
if 'image_name' not in df_filtre.columns:
    if 'image' in df_filtre.columns:
        colonnes_a_garder[0] = 'image' # Remplacer 'image_name' par 'image'
    else:
        print("Erreur: Impossible de trouver la colonne 'image_name' ou 'image'.")
        exit()

df_final = df_filtre[colonnes_a_garder]

# 4. Sauvegarder le nouveau CSV
df_final.to_csv(FICHIER_SORTIE, index=False) # index=False pour ne pas garder l'ancien index

print(f"Filtre terminé !")
print(f"Nombre d'images initiales : {len(df)}")
print(f"Nombre d'images gardées (P(Reject) < {SEUIL_REJECT}) : {len(df_final)}")
print(f"Nouveau CSV sauvegardé sous : {FICHIER_SORTIE}")
print("\n--- Aperçu des 5 premières lignes du nouveau fichier ---")
print(df_final.head())