import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# REMPLACEZ AVEC VOS NOMS DE FICHIERS
csv_ground_truth = "/workspace/Retiniax-Quality-Grding/data/test_set.csv"
csv_predictions = "/workspace/Retiniax-Quality-Grding/MCF_Net/resultDenseNet121_v3_v1.csv"

# Définition des colonnes de probabilités
proba_columns = ['Good', 'Usable', 'Reject']

# Mapping des noms de colonnes aux labels numériques (0, 1, 2)
# Assurez-vous que cet ordre correspond à vos classes !
label_map = {
    'Good': 0,
    'Usable': 1,
    'Reject': 2
}
# Noms des classes pour l'affichage (dans le bon ordre)
class_names = ['Good (0)', 'Usable (1)', 'Reject (2)']

# --- 2. Chargement des données ---
try:
    # index_col=0 gère la première colonne "," qui est probblement un index
    df_truth = pd.read_csv(csv_ground_truth, index_col=0)
    df_preds = pd.read_csv(csv_predictions, index_col=0)
except FileNotFoundError:
    print(f"Erreur: Assurez-vous que les fichiers '{csv_ground_truth}' et '{csv_predictions}' existent.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors de la lecture des CSV : {e}")
    exit()

# --- 3. Traitement et fusion ---

# Fusionner les deux DataFrames sur la base du nom de l'image
# 'image' pour le CSV de vérité, 'image_name' pour le CSV de prédictions
try:
    df_merged = pd.merge(df_truth, df_preds, left_on='image', right_on='image_name')
except KeyError:
    print("Erreur: Vérifiez les noms des colonnes 'image' et 'image_name' dans vos CSV.")
    exit()

if df_merged.empty:
    print("Erreur: La fusion n'a donné aucun résultat. Vérifiez que les noms d'images correspondent.")
    exit()

# Extraire les vrais labels
y_true = df_merged['quality']

# --- 4. Obtenir les prédictions ---

# 1. Trouver le nom de la colonne avec la plus haute probabilité (argmax)
# axis=1 signifie qu'on cherche le max sur chaque ligne
predicted_class_names = df_merged[proba_columns].idxmax(axis=1)

# 2. Convertir ces noms ('Good', 'Usable', 'Reject') en labels numériques (0, 1, 2)
y_pred = predicted_class_names.map(label_map)

# --- 5. Calcul et affichage de la matrice de confusion ---

print("Données prêtes pour la matrice de confusion.")
print(f"Nombre total d'échantillons : {len(y_true)}")

# Calculer la matrice
cm = confusion_matrix(y_true, y_pred)

print("\n--- Matrice de Confusion (Texte) ---")
print(cm)
print("--------------------------------------")
print("Lignes : Vraies classes (True Labels)")
print("Colonnes : Classes prédites (Predicted Labels)")
print("--------------------------------------")

# Afficher la matrice graphiquement
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

disp.plot(cmap=plt.cm.Blues, colorbar=True)
plt.title("Matrice de Confusion")
plt.xlabel("Classe Prédite")
plt.ylabel("Vraie Classe")
plt.tight_layout()

# Sauvegarder l'image (optionnel)
plt.savefig("matrice_de_confusion.png")

# Afficher le graphique
# plt.show()