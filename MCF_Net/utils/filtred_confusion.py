import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. Configuration ---

# Le CSV de ground truth (3 classes)
csv_ground_truth = "/workspace/Retiniax-Quality-Grding/data/test_set.csv"

# Le CSV de prédictions filtrées (celui qu'on vient de créer)
csv_predictions_filtrees = "/workspace/Retiniax-Quality-Grding/MCF_Net/results_filtred.csv"

# Noms des nouvelles classes (pour l'affichage)
# Classe 0 = 'Good' ou 'Usable'
# Classe 1 = 'Reject'
class_names = ['Good/Usable (0+1)', 'Reject (2)']

# --- 2. Chargement des données ---

try:
    # index_col=0 pour le CSV original
    df_truth = pd.read_csv(csv_ground_truth, index_col=0) 
    
    # PAS de index_col=0 pour celui-ci, car on l'a sauvé sans index
    df_preds_filtrees = pd.read_csv(csv_predictions_filtrees) 
    
except FileNotFoundError as e:
    print(f"Erreur: Fichier non trouvé. {e}")
    print("Vérifiez les noms des variables 'csv_ground_truth' et 'csv_predictions_filtrees'.")
    exit()
except Exception as e:
    print(f"Une erreur est survenue lors de la lecture : {e}")
    exit()

# --- 3. Fusion et préparation des données ---

# Fusionner les deux DataFrames
# On utilise 'image' du CSV de vérité et 'image_name' du CSV filtré
try:
    df_merged = pd.merge(df_truth, df_preds_filtrees, left_on='image', right_on='image_name')
except KeyError:
    print("Erreur: Vérifiez les noms des colonnes 'image' et 'image_name' dans vos CSV.")
    exit()

if df_merged.empty:
    print("Erreur: La fusion n'a donné aucun résultat. Les noms d'images ne correspondent pas.")
    exit()

print(f"Nombre total d'échantillons à comparer (après filtre) : {len(df_merged)}")

# --- 4. Création des listes y_true et y_pred ---

# 4a. Préparer y_true (Ground Truth)
# On mappe les classes 0 et 1 vers la NOUVELLE classe 0
# Et la classe 2 vers la NOUVELLE classe 1
# 'quality' est la colonne du CSV original
y_true = df_merged['quality'].apply(lambda x: 0 if x in [0, 1] else 1)

# 4b. Préparer y_pred (Prédictions)
# On compare 'Proba_Classe_0_ou_1' et 'Reject'
# idxmax(axis=1) renvoie le nom de la colonne qui a la plus grande valeur
pred_cols = ['Proba_Classe_0_ou_1', 'Reject']
predicted_class_names = df_merged[pred_cols].idxmax(axis=1)

# On mappe les noms de colonnes vers nos NOUVELLES classes (0 et 1)
y_pred = predicted_class_names.map({
    'Proba_Classe_0_ou_1': 0, 
    'Reject': 1
})

# --- 5. Calcul et affichage de la matrice de confusion 2x2 ---

print("\n--- Statistiques des classes ---")
print("Vraies classes (après fusion 0+1):")
print(y_true.value_counts(normalize=True).sort_index())
print("\nClasses prédites :")
print(y_pred.value_counts(normalize=True).sort_index())

# Calculer la matrice
cm = confusion_matrix(y_true, y_pred)

print("\n--- Matrice de Confusion (Texte) ---")
print(cm)
print("--------------------------------------")
print("Lignes : Vraies classes (True Labels)")
print("Colonnes : Classes prédites (Predicted Labels)")
print(f"Classes : {class_names}")
print("--------------------------------------")

# Afficher la matrice graphiquement
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)

disp.plot(cmap=plt.cm.Blues, colorbar=True)
plt.title("Matrice de Confusion (2 Classes)")
plt.xlabel("Classe Prédite")
plt.ylabel("Vraie Classe")
plt.tight_layout()

plt.savefig("matrice_de_confusion_filtrée.png")

# Afficher le graphique
# plt.show()