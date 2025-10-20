import boto3
import os

# --- Configuration S3 (Source) ---
AWS_ACCESS_KEY_ID = '35fc26da4a55434a8b3b83140637367f' 
AWS_SECRET_ACCESS_KEY = 'a943b0328afe40eabce9e8eb2eea9d48'
ENDPOINT_URL = "https://s3.sbg.io.cloud.ovh.net" 
SOURCE_BUCKET = "eyeq-dataset" 

# --- CORRECTION DE CHEMIN : AJOUT DU SLASH INITIAL ---
# Chemin de destination DANS votre Notebook (doit commencer par /workspace/)
LOCAL_DESTINATION_DIR = "/workspace/data/eyeq-dataset/" 

# --- Préparation : Création du dossier de destination ---
# Crée le dossier local s'il n'existe pas. 'exist_ok=True' évite les erreurs si le dossier est déjà là.
try:
    os.makedirs(LOCAL_DESTINATION_DIR, exist_ok=True)
    print(f"Dossier de destination créé ou confirmé : {LOCAL_DESTINATION_DIR}")
except Exception as e:
    print(f"Erreur lors de la création du dossier local : {e}")
    exit(1) # Arrête le script si le dossier ne peut pas être créé

# Connexion au client S3
s3_client = boto3.client(
    's3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# --- Processus de téléchargement ---
try:
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=SOURCE_BUCKET)

    compteur = 0
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                object_key = obj['Key']
                
                # Vérifie si l'objet est un fichier .jpeg
                if object_key.lower().endswith('.jpeg'):
                    
                    # Définit le chemin de destination local complet
                    local_path = os.path.join(LOCAL_DESTINATION_DIR, os.path.basename(object_key))
                    
                    # Téléchargement
                    s3_client.download_file(SOURCE_BUCKET, object_key, local_path)
                    print(f"Téléchargé : {object_key} -> {local_path}")
                    compteur += 1

    print(f"\n✅ Transfert terminé. {compteur} fichiers jpeg téléchargés.")

except ClientError as e:
    # Gère les erreurs spécifiques à l'API S3 (clés incorrectes, bucket non trouvé)
    print(f"\n❌ Erreur S3 (boto3) : {e}")
except Exception as e:
    print(f"\n❌ Une erreur inattendue s'est produite : {e}")