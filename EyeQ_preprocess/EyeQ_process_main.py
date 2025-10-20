import fundus_prep as prep
import glob
import os
import cv2 as cv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



import os
import cv2 as cv
# Assurez-vous que 'prep' est importé (il contient vos fonctions imread/imwrite)
# import fundus_prep as prep 

def process(image_list, save_path):
    c = 0
    c_bug = 0
    for image_path in image_list:
        c +=1
            
        # On récupère le nom du fichier sans extension
        base_filename = os.path.splitext(image_path.split('/')[-1])[0]
        
        # --- MODIFICATION ICI ---
        # On change l'extension de .jpeg à .png
        dst_image = base_filename + '.png'
        # --- FIN DE LA MODIFICATION ---
        
        dst_path = os.path.join(save_path, dst_image)
        if not os.path.exists(dst_image): 
            try :
                img = prep.imread(image_path)
                r_img, borders, mask = prep.process_without_gb(img)
                r_img = cv.resize(r_img, (800, 800))
                
                # prep.imwrite (qui utilise sûrement cv.imwrite) 
                # déduira le format .png de l'extension du fichier
                prep.imwrite(dst_path, r_img)
                
                # (Vos lignes de masque fonctionneront aussi avec la nouvelle variable dst_image)
                # mask = cv.resize(mask, (800, 800))
                # prep.imwrite(os.path.join('./original_mask', dst_image), mask)
                if c % 200 == 0 :
                    print(c)
            except :
                print('error for the ', c_bug, 'time')
                c_bug += 1

if __name__ == "__main__":

    image_list = glob.glob(os.path.join('/workspace/data/eyeq-dataset', '*.jpeg'))
    save_path = prep.fold_dir('/workspace/data/eyeq-preprocessed')

    process(image_list, save_path)







