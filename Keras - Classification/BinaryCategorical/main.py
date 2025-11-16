from PIL import Image
import os

path = "./PetImages/Dog"
for root, dirs, files in os.walk(path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                Image.open(os.path.join(root, file)).verify()
            except Exception as e:
                print("Corrompida:", os.path.join(root, file))
                os.remove(os.path.join(root, file))
