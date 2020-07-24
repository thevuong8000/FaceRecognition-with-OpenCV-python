import os
import numpy as np
import cv2
import detective
import pickle
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(BASE_DIR, os.pardir)
image_dir = os.path.join(os.path.abspath(os.path.join(src_dir, os.pardir)), "images")
# print(image_dir)

known_image = os.path.join(image_dir, "known")

cur_id = 0
label_to_id = {}
ids = []
faces_train = []

for root, dirs, files in os.walk(known_image):
    for file in files: 
        if file.endswith('jpg') or file.endswith('png'):
            path = os.path.join(root, file)
            label = os.path.basename(root)
            print(path, ":", label)
            
            # get id for the next face
            if label not in label_to_id:
                label_to_id[label] = cur_id
                cur_id += 1
            face_id = label_to_id[label]

            # pil_imge -> gray -> numpy array 
            pil_image = Image.open(path).convert("L") # grayscale
            image_arr = np.asarray(pil_image, dtype=np.uint8)
            # image_arr = cv2.imread(path)

            # detect face in the image
            faces = detective.face_detect(image_arr)
            
            for face in faces:
                (x, y, w, h) = face
                roi = image_arr[y: y + h, x: x + w]
                faces_train.append(roi)
                ids.append(face_id)

            
            # print(type(test_image), type(pil_image))


if not os.path.exists("./train"):
    os.mkdir("train")

with open("train/labels.pickle", "wb") as f:
    pickle.dump(label_to_id, f)

recognizer.train(faces_train, np.array(ids))
recognizer.save("train/trainer.yml")