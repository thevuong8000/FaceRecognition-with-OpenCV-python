import cv2
import detective
import numpy as np
import pickle

capture = cv2.VideoCapture(0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("train/trainer.yml")

labels = {}
with open("train/labels.pickle", "rb") as f:
    origin_labels = pickle.load(f)
    # reverse key-value -> value-key
    labels = {v:k for (k, v) in origin_labels.items()} 

print(labels)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # captureture frame-by-frame
    ret, frame = capture.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    frame = cv2.flip(frame, 1)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detective.face_detect(frame)

    for face in faces:
        (x, y, w, h) = face

        # face rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = frame[y: y + h, x: x + w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _id, confidence = recognizer.predict(roi_gray)
        label = "unknown person"
        if confidence <= 60:
            label = labels[_id]

        print(label, confidence)
        # put label
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
        cv2.rectangle(frame, (x, y + h), (x + label_width + 10, y + h + label_height + 10), (0, 0, 255), -1)
        cv2.putText(frame, label, (x, y + label_height + h + 3), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0))

        # eyes detection
        # eyes = detective.eye_detect(roi)
        # for (x2, y2, w2, h2) in eyes:
        #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        #     radius = int(round((w2 + h2) * 0.25))
        #     frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

        # show face only
        # cv2.imshow('roi', roi)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
