# function for finding or identify faces
import cv2

face_cascade = cv2.CascadeClassifier("../cascade/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("../cascade/haarcascade_eye_tree_eyeglasses.xml")

def face_detect(image):
    # try to convert image to grayscale if it's not in grayscale
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    
    faces = face_cascade.detectMultiScale(gray)
    return faces

def eye_detect(face):
    # try to convert image to grayscale if it's not in grayscale
    try:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    except:
        gray = face
    eyes = eye_cascade.detectMultiScale(gray)
    return eyes

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    while True:
        # captureture frame-by-frame
        ret, frame = capture.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv2.flip(frame, 1)
        
        # faces detection
        faces = face_detect(frame)

        for face in faces:
            (x, y, w, h) = face
            # print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = frame[y: y + h, x: x + w]

            # eyes detection
            eyes = eye_detect(roi)
            for (x2, y2, w2, h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2) * 0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0), 4)

            # cv2.imshow('roi', roi_gray)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
