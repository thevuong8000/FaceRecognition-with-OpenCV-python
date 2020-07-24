import cv2
import time

# Capture images through the camera
capture = cv2.VideoCapture(0)

# The size of displayed frames
frame_height = 520
frame_width = 640

if not capture.isOpened():
    print("Cannot open camera")
    exit()

# Time delayed between frames displayed(seconds)
DELAY = 2.5 

# The start time
lastTimeSaved = time.time()

# Initialize
loading = [True] + [False] * 5
labels = ['frame1', 'gray1', 'frame2', 'gray2', 'frame3', 'gray3']

ret, frame = capture.read()
frames = [frame] * 6
curIdx = 0

while True:
    
    # captureture frame-by-frame
    ret, frame = capture.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    gray = cv2.flip(gray, 1)

    # Set position for displaying frames
    cv2.moveWindow('frame1', 0, 0)
    cv2.moveWindow('gray1', 0, frame_height)
    cv2.moveWindow('frame2', frame_width, frame_height)
    cv2.moveWindow('gray2', frame_width, 0)
    cv2.moveWindow('frame3', 2 * frame_width, 0)
    cv2.moveWindow('gray3', 2 * frame_width, frame_height)

    # Get current execution time
    curTime = time.time()

    # Update frames
    for i in range(len(frames)):
        if i & 1:
            frames[i] = gray
        else:
            frames[i] = frame

    # Check if the next frame should appear
    if curTime - lastTimeSaved >= DELAY and curIdx < 6:
        curIdx += 1
        lastTimeSaved = curTime
        loading = [False] * 7
        loading[curIdx] = True
    
    # If all frames are displayed
    if curIdx == 6:
        loading = [True] * 6

    # Display frames
    for i in range(len(frames)):
        if loading[i]:
            cv2.imshow(labels[i], frames[i])

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the captureture
capture.release()
cv2.destroyAllWindows()