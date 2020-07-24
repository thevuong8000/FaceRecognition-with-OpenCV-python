import cv2
capture = cv2.VideoCapture(0)

frame_height = 520
frame_width = 640

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
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 1)
    # Set position for displaying frames
    cv2.moveWindow('frame1', 0, 0)
    cv2.moveWindow('gray1', 0, frame_height)
    cv2.moveWindow('frame2', frame_width, frame_height)
    cv2.moveWindow('gray2', frame_width, 0)
    cv2.moveWindow('frame3', 2 * frame_width, 0)
    cv2.moveWindow('gray3', 2 * frame_width, frame_height)

    # Get all frames into an array to display
    frames = [
        ('frame1', frame),
        ('gray1', gray),
        ('frame2', frame),
        ('gray2', gray),
        ('frame3', frame),
        ('gray3', gray)
    ]

    # Display the resulting frame
    # print(cv2.getWindowImageRect('frame')) # 640 x 480
    for frame in frames:
        cv2.imshow(frame[0], frame[1])

    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the captureture
capture.release()
cv2.destroyAllWindows()