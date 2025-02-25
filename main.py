import cv2
fascaet = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# Start the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show the live video feed
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face = fascaet.detectMultiScale(grey,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0), 2)
    cv2.imshow("face Detection",frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
