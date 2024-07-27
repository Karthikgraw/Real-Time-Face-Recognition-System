import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\nEnter user id and press <return>: ')

print("\n[INFO] Initializing face capture. Look at the camera and wait...")
# Initialize individual sampling face count
count = 0

while True:
    ret, img = cam.read()
    img = cv2.rotate(img, cv2.ROTATE_180)  # rotate image by 180 degrees
    img = cv2.flip(img, 0)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face samples and stop video
        break

# Cleanup
print("\n[INFO] Exiting Program and cleaning up")
cam.release()
cv2.destroyAllWindows()
