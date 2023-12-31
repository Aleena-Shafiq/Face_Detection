import cv2

# Just base function for detecting the face and counting
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function for detecting faces in an image
def face_detect(image):
    i = 0  # Counter purpose
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conversion into grayscale
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)  # Multiple face detection
    print(faces)
    print("Found {0} faces!".format(len(faces)))
    for (x, y, w, h) in faces:
        i = i + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
        cv2.putText(img, 'F-' + str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imshow("Faces found", img)
    cv2.waitKey(0)

face_detect('./test_images/t4.jpg')
