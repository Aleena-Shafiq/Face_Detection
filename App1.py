import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image

# Load the Haar Cascades model for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the SSD model for webcam face detection
prototxt = "deploy.prototxt"
model = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Threshold for face detection confidence
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (95, 207, 30), 3)
                cv2.rectangle(img, (x, y - 40), (x1, y), (95, 207, 30), -1)
                cv2.putText(img, 'F-' + str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img

def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count = len(faces)
        cv2.rectangle(image, (x, y), (x + w, y + h), (95, 207, 30), 3)
        cv2.rectangle(image, (x, y - 40), (x + w, y), (95, 207, 30), -1)
        cv2.putText(image, 'F-' + str(count), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return image, count

def run():
    st.set_page_config(
        page_title="Face Detection using Haar Cascades and SSD",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Face Detection using Haar Cascades and SSD")
    st.markdown(
        '<style>h1{color: #d73b5c;text-align:center;}</style>',
        unsafe_allow_html=True
    )

    activities = ["Image", "Webcam"]
    st.sidebar.markdown("# Choose Input Source")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Aleena Shafiq](http://github.com/spidy20)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    if choice == 'Image':
        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Face Detection is done using Haar Cascades & SSD"</h4>''',
                    unsafe_allow_html=True)
        img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
        if img_file is not None:
            img = np.array(Image.open(img_file))
            img1 = cv2.resize(img, (350, 350))
            place_h = st.columns(2)
            place_h[0].image(img1, caption='Original Image', use_column_width=True)
            fd, count = face_detect(img)  # Face detection for image (Haar Cascades)

            # Resize the detected face image to match the size of the original image
            fd_resized = cv2.resize(fd, (350, 350))
            place_h[1].image(fd_resized, caption=f'Detected Faces ({count} faces)', use_column_width=True)

            if count == 0:
                st.error("No People found!!")
            else:
                st.success("Total number of People : " + str(count))
    if choice == 'Webcam':
        st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* It might not work with Android Camera"</h4>''',
                    unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

run()