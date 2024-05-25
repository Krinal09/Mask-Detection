import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def mask_image(image_path):
    # Load our serialized face detector model from disk
    base_path = 'face_detector'
    prototxtPath = os.path.join(base_path, 'deploy.prototxt')
    weightsPath = os.path.join(base_path, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        st.error("Face detector model files not found. Please check the paths.")
        return None

    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # Load the face mask detector model from disk
    model = load_model("mask_detector.h5")

    # Load the input image from disk and grab the image spatial dimensions
    image = cv2.imread(image_path)
    if image is None:
        st.error("Image not found. Please upload an image first.")
        return None
    
    (h, w) = image.shape[:2]

    # Construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (238, 130, 238)  # Violet color

            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    
    return image

def mask_detection():
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    activities = ["Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            # Ensure the images directory exists
            if not os.path.exists('./images'):
                os.makedirs('./images')
            our_image.save('./images/out.jpg')
            st.image(image_file, caption='', use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                result_image = mask_image('/images/out.jpg')
                if result_image is not None:
                    st.image(result_image, use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        run_webcam()

def run_webcam():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    
    base_path = 'face_detector'
    prototxtPath = os.path.join(base_path, 'deploy.prototxt')
    weightsPath = os.path.join(base_path, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(prototxtPath) or not os.path.exists(weightsPath):
        st.error("Face detector model files not found. Please check the paths.")
        return

    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.h5")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                (mask, withoutMask) = model.predict(face)[0]

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (238, 130, 238)

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

mask_detection()
