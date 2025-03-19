
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load the trained emotion model
model = tf.keras.models.load_model("best_emotion_recognition_model.h5")

# Emotion labels (adjust based on your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Set Streamlit page title
st.title("🎭 Real-Time Emotion Detection")

# Sidebar selection for input type
st.sidebar.write("📷 **Choose Input Mode**")
option = st.sidebar.radio("Select:", ("Upload Image", "Use Webcam"))

# Function to preprocess image for the model
def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (48, 48))  # Resize for model
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict emotion from an image
def predict_emotion(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index], float(np.max(predictions))

# Function to detect faces and return face count
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))
    return faces

# Function to process images (for uploaded images)
def process_image(image):
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV format
    faces = detect_faces(image_cv)

    face_count_placeholder = st.empty()  # Placeholder for live updates

    if len(faces) == 0:
        face_count_placeholder.warning("🚨 No face detected in the image. Please upload an image with a visible face.")
    else:
        face_count_placeholder.success(f"✅ **{len(faces)} person(s) detected in the image.**")

        detected_emotions = []  # Store emotions for structured display
        for idx, (x, y, w, h) in enumerate(faces):
            face = image_cv[y:y+h, x:x+w]  # Crop the detected face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            emotion, confidence = predict_emotion(face_rgb)

            # Store detected emotions for display
            detected_emotions.append(f"🧑 Person {idx+1} = {emotion} ({confidence:.2f})")

            # Draw a rectangle around the face
            cv2.rectangle(image_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_cv, f"{emotion}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert back to PIL for display
        image_display = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        st.image(image_display, caption="Processed Image", width=400)  # Reduce image size

        # Display detected emotions below
        st.write("\n".join(detected_emotions))

# Function to process video (for live webcam)
def process_video():
    cap = cv2.VideoCapture(0)  # Open webcam
    frame_placeholder = st.empty()  # Placeholder for video frames
    face_count_placeholder = st.empty()  # Placeholder for face count
    emotion_placeholder = st.empty()  # Placeholder for detected emotions
    stop_button = st.button("Stop Webcam")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to capture webcam image.")
            break

        faces = detect_faces(frame)

        if len(faces) == 0:
            face_count_placeholder.warning("🚨 No face detected! Please move into the frame.")
            emotion_placeholder.empty()  # Clear emotions
        else:
            face_count_placeholder.success(f"✅ **{len(faces)} person(s) detected in the frame.**")

            detected_emotions = []  # Store detected emotions
            for idx, (x, y, w, h) in enumerate(faces):
                face = frame[y:y+h, x:x+w]  # Extract face ROI
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                emotion, confidence = predict_emotion(face_rgb)

                # Store detected emotions
                detected_emotions.append(f"🧑 Person {idx+1} = {emotion} ({confidence:.2f})")

                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            frame_placeholder.image(frame, channels="BGR", caption="Live Emotion Detection", width=600)  # Reduce size
            emotion_placeholder.write("\n".join(detected_emotions))  # Update emotions

    cap.release()
    cv2.destroyAllWindows()

### **1️⃣ Upload Image Mode**
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)  # Reduce image size

        if st.button("Predict Emotion"):
            process_image(image)

### **2️⃣ Live Webcam Mode**
elif option == "Use Webcam":
    st.write("📷 **Webcam Live Detection**")
    process_video()
