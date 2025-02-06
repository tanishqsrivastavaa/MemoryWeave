import io
import cv2
import json
import os
import streamlit as st
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from datetime import datetime
from agent import get_initial_agent
import asyncio

# Check Streamlit requirements
check_requirements("streamlit>=1.29.0")

# Ensure storage folder exists
CAPTURED_IMAGES_DIR = "static/captured"
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
JSON_FILE = "captured_images.json"
CLASS_NAMES_FILE = "captured_class_names.json"  # JSON file for detected class names

# Initialize YOLO model
def load_model(model_path):
    return YOLO(model_path)

def save_image(image, detections, class_names):
    """Save images with highest confidence per class and update detected class names."""
    highest_conf_per_class = {}
    detected_class_ids = set()  # Store only detected class IDs
    
    # Find the highest confidence detection per class
    for detection in detections:
        class_id = int(detection.cls[0])  # Get class ID
        conf_score = float(detection.conf[0])  # Get confidence score
        
        if conf_score > 0.7 and (class_id not in highest_conf_per_class or conf_score > highest_conf_per_class[class_id]["conf"]):
            highest_conf_per_class[class_id] = {"conf": conf_score, "image": image.copy()}
            detected_class_ids.add(class_id)  # Store only detected class IDs
    
    if not highest_conf_per_class:
        return  # Skip saving if no confident detections
    
    # Load existing image data
    try:
        with open(JSON_FILE, "r") as file:
            images = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        images = {}

    # Update JSON and save images
    for class_id, data in highest_conf_per_class.items():
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
        image_path = os.path.join(CAPTURED_IMAGES_DIR, f"class_{class_id}_{timestamp}.jpg")
        cv2.imwrite(image_path, data["image"])
        images[class_id] = {"path": image_path, "conf": data["conf"]}

    # Save the updated JSON with captured images
    with open(JSON_FILE, "w") as file:
        json.dump(images, file, indent=4)

    # Convert detected class IDs to class names
    if not detected_class_ids:
        detected_class_names = []
    else:
        detected_class_names = list(set(class_names[class_id] for class_id in detected_class_ids if class_id < len(class_names)))

    # Load existing class names
    try:
        with open(CLASS_NAMES_FILE, "r") as file:
            existing_class_names = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_class_names = []

    # Merge detected classes with existing ones and ensure uniqueness
    updated_class_names = list(set(existing_class_names + detected_class_names))

    with open(CLASS_NAMES_FILE, "w") as file:
        json.dump(updated_class_names, file, indent=4)

def reset_captured_images():
    """Delete all captured images and reset JSON files."""
    if os.path.exists(JSON_FILE):
        os.remove(JSON_FILE)
    if os.path.exists(CLASS_NAMES_FILE):
        os.remove(CLASS_NAMES_FILE)
    for file in os.listdir(CAPTURED_IMAGES_DIR):
        os.remove(os.path.join(CAPTURED_IMAGES_DIR, file))
    st.success("Captured images reset successfully!")



# Streamlit app
def main():
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")

    # Initialize valid_detections in session state if it doesn't exist
    if "valid_detections" not in st.session_state:
        st.session_state.valid_detections = []
    
    # LLM Call
    situation = st.chat_input("Describe what you are going to be doing for the day...")
    if situation:
        st.toast("Response has been recorded ✅")  # Subtle confirmation
        st.markdown(f"**Event:** {situation}")  # Show user's input clearly
        
        # Update valid_detections only when new input is received
        #st.session_state.valid_detections = get_initial_agent(situation)
        st.session_state.valid_detections = str(get_initial_agent(situation))
        # st.write(f"{st.session_state.valid_detections}")
        with open("story_agent_prompt.txt", "a") as file:  # Append instead of overwriting
            file.write(situation + "\n")
    else:
        st.toast("⚠️ Please enter a response.")  # More noticeable warning

    st.sidebar.title("User Configuration")
    source = st.sidebar.selectbox("Video Source", ("webcam", "video"))
    enable_trk = st.sidebar.radio("Enable Tracking", ("Yes", "No")) == "Yes"
    conf = 0.75
    iou = 0.65

    # Reset Captured Images Button
    if st.sidebar.button("Reset Captured Images"):
        reset_captured_images()

    # Load the selected model
    with st.spinner("Loading model..."):
        model = load_model("yolo11n.pt")
        class_names = list(model.names.values())

        # Get actual class names (use session_state valid_detections)
        validated_class_names = []  # Initialize an empty list

        for class_name in class_names:
            if class_name in st.session_state.valid_detections:
                validated_class_names.append(class_name)  # Add valid class names to the list

        with open("validated_class_names.txt", "w") as file:
            file.write("\n".join(validated_class_names))
    st.success("Model loaded successfully!")

    selected_ind = list(range(len(class_names)))  # Select all classes initially

    vid_file_name = ""
    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())
            with open("ultralytics.mp4", "wb") as out:
                out.write(g.read())
            vid_file_name = "ultralytics.mp4"
    elif source == "webcam":
        webcam_id = st.sidebar.number_input("Select Webcam ID", min_value=0, max_value=5, value=0, step=1)
        vid_file_name = int(webcam_id)

    if st.sidebar.button("Start"):
        stop_button = st.button("Stop")
        cap = cv2.VideoCapture(vid_file_name)
        if not cap.isOpened():
            st.error("Could not open video source.")
        else:
            col1, col2 = st.columns(2)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("Failed to read frame from video source.")
                    break

                if enable_trk:
                    results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
                else:
                    results = model(frame, conf=conf, iou=iou, classes=selected_ind)
                
                annotated_frame = results[0].plot()

                # Check for high-confidence detections and save detected class names
                save_image(annotated_frame, results[0].boxes, validated_class_names)

                if stop_button:
                    cap.release()
                    st.stop()

                col1.image(frame, channels="BGR", caption="Original Frame")
                col2.image(annotated_frame, channels="BGR", caption="Annotated Frame")

            cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
