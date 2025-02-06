import json
import streamlit as st
import os
from pydantic_ai.messages import ModelMessage,ModelRequest,ModelResponse,TextPart,UserPromptPart
from agent import get_story_agent
JSON_FILE = "captured_images.json"

# Load captured images
def load_captured_images():
    try:
        with open(JSON_FILE, "r") as file:
            images = json.load(file)
        return images  # Dictionary format: { class_id: {"path": "path/to/image", "conf": 0.95} }
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

message_history = list[ModelMessage]

def get_history():
        return message_history


# Streamlit app
def main():
    st.title("Captured Images Summary")
    with open("validated_class_names.txt", "r") as file:
        validated_class_names = file.read()
        message_history.append(ModelRequest(parts=[UserPromptPart(content=validated_class_names)]))
    summary = get_story_agent(validated_class_names) 
    message_history.append(ModelResponse(parts=[TextPart(content=summary)]))   

    
    
    
    # Load images dynamically
    captured_images = load_captured_images()

    if not captured_images:
        st.warning("No high-confidence images captured yet.")
        return

    # Initialize session state
    if "selected_image" not in st.session_state:
        st.session_state["selected_image"] = None

    # Display images with class labels
    cols = st.columns(min(len(captured_images), 5))  # Adjust columns dynamically
    for i, (class_id, data) in enumerate(captured_images.items()):
        image_path = data["path"]
        if os.path.exists(image_path):  # Ensure the file exists
            if cols[i % 5].button(f"Class {class_id}"):
                st.session_state["selected_image"] = image_path

    # Display selected image
    if st.session_state["selected_image"]:
        st.subheader("Selected Image")
        st.image(st.session_state["selected_image"], caption="Captured Image", use_container_width=True)
        st.write(f"{summary}")



if __name__ == "__main__":
    main()
