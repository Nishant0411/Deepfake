print("Script started")
import os
from transformers import pipeline
from PIL import Image

def analyze_temporal_sequence(folder_path):
    print("Summoning the AI Model from Hugging Face...")
    print("(This might take a minute or two to download the first time you run it!)")
    
    # Load the pre-trained deepfake classification pipeline
    ai_scanner = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")

    # Grab the first 5 frames from our sequence
    frames = [
f for f in sorted(os.listdir(folder_path))
if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

    print("\n--- DeepGuard AI Analysis Results ---")
    
    for frame_name in frames:
        image_path = os.path.join(folder_path, frame_name)
        img = Image.open(image_path)
        results = ai_scanner(img)
        
        top_prediction = results[0]
        label = top_prediction['label']
        confidence = round(top_prediction['score'] * 100, 2)
        
        print(f"Frame {frame_name}: {label} ({confidence}% confident)")
print("About to call function...")
analyze_temporal_sequence("extracted_frames")
print("Function finished")
    