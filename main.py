import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ==========================================
# 🏥 CONFIGURATION & SAFETY THRESHOLDS
# ==========================================
# Exact values from your successful run
METADATA = {
    "thresholds": {
        "Covid": 0.1241,
        "Normal": 0.1651,
        "Pneumonia": 0.2921,
        "Tuberculosis": 0.1766,
        "Tumor": 0.1009
    },
    # Class names mapped to ID (0-4)
    "class_names": {
        0: "Covid",
        1: "Normal",
        2: "Pneumonia",
        3: "Tuberculosis",
        4: "Tumor"
    }
}

MODEL_PATH = 'final_lung_ai_v1.h5'

# ==========================================
# 🧠 AI ENGINE
# ==========================================
class LungDoctor:
    def __init__(self):
        print("Loading AI Brain... (DenseNet121)")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file '{MODEL_PATH}' not found!")
        
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ AI System Online.")

    def preprocess_image(self, img_path):
        """Standardizes the image to match what the model learned."""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalization
        return img_array

    def diagnose(self, img_path):
        """Runs the diagnosis using the Safety Thresholds."""
        if not os.path.exists(img_path):
            print(f"❌ Error: File '{img_path}' not found.")
            return

        print(f"\n🔎 Analyzing Patient: {os.path.basename(img_path)}...")
        
        # 1. Get Probability Scores
        processed_img = self.preprocess_image(img_path)
        predictions = self.model.predict(processed_img, verbose=0)[0]

        # 2. Apply Safety Rules
        findings = []
        print("\n--- DETAILED REPORT ---")
        
        for idx, prob in enumerate(predictions):
            name = METADATA["class_names"][idx]
            threshold = METADATA["thresholds"][name]
            
            is_detected = prob > threshold
            status = "⚠️ DETECTED" if is_detected else "Clear"
            
            if is_detected:
                findings.append(name)
            
            # Print row
            print(f"{name:<15}: {prob:.1%} (Cutoff: {threshold:.1%}) -> {status}")

        # 3. Final Conclusion
        print("-" * 30)
        if "Normal" in findings and len(findings) == 1:
            print("✅ FINAL DIAGNOSIS: Healthy Lungs")
        elif findings:
            real_diseases = [f for f in findings if f != "Normal"]
            if real_diseases:
                print(f"🚨 FINAL DIAGNOSIS: Signs of {', '.join(real_diseases)}")
            else:
                print("✅ FINAL DIAGNOSIS: Healthy Lungs")
        else:
            print("❓ FINAL DIAGNOSIS: Inconclusive (Low confidence)")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Initialize
    try:
        doctor = LungDoctor()
        
        # 2. Simple Loop to Test Multiple Files
        while True:
            path = input("\nDrag & Drop X-ray file here (or type 'exit'): ").strip().replace('"', '').replace("'", "")
            if path.lower() in ['exit', 'quit']:
                break
            
            if path:
                doctor.diagnose(path)
            
    except Exception as e:
        print(f"\n❌ Critical Error: {e}")