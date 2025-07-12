import joblib
import pandas as pd # Needed if you were loading a new resume from a DataFrame, but not strictly for a single string
import numpy as np # Often useful with scikit-learn outputs

# --- 1. Load the Saved Model and Vectorizer ---
print("1. Loading the trained model and TF-IDF vectorizer...")
try:
    loaded_model = joblib.load('resume_screening_model.pkl')
    loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer files not found.")
    print("Please ensure 'resume_screening_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory.")
    print("You need to run the training script ('model_training_code' Canvas) first to create these files.")
    exit() # Exit if files are not found

# --- 2. Define a New Resume for Prediction ---
# This is where you would input a new resume text that your app receives.
new_resume_text = """
Experienced Software Engineer with 8 years of hands-on experience in developing scalable web applications using Python, Django, and React. Proficient in cloud deployment on AWS, including Docker and Kubernetes. Strong understanding of algorithms and data structures. Contributed to open-source projects and passionate about clean code and continuous integration. Excellent problem-solver and team player.
"""

print("\n2. New Resume for Prediction:")
print(new_resume_text)

# --- 3. Preprocess the New Resume ---
# It's CRUCIAL to use the SAME vectorizer that was fitted on the training data.
print("\n3. Preprocessing the new resume text...")
new_resume_vectorized = loaded_vectorizer.transform([new_resume_text])
print(f"New resume vectorized. Feature shape: {new_resume_vectorized.shape}")

# --- 4. Make a Prediction ---
print("\n4. Making a prediction using the loaded model...")
predicted_category = loaded_model.predict(new_resume_vectorized)

print(f"\nPredicted Job Category: {predicted_category[0]}")

# --- Optional: Get Prediction Probabilities (for confidence) ---
# This can be useful if you want to see how confident the model is about its prediction.
# For a perfect dummy dataset, probabilities will likely be 1.0 for the predicted class.
if hasattr(loaded_model, 'predict_proba'):
    prediction_probabilities = loaded_model.predict_proba(new_resume_vectorized)
    # Get the class labels from the loaded model
    class_labels = loaded_model.classes_
    
    print("\nPrediction Probabilities:")
    for i, prob in enumerate(prediction_probabilities[0]):
        print(f"  {class_labels[i]}: {prob:.4f}")
    
    # You can also sort them for better readability
    sorted_probs = sorted(zip(class_labels, prediction_probabilities[0]), key=lambda x: x[1], reverse=True)
    print("\nSorted Prediction Probabilities:")
    for label, prob in sorted_probs:
        print(f"  {label}: {prob:.4f}")

print("\nPrediction process complete!")
