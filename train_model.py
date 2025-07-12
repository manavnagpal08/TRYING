import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer # For text data
import matplotlib.pyplot as plt
import seaborn as sns
import random # This is used for the *original* in-memory data generation, but will be removed if loading from file
import joblib # For saving/loading the model and vectorizer

# --- 1. Data Loading (for Resume Screening) ---
# Replace this section with your actual resume screening data loading.
# Your data will likely be in a CSV, JSON, or database format.
# Assume your data has at least two columns: one for the resume text and one for the target label.
print("1. Loading resume screening dataset...")

# --- LOAD DATA FROM GENERATED FILES ---
# Choose one of the following lines to load from the CSV or JSON file you generated.
# Uncomment the one you want to use and comment out the other.

# For CSV:
df = pd.read_csv('dummy_resume_data.csv')
# For JSON:
# df = pd.read_json('dummy_resume_data.json')


# Separate features (X) and target variable (y)
# 'resume_text' will be our feature (X) and 'job_category' will be our target (y)
X_raw = df['resume_text']
y = df['job_category']

print(f"Dataset loaded. Number of resumes: {len(X_raw)}")
print("First 5 resume texts:\n", X_raw[:5].tolist())
print("First 5 job categories:\n", y[:5].tolist())
print(f"\nValue counts for job categories:\n{y.value_counts()}")


# --- 2. Text Preprocessing (Vectorization) ---
# Machine learning models cannot directly process raw text.
# We need to convert text into numerical features using a technique like TF-IDF.
print("\n2. Performing text vectorization using TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=7000, # Increased max_features for a larger dataset and richer vocabulary
    stop_words='english', # Remove common English stop words (e.g., 'the', 'is')
    ngram_range=(1, 2) # Consider single words and two-word phrases
)
X = vectorizer.fit_transform(X_raw)
print(f"Text data vectorized. New feature matrix shape: {X.shape}")

# --- 3. Data Splitting ---
# Split the data into training and testing sets.
# The test set will be used for final evaluation and should not be touched during training/tuning.
print("\n3. Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Testing data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 4. Model Selection and Hyperparameter Tuning (GridSearchCV) ---
# We'll use a RandomForestClassifier as an example.
# GridSearchCV helps find the best hyperparameters by trying all combinations
# within a defined grid and using cross-validation.
print("\n4. Initializing RandomForestClassifier and defining hyperparameter grid...")
model = RandomForestClassifier(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],      # Maximum depth of the tree (None means unlimited)
    'min_samples_split': [2, 5],      # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2]        # Minimum number of samples required to be at a leaf node
}

print("Starting GridSearchCV for hyperparameter tuning (this may take some time)...")
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,                  # Keeping CV at 5 for robustness
    scoring='accuracy',    # Metric to optimize
    n_jobs=-1,             # Use all available CPU cores
    verbose=1              # Show progress
)

grid_search.fit(X_train, y_train)

print("\nGridSearchCV completed.")
print(f"Best hyperparameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# --- 5. Cross-Validation (on the best model) ---
# Evaluate the best model's performance using cross-validation on the training data.
# This gives a more robust estimate of performance than a single train/validation split.
print("\n5. Performing cross-validation on the best model...")
cv_scores = cross_val_score(best_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1) # Keeping CV at 10
print(f"Cross-validation accuracies: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# --- 6. Final Model Evaluation on Test Set ---
# Evaluate the best model on the unseen test set to get a final, unbiased performance estimate.
print("\n6. Evaluating the best model on the unseen test set...")
y_pred = best_model.predict(X_test)

final_accuracy = accuracy_score(y_test, y_pred)
print(f"Final Test Set Accuracy: {final_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- 7. Visualization of Confusion Matrix ---
# Note: For multi-class classification, the confusion matrix will be larger.
# The labels will correspond to the unique categories in your 'job_category' column.
plt.figure(figsize=(16, 14)) # Adjusted figure size for more categories and better readability
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y), # Use actual class labels for x-axis
            yticklabels=np.unique(y)) # Use actual class labels for y-axis
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Resume Screening')
plt.show()

# --- 8. Save the Trained Model and Vectorizer ---
# It's crucial to save your trained model and the TF-IDF vectorizer
# so you can use them later to make predictions on new, unseen resumes
# without retraining the model or re-fitting the vectorizer.
model_filename = 'resume_screening_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

joblib.dump(best_model, model_filename)
joblib.dump(vectorizer, vectorizer_filename)

print(f"\nModel saved to {model_filename}")
print(f"TF-IDF Vectorizer saved to {vectorizer_filename}")

print("\nModel training and evaluation complete!")
