🌍 Tourist Place Recommendation System

This project is a machine learning-based web application designed to help users discover the best tourist destinations across India based on their preferences. It leverages classification models and similarity measures to suggest personalized travel options, making trip planning smarter and more efficient.

🔍 Project Overview

The system uses a clean and structured dataset of Indian tourist destinations, enriched with attributes like state, significance (cultural, historical, natural, etc.), type (fort, temple, beach, etc.), entrance fee, and time required to visit. Using user-provided preferences, the model predicts whether a place is popular and recommends the top 3 closest matching locations using a distance-based similarity metric.

⚙️ Key Features

* Predicts the popularity of tourist places based on user preferences.
* Recommends top 3 matching destinations using Euclidean distance.
* Interactive web interface built with Streamlit.
* Uses machine learning models including K-Nearest Neighbors, Decision Tree, and Naive Bayes.
* Handles imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique).
* Saves and loads trained models and scalers using Joblib.

🛠️ Technologies Used

* Python 3.9
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* imbalanced-learn (SMOTE)
* Streamlit (for web UI)
* Joblib (for model persistence)
  
📁 Project Structure

 app.py                          -   Streamlit application
 train.py                        -   Model training and evaluation script
 Top Indian Places to Visit.csv  -   Dataset
 *.pkl                           -   Saved models and encoders
 README.md                       -   Project documentation


🚀 How to Run

1. Clone this repository.
2. Install required packages: `pip install -r requirements.txt`
3. Run the training script: `python train.py`
4. Start the web app: `streamlit run app.py`
