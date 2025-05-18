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
  
Certainly! Here's your **📁 Project Structure** section formatted point-wise, as requested:

---

📁 Project Structure

1. `app.py` – Streamlit-based web application for user interaction and recommendations.
2. `train.py` – Script for training and evaluating machine learning models.
3. `Top Indian Places to Visit.csv` – Dataset containing details of tourist destinations across India.
4. `.pkl` – Serialized files for saved models, scalers, and label encoders using Joblib.
5. `README.md` – Project documentation and usage instructions.

---

Let me know if you'd like to include a `requirements.txt` or any other file as well.


🚀 How to Run

1. Clone this repository.
2. Install required packages: `pip install -r requirements.txt`
3. Run the training script: `python train.py`
4. Start the web app: `streamlit run app.py`
