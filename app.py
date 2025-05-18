import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import euclidean_distances

model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

df = pd.read_csv("Top Indian Places to Visit.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

for col in ['state', 'significance', 'type']:
    df[col] = label_encoders[col].transform(df[col].astype(str))

df['popularity'] = (df['google_review_rating'] >= 3.5).astype(int)

st.sidebar.header("🌍 Explore India: Your Personalized Travel Guide")
state_option = st.sidebar.selectbox("Select State", label_encoders['state'].classes_)
significance_option = st.sidebar.selectbox("Select Significance", label_encoders['significance'].classes_)
type_option = st.sidebar.selectbox("Select Type", label_encoders['type'].classes_)
entrance_fee = st.sidebar.slider("Maximum Entrance Fee (₹)", 0, 5000, 50)
visit_time = st.sidebar.slider("Maximum Time to Visit (Hours)", 0, 10, 1)

if st.sidebar.button("Submit"):

    if visit_time == 0:
        st.warning("⛔ Visit time must be greater than 0 to suggest places.")
        st.stop()

    input_vector = {
        'state': label_encoders['state'].transform([state_option])[0],
        'significance': label_encoders['significance'].transform([significance_option])[0],
        'type': label_encoders['type'].transform([type_option])[0],
        'entrance_fee_in_inr': entrance_fee,
        'time_needed_to_visit_in_hrs': visit_time
    }

    input_df = pd.DataFrame([input_vector])
    scaled_input = scaler.transform(input_df)

    predicted_popularity = model.predict(scaled_input)[0]

    matching_places = df[
        (df['state'] == input_vector['state']) & 
        (df['significance'] == input_vector['significance']) & (df['type'] == input_vector['type'])

    ]

    if matching_places.empty:
        st.info("ℹ️ No places match your filters. Try adjusting your preferences.")
    else:
        feature_cols = ['state', 'significance', 'type', 'entrance_fee_in_inr', 'time_needed_to_visit_in_hrs']
        distances = euclidean_distances(scaled_input, scaler.transform(matching_places[feature_cols]))[0]
        matching_places = matching_places.copy()
        matching_places['distance'] = distances

        if predicted_popularity == 1:
            filtered_df = matching_places[matching_places['popularity'] == 1].sort_values(by='distance').head(3)
            title = "🌟 Top Similar Popular Places for You"
        else:
            filtered_df = matching_places.sort_values(by='distance').head(3)
            title = "🔍 Closest Matching Places "

        st.title("🌏 Tourist Visit Recommendation System")
        st.subheader(title)

        for _, place in filtered_df.iterrows():
            st.markdown(f"""
            ### 🏞️ {place['name']}
            - 📍 **State**: {label_encoders['state'].inverse_transform([place['state']])[0]}
            - ⭐ **Significance**: {label_encoders['significance'].inverse_transform([place['significance']])[0]}
            - 🏷️ **Type**: {label_encoders['type'].inverse_transform([place['type']])[0]}
            - 💸 **Entrance Fee**: ₹{place['entrance_fee_in_inr']}
            - ⏱️ **Time Needed**: {place['time_needed_to_visit_in_hrs']} hrs
            - 🌟 **Google Review Rating**: {place['google_review_rating']}
            - 🛬 **Airport within 50km**: {place['airport_with_50km_radius']}
            - 📷 **DSLR Allowed**: {place['dslr_allowed']}
            - 📅 **Best Time to Visit**: {place['best_time_to_visit']}
            """)
