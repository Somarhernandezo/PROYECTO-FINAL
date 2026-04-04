from pickle import load
import streamlit as st
import numpy as np

# Cargar modelo, scaler y selector
model = load(open("../models/best_model.sav", "rb"))
scaler = load(open("../models/scaler.sav", "rb"))
selector = load(open("../models/selector.sav", "rb"))

class_dict = {
    0: "No abandonará el curso",
    1: "Abandonará el curso"
}

st.title("Predicción de Abandono Estudiantil")
st.write("Ingresa las características del estudiante para predecir si abandonará el curso.")

# Sliders para cada feature
age = st.slider("Edad", min_value=18, max_value=45, value=25, step=1)
gender = st.selectbox("Género", ["Male", "Female"])
country = st.selectbox("País", ["USA", "India", "UK", "Canada", "Australia", "Germany", "Brazil", "Japan", "Mexico", "France"])
device_type = st.selectbox("Dispositivo", ["Laptop", "Tablet", "Smartphone"])
internet_speed = st.slider("Velocidad de internet (Mbps)", min_value=5.0, max_value=100.0, value=50.0, step=1.0)
study_hours = st.slider("Horas de estudio semanales", min_value=1.0, max_value=40.0, value=15.0, step=0.5)
login_freq = st.slider("Frecuencia de login semanal", min_value=1, max_value=7, value=4, step=1)
session_duration = st.slider("Duración promedio de sesión (min)", min_value=10.0, max_value=120.0, value=45.0, step=1.0)
video_time = st.slider("Tiempo de video (min)", min_value=0.0, max_value=300.0, value=60.0, step=5.0)
assignments = st.slider("Tareas entregadas", min_value=0, max_value=20, value=8, step=1)
forum_posts = st.slider("Posts en foro", min_value=0, max_value=20, value=3, step=1)
quiz_attempts = st.slider("Intentos de quiz", min_value=0, max_value=15, value=5, step=1)
quiz_score = st.slider("Promedio de quiz", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
attendance = st.slider("Tasa de asistencia", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
engagement = st.slider("Score de engagement", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
final_grade = st.slider("Nota final", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

if st.button("Predecir"):
    # Codificar categóricas con factorize (mismo orden que en entrenamiento)
    gender_map = {"Female": 0, "Male": 1}
    device_map = {"Laptop": 0, "Tablet": 1, "Smartphone": 2}
    country_map = {"USA": 0, "India": 1, "UK": 2, "Canada": 3, "Australia": 4,
                   "Germany": 5, "Brazil": 6, "Japan": 7, "Mexico": 8, "France": 9}

    input_data = np.array([[age, gender_map[gender], country_map[country],
                            device_map[device_type], internet_speed, study_hours,
                            login_freq, session_duration, video_time, assignments,
                            forum_posts, quiz_attempts, quiz_score, attendance,
                            engagement, final_grade]])

    # Escalar y seleccionar features
    input_scaled = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)

    # Predecir
    prediction = model.predict(input_selected)[0]
    pred_class = class_dict[prediction]
    st.write("Predicción:", pred_class)
