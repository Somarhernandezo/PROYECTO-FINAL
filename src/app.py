from pickle import load
import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

# Rutas relativas a la raíz del proyecto (funciona tanto local como en Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------- Config ----------
st.set_page_config(page_title="EduTrack", page_icon="📘", layout="wide")

# Paleta educativa
AZUL = "#1F4E79"
VERDE = "#2E8B57"
AMARILLO = "#E1A140"
ROJO = "#C0392B"
GRIS = "#F2F4F7"

st.markdown(f"""
    <style>
    .main {{ background-color: {GRIS}; }}
    .titulo-app {{
        color: {AZUL}; font-size: 42px; font-weight: 800; margin-bottom: 0px;
    }}
    .subtitulo-app {{
        color: #555; font-size: 17px; margin-top: 0px; margin-bottom: 25px;
    }}
    .tarjeta {{
        background-color: white; padding: 20px; border-radius: 12px;
        border-left: 6px solid {AZUL}; box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .tarjeta h4 {{ margin-top: 0; color: {AZUL}; }}
    </style>
""", unsafe_allow_html=True)

# ---------- Cargar modelo y datos ----------
model = load(open(os.path.join(BASE_DIR, "models/best_model.sav"), "rb"))
scaler = load(open(os.path.join(BASE_DIR, "models/scaler.sav"), "rb"))
selector = load(open(os.path.join(BASE_DIR, "models/selector.sav"), "rb"))

class_dict = {0: "Continuará el curso", 1: "Riesgo de abandono"}

@st.cache_data
def cargar_dataset():
    return pd.read_csv(os.path.join(BASE_DIR, "online_learning_engagement_dataset.csv"))

df_full = cargar_dataset()

# Mapeos de variables categóricas (mismos que en entrenamiento)
GENDER_MAP = {"Female": 0, "Male": 1}
DEVICE_MAP = {"Laptop": 0, "Tablet": 1, "Smartphone": 2}
COUNTRY_MAP = {"USA": 0, "India": 1, "UK": 2, "Canada": 3, "Australia": 4,
               "Germany": 5, "Brazil": 6, "Japan": 7, "Mexico": 8, "France": 9}

def preparar_features(df):
    """Convierte un dataframe con las columnas originales en la matriz X que espera el modelo."""
    X = pd.DataFrame()
    X["age"] = df["age"]
    X["gender"] = df["gender"].map(GENDER_MAP).fillna(0)
    X["country"] = df["country"].map(COUNTRY_MAP).fillna(0)
    X["device_type"] = df["device_type"].map(DEVICE_MAP).fillna(0)
    X["internet_speed_mbps"] = df["internet_speed_mbps"]
    X["study_hours_weekly"] = df["study_hours_weekly"]
    X["login_frequency_weekly"] = df["login_frequency_weekly"]
    X["avg_session_duration_min"] = df["avg_session_duration_min"]
    X["video_watch_time_min"] = df["video_watch_time_min"]
    X["assignments_submitted"] = df["assignments_submitted"]
    X["forum_posts"] = df["forum_posts"]
    X["quiz_attempts"] = df["quiz_attempts"]
    X["avg_quiz_score"] = df["avg_quiz_score"]
    X["attendance_rate"] = df["attendance_rate"]
    X["engagement_score"] = df["engagement_score"]
    X["final_grade"] = df["final_grade"]
    return X.values

@st.cache_data
def metricas_modelo():
    """Calcula precisión, recall y accuracy del modelo sobre el dataset completo."""
    X = preparar_features(df_full)
    X_s = scaler.transform(X)
    X_sel = selector.transform(X_s)
    y_true = df_full["dropout"].values
    y_pred = model.predict(X_sel)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    return accuracy, precision, recall

@st.cache_data
def predecir_batch(df):
    """Predice riesgo para todo un dataframe y devuelve el df con probabilidades ordenadas."""
    X = preparar_features(df)
    X_s = scaler.transform(X)
    X_sel = selector.transform(X_s)
    try:
        probs = model.predict_proba(X_sel)[:, 1]
    except Exception:
        probs = model.predict(X_sel).astype(float)
    out = df.copy()
    out["probabilidad_abandono"] = (probs * 100).round(1)
    out["nivel_riesgo"] = pd.cut(
        probs, bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["🟢 Bajo", "🟡 Medio", "🔴 Alto"]
    )
    return out.sort_values("probabilidad_abandono", ascending=False)

# ---------- Funciones ----------
def gauge(prob):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')

    if prob >= 0.66: color = ROJO
    elif prob >= 0.33: color = AMARILLO
    else: color = VERDE

    ax.add_patch(Wedge((0, 0), 1.0, 0, 360, width=0.22, facecolor="#E8ECF1", edgecolor='none'))
    angulo_fin = 90 - (prob * 360)
    ax.add_patch(Wedge((0, 0), 1.0, angulo_fin, 90, width=0.22, facecolor=color, edgecolor='none'))
    ax.add_patch(Circle((0, 0), 0.78, facecolor='white', edgecolor='none'))

    ax.text(0, 0.08, f"{int(prob*100)}%", ha='center', va='center',
            fontsize=44, fontweight='bold', color=color)
    ax.text(0, -0.22, "riesgo de abandono", ha='center', va='center', fontsize=11, color="#666")

    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal'); ax.axis('off')
    return fig


def grafica_variable(columna, valor_alumno, etiqueta, umbral, modo, unidad=""):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor('white')

    datos = df_full[columna].dropna()
    n, bins, patches = ax.hist(datos, bins=30, color="#CFD8E3", edgecolor='white')

    for patch, left in zip(patches, bins[:-1]):
        if modo == "menor" and left < umbral:
            patch.set_facecolor(ROJO); patch.set_alpha(0.55)
        elif modo == "mayor" and left >= umbral:
            patch.set_facecolor(ROJO); patch.set_alpha(0.55)

    ax.axvline(umbral, color=AMARILLO, linestyle='--', linewidth=2, label="Umbral de riesgo")

    color_alumno = ROJO if (
        (modo == "menor" and valor_alumno < umbral) or
        (modo == "mayor" and valor_alumno > umbral)
    ) else VERDE
    ax.axvline(valor_alumno, color=color_alumno, linewidth=3, label="Este alumno")

    ax.set_title(etiqueta, fontsize=12, color=AZUL, fontweight='bold', loc='left')
    ax.set_xlabel(unidad, fontsize=9, color="#666")
    ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    ax.tick_params(colors='#666', labelsize=9)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


def grafica_comparativa(columna, etiqueta, unidad=""):
    """Compara la distribución de alumnos que desertan vs los que no."""
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    fig.patch.set_facecolor('white')

    no_des = df_full[df_full["dropout"] == 0][columna].dropna()
    si_des = df_full[df_full["dropout"] == 1][columna].dropna()

    bins = np.linspace(min(no_des.min(), si_des.min()), max(no_des.max(), si_des.max()), 25)
    ax.hist(no_des, bins=bins, color=VERDE, alpha=0.6, label="No desertan", edgecolor='white')
    ax.hist(si_des, bins=bins, color=ROJO, alpha=0.6, label="Sí desertan", edgecolor='white')

    ax.axvline(no_des.mean(), color=VERDE, linestyle='--', linewidth=2)
    ax.axvline(si_des.mean(), color=ROJO, linestyle='--', linewidth=2)

    ax.set_title(etiqueta, fontsize=12, color=AZUL, fontweight='bold', loc='left')
    ax.set_xlabel(unidad, fontsize=9, color="#666")
    ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    ax.tick_params(colors='#666', labelsize=9)
    ax.legend(loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


def factores_riesgo(d):
    factores = []
    if d["attendance"] < 0.5:
        factores.append(("Baja asistencia", f"{int(d['attendance']*100)}% de clases asistidas"))
    if d["study_hours"] < 8:
        factores.append(("Pocas horas de estudio", f"{d['study_hours']} hrs/semana"))
    if d["login_freq"] <= 2:
        factores.append(("Poca frecuencia de acceso", f"{d['login_freq']} logins/semana"))
    if d["quiz_score"] < 60:
        factores.append(("Bajo desempeño en quizzes", f"promedio {int(d['quiz_score'])}/100"))
    if d["assignments"] < 5:
        factores.append(("Pocas tareas entregadas", f"{d['assignments']} de 20"))
    if d["engagement"] < 4:
        factores.append(("Engagement bajo", f"{d['engagement']}/10"))
    if d["video_time"] < 40:
        factores.append(("Poco contenido visto", f"{int(d['video_time'])} min de video"))
    if d["forum_posts"] == 0:
        factores.append(("Sin participación en foros", "0 posts"))
    return factores[:3]


def recomendaciones(prob, d):
    recs = []
    if d["attendance"] < 0.5:
        recs.append(("📅", "Enviar recordatorios automáticos antes de cada clase y notificar a un tutor si falta más de 2 sesiones seguidas."))
    if d["study_hours"] < 8:
        recs.append(("⏰", "Sugerir bloques fijos de estudio en su calendario (mínimo 2 hrs, 3 veces por semana)."))
    if d["login_freq"] <= 2:
        recs.append(("🔔", "Activar notificaciones push diarias y un email semanal de progreso."))
    if d["quiz_score"] < 60:
        recs.append(("📚", "Asignar material de refuerzo en los temas con peor desempeño y ofrecer una sesión 1 a 1 con un tutor."))
    if d["assignments"] < 5:
        recs.append(("📝", "Dividir las tareas pendientes en mini-entregas semanales con feedback rápido."))
    if d["forum_posts"] == 0:
        recs.append(("💬", "Invitarlo a un grupo de estudio en Discord/Slack y darle un mentor par."))
    if d["engagement"] < 4:
        recs.append(("🎯", "Aplicar gamificación: badges, retos cortos y un ranking de su cohorte."))
    if d["video_time"] < 40:
        recs.append(("🎬", "Enviar resúmenes en video de 5 min de las clases que no ha visto."))

    if not recs:
        if prob >= 0.66:
            recs = [
                ("📞", "Contactar al estudiante esta misma semana con una llamada personalizada."),
                ("👨‍🏫", "Asignar un tutor académico de seguimiento."),
                ("📈", "Ofrecer un plan de recuperación con metas semanales."),
            ]
        elif prob >= 0.33:
            recs = [
                ("👀", "Monitorear semanalmente sus métricas de engagement."),
                ("👥", "Invitarlo a sesiones grupales de estudio."),
            ]
        else:
            recs = [
                ("🏆", "Mantener la motivación con reconocimientos y badges."),
                ("🚀", "Ofrecerle contenido avanzado o retos opcionales."),
                ("🤝", "Invitarlo a participar como mentor de otros alumnos."),
            ]

    if prob >= 0.66 and len(recs) > 0:
        recs.insert(0, ("🚨", "ACCIÓN URGENTE: Contacto humano directo en las próximas 48 horas."))

    return recs[:5]


def valores_demo(tipo):
    if tipo == "riesgo":
        return dict(age=22, study_hours=3, login_freq=1, session_duration=15,
                    video_time=20, assignments=2, forum_posts=0, quiz_attempts=1,
                    quiz_score=40, attendance=0.3, engagement=2.0, final_grade=45)
    if tipo == "estrella":
        return dict(age=24, study_hours=25, login_freq=6, session_duration=80,
                    video_time=200, assignments=18, forum_posts=12, quiz_attempts=10,
                    quiz_score=92, attendance=0.95, engagement=8.5, final_grade=90)
    if tipo == "intermedio":
        return dict(age=26, study_hours=12, login_freq=4, session_duration=45,
                    video_time=80, assignments=9, forum_posts=4, quiz_attempts=5,
                    quiz_score=68, attendance=0.65, engagement=5.5, final_grade=68)
    return dict(age=25, study_hours=15, login_freq=4, session_duration=45,
                video_time=60, assignments=8, forum_posts=3, quiz_attempts=5,
                quiz_score=65, attendance=0.7, engagement=5.0, final_grade=65)

# ---------- Header ----------
col_logo, col_titulo = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:70px; text-align:center;'>📘</div>", unsafe_allow_html=True)
with col_titulo:
    st.markdown("<p class='titulo-app'>EduTrack</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitulo-app'>Detectamos a tiempo a los estudiantes en riesgo de abandonar sus cursos en línea.</p>", unsafe_allow_html=True)

# ---------- Métricas del modelo ----------
if False:
    acc, prec, rec = metricas_modelo()
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{AZUL};'>
                <div style='color:#666; font-size:12px;'>Precisión del modelo</div>
                <div style='color:{AZUL}; font-size:26px; font-weight:bold;'>{acc*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{VERDE};'>
                <div style='color:#666; font-size:12px;'>Precision</div>
                <div style='color:{VERDE}; font-size:26px; font-weight:bold;'>{prec*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with mc3:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{AMARILLO};'>
                <div style='color:#666; font-size:12px;'>Recall</div>
                <div style='color:{AMARILLO}; font-size:26px; font-weight:bold;'>{rec*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with mc4:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{ROJO};'>
                <div style='color:#666; font-size:12px;'>Alumnos en entrenamiento</div>
                <div style='color:{ROJO}; font-size:26px; font-weight:bold;'>{len(df_full):,}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Sidebar ----------
if "demo" not in st.session_state:
    st.session_state.demo = None

d = valores_demo(st.session_state.demo)

st.sidebar.header("Datos del estudiante")
st.sidebar.caption("Ajusta los valores o usa un caso de ejemplo desde la pestaña de análisis.")

with st.sidebar.expander("Información personal", expanded=True):
    age = st.slider("Edad", 18, 45, d["age"])
    gender = st.selectbox("Género", ["Male", "Female"])
    country = st.selectbox("País", ["USA", "India", "UK", "Canada", "Australia",
                                     "Germany", "Brazil", "Japan", "Mexico", "France"])
    device_type = st.selectbox("Dispositivo", ["Laptop", "Tablet", "Smartphone"])
    internet_speed = st.slider("Velocidad de internet (Mbps)", 5.0, 100.0, 50.0)

with st.sidebar.expander("Actividad académica", expanded=True):
    study_hours = st.slider("Horas de estudio semanales", 1.0, 40.0, float(d["study_hours"]))
    login_freq = st.slider("Logins por semana", 1, 7, d["login_freq"])
    session_duration = st.slider("Duración promedio de sesión (min)", 10.0, 120.0, float(d["session_duration"]))
    video_time = st.slider("Minutos de video vistos", 0.0, 300.0, float(d["video_time"]))
    attendance = st.slider("Tasa de asistencia", 0.0, 1.0, float(d["attendance"]))

with st.sidebar.expander("Desempeño", expanded=True):
    assignments = st.slider("Tareas entregadas", 0, 20, d["assignments"])
    forum_posts = st.slider("Posts en foro", 0, 20, d["forum_posts"])
    quiz_attempts = st.slider("Intentos de quiz", 0, 15, d["quiz_attempts"])
    quiz_score = st.slider("Promedio de quiz", 0.0, 100.0, float(d["quiz_score"]))
    engagement = st.slider("Score de engagement", 0.0, 10.0, float(d["engagement"]))
    final_grade = st.slider("Nota final", 0.0, 100.0, float(d["final_grade"]))

predecir = st.sidebar.button("🔍 Analizar estudiante", use_container_width=True, type="primary")

# ---------- Contenido principal ----------
if True:
    st.markdown("### Casos de ejemplo")
    st.caption("Carga un perfil rápido para ver cómo funciona EduTrack:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("👨‍🎓 Estudiante en riesgo", use_container_width=True):
            st.session_state.demo = "riesgo"; st.rerun()
    with c2:
        if st.button("⭐ Estudiante estrella", use_container_width=True):
            st.session_state.demo = "estrella"; st.rerun()
    with c3:
        if st.button("🤔 Caso intermedio", use_container_width=True):
            st.session_state.demo = "intermedio"; st.rerun()
    with c4:
        if st.button("🔄 Limpiar", use_container_width=True):
            st.session_state.demo = None; st.rerun()

    if predecir or st.session_state.demo is not None:
        gender_map = {"Female": 0, "Male": 1}
        device_map = {"Laptop": 0, "Tablet": 1, "Smartphone": 2}
        country_map = {"USA": 0, "India": 1, "UK": 2, "Canada": 3, "Australia": 4,
                       "Germany": 5, "Brazil": 6, "Japan": 7, "Mexico": 8, "France": 9}

        input_data = np.array([[age, gender_map[gender], country_map[country],
                                device_map[device_type], internet_speed, study_hours,
                                login_freq, session_duration, video_time, assignments,
                                forum_posts, quiz_attempts, quiz_score, attendance,
                                engagement, final_grade]])

        input_scaled = scaler.transform(input_data)
        input_selected = selector.transform(input_scaled)
        prediction = model.predict(input_selected)[0]

        try:
            prob = model.predict_proba(input_selected)[0][1]
        except Exception:
            prob = float(prediction)

        if prob >= 0.66:
            nivel, color_nivel = "Alto", ROJO
        elif prob >= 0.33:
            nivel, color_nivel = "Medio", AMARILLO
        else:
            nivel, color_nivel = "Bajo", VERDE

        st.markdown("---")
        st.markdown("## Resultado del análisis")

        col_g, col_m = st.columns([1.2, 1])
        with col_g:
            st.pyplot(gauge(prob))
        with col_m:
            st.markdown(f"""
                <div class='tarjeta' style='border-left-color:{color_nivel};'>
                    <h4 style='color:{color_nivel};'>Nivel de riesgo: {nivel}</h4>
                    <p style='font-size:15px; color:#444;'>
                        {class_dict[prediction]}.<br>
                        Probabilidad estimada de abandono: <b>{int(prob*100)}%</b>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### ¿Por qué este resultado?")
        facs = factores_riesgo({
            "attendance": attendance, "study_hours": study_hours, "login_freq": login_freq,
            "quiz_score": quiz_score, "assignments": assignments, "engagement": engagement,
            "video_time": video_time, "forum_posts": forum_posts
        })

        if facs:
            cols = st.columns(len(facs))
            for col, (titulo, detalle) in zip(cols, facs):
                with col:
                    st.markdown(f"""
                        <div class='tarjeta' style='border-left-color:{AMARILLO};'>
                            <h4 style='color:{AMARILLO};'>{titulo}</h4>
                            <p style='color:#555; margin:0;'>{detalle}</p>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("No se detectaron factores críticos. El estudiante muestra buen desempeño general.")

        st.markdown("### Comparación con todos los estudiantes")
        st.caption("La línea de color marca a este alumno. La zona roja es donde aumenta el riesgo.")

        variables = [
            ("attendance_rate", attendance, "Asistencia", 0.5, "menor", "0 a 1"),
            ("study_hours_weekly", study_hours, "Horas de estudio semanales", 8, "menor", "horas"),
            ("avg_quiz_score", quiz_score, "Promedio en quizzes", 60, "menor", "puntos"),
            ("login_frequency_weekly", login_freq, "Logins por semana", 2, "menor", "logins"),
            ("engagement_score", engagement, "Engagement", 4, "menor", "0 a 10"),
            ("video_watch_time_min", video_time, "Minutos de video vistos", 40, "menor", "minutos"),
        ]
        fila1 = st.columns(3); fila2 = st.columns(3)
        columnas = list(fila1) + list(fila2)
        for col, (cname, val, label, thr, modo, unidad) in zip(columnas, variables):
            with col:
                st.pyplot(grafica_variable(cname, val, label, thr, modo, unidad))

        st.markdown("### Plan de acción personalizado")
        st.caption("Recomendaciones específicas basadas en los puntos débiles de este alumno.")
        datos_alumno = {
            "attendance": attendance, "study_hours": study_hours, "login_freq": login_freq,
            "quiz_score": quiz_score, "assignments": assignments, "forum_posts": forum_posts,
            "engagement": engagement, "video_time": video_time
        }
        for icono, texto in recomendaciones(prob, datos_alumno):
            st.markdown(f"""
                <div class='tarjeta' style='border-left-color:{AZUL}; margin-bottom:10px;'>
                    <span style='font-size:22px; margin-right:10px;'>{icono}</span>
                    <span style='color:#333; font-size:15px;'>{texto}</span>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 💰 Calculadora de impacto para tu institución")
        st.markdown("**Ingresa los datos reales de tu academia y descubre cuánto puedes ahorrar con EduTrack.**")
        st.caption("Puedes escribir los valores directamente o usar las flechas. Los resultados se actualizan al instante.")

        tasa_real = float(df_full["dropout"].mean())

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("##### 🏫 Sobre tu institución")
            alumnos_totales = st.number_input(
                "Número de alumnos activos",
                min_value=100, max_value=1_000_000, value=50000, step=500,
                help="¿Cuántos estudiantes activos tiene tu plataforma actualmente?"
            )
            costo_alumno = st.number_input(
                "Ingreso promedio por alumno (USD/año)",
                min_value=50, max_value=20000, value=1200, step=50,
                help="¿Cuánto paga cada alumno al año? Referencias: Platzi ~$240 · Coursera Plus ~$399 · Bootcamps $4,000–$8,000"
            )
        with sc2:
            st.markdown("##### 📉 Sobre tu deserción")
            tasa_desercion_pct = st.number_input(
                "Tasa de deserción anual (%)",
                min_value=1.0, max_value=95.0, value=round(tasa_real * 100, 1), step=0.5,
                help=f"¿Qué % de tus alumnos abandona al año? Si no lo sabes exacto, en el sector típico es 20–35%. Nuestro dataset muestra {tasa_real*100:.1f}%."
            )
            tasa_desercion_actual = tasa_desercion_pct / 100

            # La retención es nuestra PROMESA, no un dato del cliente
            recuperacion = 0.40
            st.markdown(f"""
                <div style='background:white; padding:14px; border-radius:8px; border-left:4px solid {VERDE}; margin-top:10px;'>
                    <div style='color:#666; font-size:13px;'>Nuestra promesa de retención</div>
                    <div style='color:{VERDE}; font-size:28px; font-weight:bold;'>40%</div>
                    <div style='color:#888; font-size:12px;'>de los alumnos en riesgo detectados son retenidos con EduTrack.<br>
                    <i>Basado en estudios de Tinto (1993) y casos reales como Georgia State University (~30%).</i></div>
                </div>
            """, unsafe_allow_html=True)

        perdida_actual = alumnos_totales * tasa_desercion_actual * costo_alumno
        ahorro = perdida_actual * recuperacion

        st.markdown("")
        m1, m2, m3 = st.columns(3)
        m1.metric("Pérdida anual sin EduTrack", f"${perdida_actual:,.0f}")
        m2.metric("Ahorro estimado con EduTrack", f"${ahorro:,.0f}", delta=f"+{int(recuperacion*100)}%")
        m3.metric("Estudiantes retenidos / año", f"{int(alumnos_totales * tasa_desercion_actual * recuperacion):,}")

        # Origen de los números
        with st.expander("📖 ¿De dónde salen estos números?"):
            st.markdown(f"""
**Tamaño de la institución → {alumnos_totales:,} alumnos**
Ajustable. Por defecto usamos 50,000, que es el tamaño de nuestro dataset de entrenamiento.

**Ingreso promedio por alumno → ${costo_alumno:,} USD/año**
Asunción de mercado. Referencias reales:
- Platzi: ~$240/año
- Coursera Plus: ~$399/año
- Udemy Business: ~$360/usuario/año
- Bootcamps (Ironhack, Le Wagon, 4Geeks): $4,000 – $8,000

**Tasa de deserción → {tasa_desercion_actual*100:.1f}%**
Sale **directo del dataset real**: es el promedio de la columna `dropout` en los 50,000 registros.
Coincide con estudios públicos:
- MIT/Harvard (MOOCs en edX): 85–95% de deserción en cursos gratuitos
- Bootcamps pagados: 20–35% de deserción

**% de retención con EduTrack → {int(recuperacion*100)}%**
Asunción conservadora basada en literatura académica:
- Modelo de Tinto (1993): intervenciones tempranas pueden reducir deserción hasta 50%
- Georgia State University implementó un sistema predictivo similar y retuvo al ~30% adicional de alumnos en riesgo
- Programas "Early Alert" en universidades de EE.UU.: 30–45% de recuperación

**Fórmulas:**
```
Pérdida anual  = alumnos × tasa_deserción × ingreso_por_alumno
               = {alumnos_totales:,} × {tasa_desercion_actual*100:.1f}% × ${costo_alumno:,}
               = ${perdida_actual:,.0f}

Ahorro EduTrack = pérdida × % de retención
                = ${perdida_actual:,.0f} × {int(recuperacion*100)}%
                = ${ahorro:,.0f}

Alumnos salvados = alumnos × tasa_deserción × % de retención
                 = {int(alumnos_totales * tasa_desercion_actual * recuperacion):,} estudiantes/año
```
            """)

    else:
        st.info("👈 Selecciona un caso de ejemplo o ajusta los datos en la barra lateral, luego presiona **Analizar estudiante**.")


# ============ TAB 2: PERFIL IDEAL VS EN RIESGO ============
# ---------- Procesamiento masivo ----------
st.markdown("---")
st.markdown("## 📂 Análisis masivo de toda tu base")
st.markdown("Sube el CSV de tus alumnos o usa nuestra base de demostración. EduTrack analiza miles de estudiantes en segundos y te devuelve la lista priorizada.")

bcol1, bcol2 = st.columns([1, 2])
with bcol1:
    usar_demo = st.button("⚡ Procesar base de demostración (50,000 alumnos)", use_container_width=True, type="primary")
with bcol2:
    archivo = st.file_uploader("…o sube tu propio CSV", type=["csv"], label_visibility="collapsed")

df_batch = None
if usar_demo:
    df_batch = df_full
elif archivo is not None:
    try:
        df_batch = pd.read_csv(archivo)
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")

if df_batch is not None:
    try:
        with st.spinner("Analizando estudiantes..."):
            resultado = predecir_batch(df_batch)

        total = len(resultado)
        alto = int((resultado["probabilidad_abandono"] >= 66).sum())
        medio = int(((resultado["probabilidad_abandono"] >= 33) & (resultado["probabilidad_abandono"] < 66)).sum())
        bajo = int((resultado["probabilidad_abandono"] < 33).sum())

        st.success(f"✅ Analizados **{total:,} estudiantes** en segundos.")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("🔴 Riesgo alto", f"{alto:,}", help="Necesitan intervención urgente")
        rc2.metric("🟡 Riesgo medio", f"{medio:,}", help="Monitoreo y acciones preventivas")
        rc3.metric("🟢 Riesgo bajo", f"{bajo:,}", help="Mantener motivación")

        st.markdown("##### 🎯 Top 50 alumnos en riesgo (priorizados)")
        cols_mostrar = ["student_id", "age", "country", "attendance_rate",
                        "study_hours_weekly", "avg_quiz_score", "engagement_score",
                        "probabilidad_abandono", "nivel_riesgo"]
        cols_mostrar = [c for c in cols_mostrar if c in resultado.columns]
        st.dataframe(resultado[cols_mostrar].head(50), use_container_width=True, hide_index=True)

        # Descargar como CSV
        csv_export = resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar reporte completo (CSV)",
            csv_export,
            file_name="edutrack_reporte_riesgo.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Error procesando los datos: {e}")


if False:
    st.markdown("## ¿Qué diferencia a un alumno que termina vs uno que abandona?")
    st.caption("Comparativa basada en los datos reales de los 10,000 estudiantes de nuestra base.")

    # Cálculo de medias por grupo
    no_des = df_full[df_full["dropout"] == 0]
    si_des = df_full[df_full["dropout"] == 1]

    metricas = [
        ("attendance_rate", "Tasa de asistencia", "%", 100),
        ("study_hours_weekly", "Horas de estudio / semana", "hrs", 1),
        ("login_frequency_weekly", "Logins por semana", "logins", 1),
        ("avg_quiz_score", "Promedio en quizzes", "/100", 1),
        ("engagement_score", "Engagement", "/10", 1),
        ("video_watch_time_min", "Minutos de video vistos", "min", 1),
        ("assignments_submitted", "Tareas entregadas", "", 1),
        ("forum_posts", "Posts en foro", "", 1),
    ]

    # Tarjetas de "perfil ideal" vs "perfil en riesgo"
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{VERDE}; min-height:120px;'>
                <h4 style='color:{VERDE};'>✅ Perfil del alumno que TERMINA</h4>
                <p style='color:#555; margin:0;'>Activo, constante y participativo. Estos son los promedios reales:</p>
            </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
            <div class='tarjeta' style='border-left-color:{ROJO}; min-height:120px;'>
                <h4 style='color:{ROJO};'>⚠️ Perfil del alumno que ABANDONA</h4>
                <p style='color:#555; margin:0;'>Desconectado, baja participación, poco compromiso:</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Tabla comparativa con insights
    for col, label, unidad, mult in metricas:
        media_ok = no_des[col].mean() * mult
        media_drop = si_des[col].mean() * mult
        diff_pct = ((media_ok - media_drop) / media_drop) * 100 if media_drop != 0 else 0

        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            st.markdown(f"<div style='padding:10px; background:white; border-radius:8px; border-left:4px solid {VERDE};'>"
                        f"<b style='color:#333;'>{label}</b><br>"
                        f"<span style='color:{VERDE}; font-size:22px; font-weight:bold;'>{media_ok:.1f} {unidad}</span>"
                        f"</div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div style='padding:10px; background:white; border-radius:8px; border-left:4px solid {ROJO};'>"
                        f"<b style='color:#333;'>{label}</b><br>"
                        f"<span style='color:{ROJO}; font-size:22px; font-weight:bold;'>{media_drop:.1f} {unidad}</span>"
                        f"</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='padding:14px;'>"
                        f"<span style='color:#666;'>Los que terminan tienen </span>"
                        f"<b style='color:{AZUL};'>{abs(diff_pct):.0f}% más</b> "
                        f"<span style='color:#666;'>en esta métrica.</span>"
                        f"</div>", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")
    st.markdown("### 📈 Distribuciones lado a lado")
    st.caption("Verde = alumnos que terminan. Rojo = alumnos que abandonan. Las líneas punteadas son los promedios.")

    graficas = [
        ("attendance_rate", "Asistencia", "0 a 1"),
        ("study_hours_weekly", "Horas de estudio semanales", "horas"),
        ("avg_quiz_score", "Promedio en quizzes", "puntos"),
        ("engagement_score", "Engagement", "0 a 10"),
        ("login_frequency_weekly", "Logins por semana", "logins"),
        ("video_watch_time_min", "Minutos de video vistos", "minutos"),
    ]
    fila1 = st.columns(2); fila2 = st.columns(2); fila3 = st.columns(2)
    cols_g = list(fila1) + list(fila2) + list(fila3)
    for col, (cname, label, unidad) in zip(cols_g, graficas):
        with col:
            st.pyplot(grafica_comparativa(cname, label, unidad))

    st.markdown("---")
    st.markdown("### 💡 Cómo usar esta información")
    st.markdown(f"""
    <div class='tarjeta'>
        <h4>Para tratar a un nuevo alumno:</h4>
        <ul style='color:#444; line-height:1.8;'>
            <li><b style='color:{VERDE};'>Imítale al perfil verde:</b> garantiza que asista regularmente, lo motives a participar en foros y mantengas un engagement alto desde el día 1.</li>
            <li><b style='color:{ROJO};'>Evita el perfil rojo:</b> si en las primeras 2 semanas detectas baja asistencia, pocas horas o cero participación, EduTrack lo marca y lanzas una intervención antes de que sea tarde.</li>
            <li><b style='color:{AZUL};'>La regla de oro:</b> los alumnos que terminan tienen el doble (o más) de actividad que los que abandonan. La actividad temprana es el mejor predictor.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("EduTrack · Proyecto final 4Geeks Academy · Hecho por estudiantes de Data Science")
