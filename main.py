# python -m streamlit run main.py

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import pandas as pd
import tempfile
import os


def calculer_derivative(temps, valeurs):
    """
    Calcule la dérivée discrète approximative d'une série temporelle à partir des valeurs mesurées.

    f'(t_i) ≈ (f(t_{i+1}) - f(t_i)) / (t_{i+1} - t_i)

    Args:
        temps (array-like): Timestamps en secondes.
        valeurs (array-like): Valeurs du signal.

    Returns:
        np.array: Tableau des dérivées (longueur = len(temps)-1).
    """
    temps = np.array(temps, dtype=float)
    valeurs = np.array(valeurs, dtype=float)
    dt = np.diff(temps)
    dv = np.diff(valeurs)
    dt = np.where(dt == 0, 1e-6, dt)
    deriv = dv / dt
    return deriv


def convertir_en_min_sec(seconds):
    """
    Convertit un temps (en secondes) en format mm:ss.

    Args:
        seconds (float): Temps en secondes.

    Returns:
        str: Temps formaté en mm:ss.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    return f"{minutes:02d}:{sec:02d}"


def transcrire_audio_whisper(uploaded_file):
    """
    Transcrit le fichier audio uploadé avec Whisper.
    Le fichier audio est sauvegardé temporairement pour la transcription.

    Returns:
        list: Liste des segments de transcription sous forme de dictionnaire avec 'start', 'end' et 'text'.
    """
    try:
        import whisper
    except ImportError:
        st.error("Le module 'whisper' n'est pas installé. Installez-le avec 'pip install -U openai-whisper'.")
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_audio_path = temp_file.name

    model = whisper.load_model("small")
    result = model.transcribe(temp_audio_path, language="fr")
    os.remove(temp_audio_path)
    return result.get("segments", [])


# Titre de l'application
st.title("Analyse du signal audio et concordancier des variations")

st.markdown("""
### Instructions et explications
- **Importation du fichier audio :**  
  Le fichier audio (WAV) est lu et, s'il est stéréo, converti en mono.
- **Échantillonnage :**  
  Pour limiter la quantité de données envoyées au navigateur, le signal est échantillonné en sélectionnant uniformément un sous-ensemble des points.  
  Plus le nombre de points est élevé, plus la représentation est précise, mais cela peut ralentir l'affichage.
- **Analyse statistique :**  
  La moyenne (μ) et l'écart‑type (σ) du signal échantillonné sont calculés. L'intervalle de détection [μ ± k×σ], avec k ajusté via un slider, permet d'identifier les points atypiques.
- **Transcription :**  
  Si vous cochez l'option "Afficher la transcription", le fichier audio sera transcrit via Whisper. Pour chaque point atypique, le script recherche dans la transcription les segments couvrant la fenêtre [t − 1, t + 1].
""")

# Téléchargement du fichier audio
uploaded_file = st.file_uploader("Importer un fichier audio (WAV)", type=["wav"])

# Option pour afficher la transcription
afficher_transcription = st.checkbox("Afficher la transcription (avant l'analyse)", value=False)

# Slider pour le nombre maximum de points à afficher
nb_points = st.slider("Nombre de points à afficher", min_value=100, max_value=100000, value=50000, step=100)

# Slider pour le paramètre k (pour l'intervalle [μ ± k×σ])
k_value = st.slider("Définissez le paramètre k (pour l'intervalle [μ ± k×σ])", min_value=0.5, max_value=6.0, value=2.0,
                    step=0.1)

if st.button("Lancer l'analyse"):
    if uploaded_file is not None:
        try:
            data, samplerate = sf.read(uploaded_file)
            st.write(f"Taux d'échantillonnage : {samplerate} Hz")
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier audio : {e}")
            st.stop()

        # Conversion en mono si le signal est stéréo
        if data.ndim > 1:
            data = data.mean(axis=1)

        n_samples = len(data)
        duration = n_samples / samplerate
        st.info(f"Durée du fichier audio : **{duration:.2f} secondes** ({n_samples} échantillons).")

        st.info("Pour de bonnes performances, le signal sera échantillonné uniformément.")
        st.markdown("""
        **Remarque sur l'échantillonnage :**  
        Un échantillonnage consiste à ne conserver qu'un sous-ensemble des points du signal.  
        Cela simplifie le signal (en prenant par exemple la moyenne sur de petits intervalles) et améliore la réactivité de l'interface,  
        mais peut lisser certains détails fins.
        """)

        # Création du vecteur temps complet
        temps_complet = np.linspace(0, duration, n_samples)
        if n_samples > nb_points:
            indices = np.linspace(0, n_samples - 1, nb_points, dtype=int)
        else:
            indices = np.arange(n_samples)
        temps_sample = np.array(temps_complet)[indices]
        data_sample = np.array(data)[indices]

        # Calcul de la moyenne et de l'écart‑type
        mu = np.mean(data_sample)
        sigma = np.std(data_sample)
        st.write(f"Moyenne (μ) : {mu:.4f}")
        st.write(f"Écart‑type (σ) : {sigma:.4f}")

        lower_bound = mu - k_value * sigma
        upper_bound = mu + k_value * sigma
        st.markdown(f"Intervalle de détection : **[{lower_bound:.4f}, {upper_bound:.4f}]**")

        # Détection des points atypiques
        indices_outliers = np.where((data_sample < lower_bound) | (data_sample > upper_bound))[0]
        temps_out = temps_sample[indices_outliers]
        data_out = data_sample[indices_outliers]

        #### Graphique 1 : Signal audio et intervalle statistique
        fig_signal = go.Figure()
        fig_signal.add_trace(go.Scatter(
            x=temps_sample,
            y=data_sample,
            mode="lines+markers",
            name="Signal audio",
            marker=dict(size=3),
            line=dict(color="blue")
        ))
        fig_signal.add_trace(go.Scatter(
            x=np.concatenate([temps_sample, temps_sample[::-1]]),
            y=np.concatenate([np.full_like(temps_sample, lower_bound),
                              np.full_like(temps_sample, upper_bound)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"Intervalle μ ± {k_value}σ"
        ))
        if len(temps_out) > 0:
            fig_signal.add_trace(go.Scatter(
                x=temps_out,
                y=data_out,
                mode="markers",
                marker=dict(color="red", size=6, symbol="diamond"),
                name="Points atypiques"
            ))
        fig_signal.update_layout(
            title="Signal audio et intervalle statistique",
            xaxis_title="Temps (s)",
            yaxis_title="Amplitude",
            hovermode="x unified",
            width=800,
            height=500
        )
        st.plotly_chart(fig_signal, use_container_width=True)

        #### Si l'option de transcription est activée, réaliser la transcription
        if afficher_transcription:
            st.info("Transcription en cours (cela peut prendre quelques minutes)...")
            transcription_segments = transcrire_audio_whisper(uploaded_file)
            if len(transcription_segments) == 0:
                st.warning("Aucun segment de transcription n'a été généré.")
        else:
            transcription_segments = []  # Définie comme liste vide

        #### Concordancier des points atypiques avec segments de texte
        st.subheader("Concordancier des points atypiques")
        if len(temps_out) > 0:
            concordance = []
            for t, val in zip(temps_out, data_out):
                segment_text = ""
                if transcription_segments:
                    # On concatène le texte des segments dont l'intervalle se superpose à [t-1, t+1]
                    segment_text = " ".join(
                        seg["text"].strip()
                        for seg in transcription_segments
                        if seg["end"] >= t - 1 and seg["start"] <= t + 1
                    )
                concordance.append({
                    "Timestamp (s)": f"{t:.3f}",
                    "Time (mm:ss)": convertir_en_min_sec(t),
                    "Amplitude": f"{val:.4f}",
                    "Segment texte": segment_text
                })
            df_concordance = pd.DataFrame(concordance)
            st.dataframe(df_concordance)

            st.download_button(
                label="Télécharger le concordancier en CSV",
                data=df_concordance.to_csv(index=False).encode("utf-8"),
                file_name="concordancier.csv",
                mime="text/csv"
            )
        else:
            st.info("Aucun point atypique n'a été détecté.")

        st.markdown("""
        **Interprétation générale :**
        - **Graphique :** Le signal audio (en bleu) et la zone ombragée indiquant l'intervalle [μ ± k×σ] permettent d'identifier les variations atypiques (affichées en vert foncé).  
        - **Concordancier :** La table liste, pour chaque point atypique, son timestamp (en secondes et format mm:ss), son amplitude, et, le cas échéant, le segment textuel correspondant (sur la fenêtre [t-1, t+1]).
        """)
    else:
        st.info("Veuillez importer un fichier audio (WAV).")

