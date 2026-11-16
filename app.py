import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="LeafGuard AI - Deteksi Penyakit Tanaman",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS (Tema Botanical Green & Modern Card UI)
st.markdown("""
<style>
    /* Latar belakang utama gradasi hijau lembut */
    .stApp {
        background: linear-gradient(135deg, #f1f8f4 0%, #e3f1e8 100%);
    }
    
    /* Styling Sidebar */
    [data-testid="stSidebar"] {
        background-color: #143625;
    }
    [data-testid="stSidebar"] * {
        color: #f1f8f4 !important;
    }

    /* Header Banner */
    .hero-box {
        background: linear-gradient(120deg, #1b4332 0%, #2d6a4f 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        box-shadow: 0 10px 25px rgba(27, 67, 50, 0.18);
        margin-bottom: 1.8rem;
    }
    .hero-box h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
        font-weight: 700;
    }
    .hero-box p {
        color: #d8f3dc !important;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Card Container */
    .result-card {
        background-color: #ffffff;
        padding: 1.6rem;
        border-radius: 14px;
        border-left: 6px solid #2d6a4f;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
    }
    .result-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #52796f;
        font-weight: 600;
    }
    .result-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1b4332;
        margin: 0.3rem 0;
    }

    /* Tombol Deteksi Custom */
    div.stButton > button {
        background: linear-gradient(90deg, #2d6a4f 0%, #40916c 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.25);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 100%);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# 3. Memuat Model AI
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_daun.h5')

model = load_model()

CLASS_NAMES = [
    'Early Blight (Bercak Kering)',
    'Late Blight (Busuk Daun)',
    'Healthy (Daun Sehat)'
]

INFO_PENYAKIT = {
    'Early Blight (Bercak Kering)': {
        'status': '⚠️ Terinfeksi Jamur Alternaria solani',
        'warna': '#d97706',
        'gejala': 'Muncul bercak cokelat tua melingkar seperti papan target dengan tepi menguning pada daun bagian bawah.',
        'solusi': 'Pangkas daun yang terinfeksi, hindari menyiram air langsung ke permukaan daun, dan aplikasikan fungisida berbahan aktif mankozeb atau klorotalonil.'
    },
    'Late Blight (Busuk Daun)': {
        'status': '🚨 Terinfeksi Oomycete Phytophthora infestans',
        'warna': '#dc2626',
        'gejala': 'Bercak basah berwarna hijau keabu-abuan atau cokelat kehitaman yang menyebar sangat cepat saat cuaca lembap.',
        'solusi': 'Segera isolasi atau musnahkan tanaman yang terinfeksi berat agar spora tidak menular ke seluruh lahan, lalu semprotkan fungisida berbahan tembaga.'
    },
    'Healthy (Daun Sehat)': {
        'status': '✅ Kondisi Optimal & Bebas Patogen',
        'warna': '#16a34a',
        'gejala': 'Permukaan daun berwarna hijau merata, segar, dan tidak ditemukan nekrosis maupun bercak jamur.',
        'solusi': 'Pertahankan jadwal penyiraman di area perakaran, jaga sirkulasi udara antar tanaman, dan berikan pupuk berimbang secara berkala.'
    }
}

# 4. Sidebar Informasi Proyek (Berguna saat Presentasi)
with st.sidebar:
    st.markdown("## 🌱 LeafGuard AI")
    st.caption("Proyek Akhir Mata Kuliah Kecerdasan Buatan")
    st.markdown("---")
    st.markdown("**Spesifikasi Sistem:**")
    st.markdown("• **Algoritma:** CNN (MobileNetV2)")
    st.markdown("• **Dataset:** PlantVillage (300 Citra)")
    st.markdown("• **Input Citra:** 224 × 224 px (RGB)")
    st.markdown("• **Kelas Target:** 3 Kondisi Daun")
    st.markdown("---")
    st.markdown("**Alur Pemrosesan AI:**")
    st.markdown("1. *Resize* & Normalisasi Piksel (0–1)")
    st.markdown("2. Ekstraksi Fitur Konvolusi")
    st.markdown("3. Klasifikasi Probabilitas *Softmax*")

# 5. Header Banner Utama
st.markdown("""
<div class="hero-box">
    <h1>🌿 LeafGuard AI — Deteksi Penyakit Tanaman</h1>
    <p>Sistem klasifikasi kesehatan daun berbasis <b>Convolutional Neural Network (CNN)</b> untuk diagnosis cepat dan rekomendasi penanganan.</p>
</div>
""", unsafe_allow_html=True)

# 6. Tata Letak 2 Kolom (Kiri: Input Gambar | Kanan: Hasil Analisis)
col_kiri, col_kanan = st.columns([1, 1.15], gap="large")

with col_kiri:
    st.subheader("📤 1. Unggah Citra Daun")
    uploaded_file = st.file_uploader(
        "Pilih foto daun tanaman (JPG, JPEG, atau PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Pratinjau Citra Daun', use_container_width=True)
        tombol_deteksi = st.button("🔍 Analisis Kondisi Daun")
    else:
        st.info("Silakan unggah foto daun di kotak atas untuk memulai proses deteksi.")
        tombol_deteksi = False

with col_kanan:
    st.subheader("📊 2. Hasil Diagnosis AI")

    if uploaded_file is not None and tombol_deteksi:
        with st.spinner('Mengekstrak fitur visual dan menghitung probabilitas...'):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)[0]
            predicted_index = np.argmax(predictions)
            hasil_kelas = CLASS_NAMES[predicted_index]
            confidence = predictions[predicted_index] * 100
            detail = INFO_PENYAKIT[hasil_kelas]

        # Kartu Hasil Utama
        st.markdown(f"""
        <div class="result-card" style="border-left-color: {detail['warna']};">
            <div class="result-title">Hasil Klasifikasi</div>
            <div class="result-value" style="color: {detail['warna']};">{hasil_kelas}</div>
            <div style="font-weight: 600; color: #374151; margin-bottom: 0.4rem;">{detail['status']}</div>
            <div style="font-size: 0.95rem; color: #4b5563;">Tingkat Keyakinan (Confidence): <b>{confidence:.2f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

        # Detail Gejala & Solusi dalam Tab Rapi
        tab1, tab2, tab3 = st.tabs(["🩺 Karakteristik Gejala", "💊 Saran Penanganan", "📈 Grafik Probabilitas"])
        
        with tab1:
            st.write(detail['gejala'])
        with tab2:
            st.write(detail['solusi'])
        with tab3:
            df_prob = pd.DataFrame({
                'Kelas Kondisi': CLASS_NAMES,
                'Probabilitas (%)': [round(float(p) * 100, 2) for p in predictions]
            }).set_index('Kelas Kondisi')
            st.bar_chart(df_prob, color="#2d6a4f")

    elif uploaded_file is not None:
        st.warning("Klik tombol **'🔍 Analisis Kondisi Daun'** di sebelah kiri untuk melihat hasil prediksi.")
    else:
        st.markdown("""
        <div class="result-card">
            <div class="result-title">Menunggu Input</div>
            <div class="result-value">Belum Ada Gambar</div>
            <p style="color: #6b7280; margin: 0;">Hasil deteksi penyakit, tingkat keyakinan model, grafik probabilitas, dan saran penanganan akan muncul di panel ini.</p>
        </div>
        """, unsafe_allow_html=True)