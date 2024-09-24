import streamlit as st
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import joblib

# Pastikan stopwords diunduh jika belum ada
nltk.download('stopwords')

# Fungsi Preprocessing
def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join(text)

    # Lowercase teks
    txt_casefold = text.lower()

    # Menghapus URL
    txt_nourl = re.sub(r'http\S+|www\S+|https\S+', '', txt_casefold, flags=re.MULTILINE)

    # Menghapus mentions (@username)
    txt_no_mentions = re.sub(r'@\w+', '', txt_nourl)

    # Menghapus hashtag
    txt_no_hashtags = re.sub(r'#\w+', '', txt_no_mentions)

    # Menghapus angka
    txt_no_numbers = re.sub(r'\d+', '', txt_no_hashtags)

    # Menghapus tanda baca dan tokenisasi
    tokenizer = RegexpTokenizer(r'\w+')
    text_token = tokenizer.tokenize(txt_no_numbers)

    # Menghapus stop words Bahasa Indonesia
    stop_words = stopwords.words('indonesian')
    stop_words.extend(['scroll', 'to', 'continue', 'with', 'advertisement', 'content'])
    stop_words = set(stop_words)

    text_filtered = [word for word in text_token if word not in stop_words]

    # Menggabungkan kembali kata-kata menjadi kalimat
    clean_text = ' '.join(text_filtered)
    return clean_text

# Fungsi untuk memberi label kategori berita
def label_berita(hasil):
    label = {0: 'Kesehatan', 1: 'Pariwisata'}
    return label.get(hasil, "Kategori tidak dikenali")

# Memuat model yang sudah disimpan
loaded_model = joblib.load('logistic_model.pkl')
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')

# Menambahkan opsi pilihan fitur
st.title("Pilih Fitur Aplikasi")

# Opsi pilihan pengguna
choice = st.radio("Pilih fitur yang ingin Anda gunakan:", ("Upload CSV dan Tampilkan Data", "Aplikasi Klasifikasi Berita"))

# Fitur Upload CSV
if choice == "Upload CSV dan Tampilkan Data":
    st.header("Upload CSV dan Tampilkan Data")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)
        st.write("Isi File CSV:")
        st.write(df)

# Fitur Aplikasi Klasifikasi Berita
elif choice == "Aplikasi Klasifikasi Berita":
    st.header("Aplikasi Klasifikasi Berita")
    
    # Input dari pengguna
    input_text = st.text_area("Masukkan teks berita yang ingin diklasifikasi:")
    
    # Tombol untuk klasifikasi
    if st.button("Prediksi Kategori Berita"):
        if input_text:
            # Preprocess teks input
            processed_text = preprocess_text(input_text)
            
            # Lakukan prediksi
            hasil_prediksi = loaded_model.predict(loaded_tfidf.transform([processed_text]))
            kategori_prediksi = label_berita(hasil_prediksi[0])
    
            # Tampilkan hasil prediksi
            st.write(f"Kategori berita: **{kategori_prediksi}**")
        else:
            st.write("Silakan masukkan teks berita untuk diklasifikasi.")
