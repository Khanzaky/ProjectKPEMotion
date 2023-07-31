# Core Pkgs
import streamlit as st 
import sqlite3
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

from PIL import Image

gambar_lokal = 'foto/logo.png'
st.image(gambar_lokal)


# Utils
import joblib 
pipe_lr = joblib.load(open("models/emotiondetector4.pickle","rb"))

conn = sqlite3.connect('data.db')

# Track Utils
from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# Fxn
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0].capitalize()

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"kesal":"😠", "senang":"🤗", "puas":"😂", "biasa saja":"😐", "sedih":"😢", "kecewa":"😔"}


# Main Application
def main():
    st.title("Aplikasi Kepuasan Pelanggan")
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "Home":
        add_page_visited_details("Home",datetime.now())
        st.subheader("Home-Emotion In Text")

        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply Fxn Here
            prediction = predict_emotions(raw_text)
            prediction = prediction.strip()
            probability = get_prediction_proba(raw_text)
            add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                if prediction in emotions_emoji_dict:
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                else:
                    st.write("Prediction: {}".format(prediction))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename', 'Time_of_Visit'])
            st.dataframe(page_visited_details)	

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)	

            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            df_emotions = df_emotions.apply(lambda col: col.astype(str, errors='ignore'))
            for entry in df_emotions.to_dict('records'):
                for key, value in entry.items():
                    if isinstance(value, str):
                        entry[key] = value.encode('utf-8', 'ignore').decode('utf-8')

            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)	

    else:
        st.subheader("About")
        st.markdown("Aplikasi Kepuasan Pelanggan di STAI Al-Azhary Cianjur adalah sebuah inovasi teknologi yang dikembangkan oleh STAI Al-Azhary Cianjur untuk memantau dan meningkatkan tingkat kepuasan mahasiswa dan pelanggan lainnya terhadap layanan dan fasilitas yang disediakan oleh institusi ini. Aplikasi ini bertujuan untuk memahami dan merespons kebutuhan dan harapan pelanggan, serta memastikan bahwa kualitas pendidikan dan pelayanan yang diberikan di STAI Al-Azhary Cianjur senantiasa meningkat.")
        st.markdown("## Fitur-fitur Utama Aplikasi Kepuasan Pelanggan:")
        st.markdown("- **Kuesioner Kepuasan Pelanggan**: Aplikasi ini menyediakan kuesioner interaktif yang dapat diisi oleh mahasiswa dan pelanggan lainnya. Kuesioner ini berisi pertanyaan terkait berbagai aspek pelayanan dan fasilitas yang ada di kampus, seperti proses pendaftaran, kualitas pengajaran, fasilitas laboratorium, perpustakaan, dan lain sebagainya. Kuesioner ini membantu mengumpulkan data dan umpan balik dari pelanggan, sehingga manajemen dapat melakukan evaluasi dan perbaikan yang diperlukan.")
        st.markdown("- **Analisis dan Laporan**: Aplikasi ini dilengkapi dengan fitur analisis data yang dapat menyajikan hasil survei dalam bentuk grafik dan laporan. Hal ini memungkinkan manajemen STAI Al-Azhary Cianjur untuk dengan cepat melihat tren kepuasan pelanggan, mengidentifikasi area yang memerlukan perhatian lebih, dan mengambil keputusan berdasarkan data yang telah terkumpul.")
        st.markdown("- **Notifikasi dan Tindak Lanjut**: Aplikasi ini juga memungkinkan pemberian notifikasi kepada manajemen setiap kali ada tanggapan kuesioner yang memerlukan tindak lanjut atau penanganan lebih lanjut. Fitur ini membantu memastikan bahwa keluhan atau masalah yang diajukan pelanggan mendapatkan respon dan solusi yang tepat secara tepat waktu.")
        st.markdown("- **Riwayat Kepuasan**: Aplikasi ini menyimpan riwayat kepuasan pelanggan dari waktu ke waktu. Data ini membantu melacak perubahan tingkat kepuasan dari semester ke semester atau dari tahun ke tahun, dan membantu membandingkan tren kepuasan antara periode waktu tertentu.")
        st.markdown("## Manfaat Aplikasi Kepuasan Pelanggan:")
        st.markdown("- **Meningkatkan Kualitas Layanan**: Dengan aplikasi ini, manajemen STAI Al-Azhary Cianjur dapat dengan cepat mengetahui kebutuhan dan masalah yang dihadapi pelanggan. Dengan begitu, mereka dapat melakukan perbaikan dan peningkatan layanan secara tepat waktu.")
        st.markdown("- **Memberikan Respons yang Cepat**: Aplikasi ini memungkinkan tanggapan cepat terhadap keluhan atau masalah yang disampaikan pelanggan. Dengan begitu, pelanggan merasa didengar dan dihargai, serta menumbuhkan rasa kepercayaan terhadap institusi.")
        st.markdown("- **Pengambilan Keputusan yang Lebih Baik**: Berkat analisis data yang disajikan oleh aplikasi, manajemen dapat mengambil keputusan yang didukung oleh data empiris dan menghindari keputusan berdasarkan asumsi semata.")
        st.markdown("- **Peningkatan Reputasi dan Daya Saing**: Dengan terus meningkatkan kualitas layanan berdasarkan umpan balik pelanggan, STAI Al-Azhary Cianjur dapat memperkuat reputasi sebagai institusi pendidikan yang peduli dan responsif terhadap kebutuhan mahasiswa.")
        st.markdown("Dengan Aplikasi Kepuasan Pelanggan di STAI Al-Azhary Cianjur, institusi ini dapat bergerak maju menuju tingkat keunggulan dalam memberikan pendidikan dan pelayanan yang terbaik untuk para pelanggannya.")
        add_page_visited_details("About", datetime.now())

if __name__ == '__main__':
    main()
