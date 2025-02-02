import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

st.title('Analisis Kualitas Udara')
st.markdown(
    """
**Pertanyaan Bisnis:** \n
1. Bagaimana tren konsentrasi PM2.5 di berbagai stasiun dari tahun 2013 hingga 2017?
2. Bagaimana hubungan antara faktor meteorologi (suhu, tekanan udara, kecepatan angin) dengan tingkat polusi PM2.5?
"""
)

@st.cache_data  # Mengganti st.cache dengan st.cache_data
def load_data():
    # Sesuaikan path dengan struktur folder GitHub
    base_path = os.path.dirname(os.path.abspath(__file__))  # Path file dashboard.py
    folder_path = os.path.abspath(os.path.join(base_path, "../air_quality_datasets"))
    
    # Debugging untuk Streamlit Cloud
    st.write("Base path:", base_path)
    st.write("Absolute folder path:", folder_path)
    
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Debugging: Tampilkan file yang ditemukan
    print(f"Found {len(csv_files)} CSV files")
    
    # Handle case tidak ada file CSV
    if not csv_files:
        raise ValueError("Tidak ada file CSV yang ditemukan di folder yang ditentukan")
    
    df_list = []
    for file in csv_files:
        try:
            # Skip file kosong
            if os.path.getsize(file) > 0:
                df = pd.read_csv(file)
                # Pastikan kolom yang diperlukan ada
                if {'year', 'month', 'day', 'hour'}.issubset(df.columns):
                    df_list.append(df)
        except Exception as e:
            print(f"Error membaca file {file}: {str(e)}")
    
    # Handle case semua file error/kosong
    if not df_list:
        raise ValueError("Tidak ada data yang valid untuk diproses")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Handle missing dates
    combined_df['date'] = pd.to_datetime(
        combined_df[['year', 'month', 'day', 'hour']].astype(str).agg('-'.join, axis=1) + ':00:00',
        format='%Y-%m-%d-%H:%M:%S',
        errors='coerce'
    )
    
    # Handle missing values
    combined_df['PM2.5'] = combined_df.groupby('station')['PM2.5'].transform(
        lambda x: x.fillna(x.median() if not x.isnull().all() else 0)
    )
    
    return combined_df

try:
    df = load_data()
except ValueError as e:
    st.error(str(e))
    st.stop()

# Sidebar Filter
st.sidebar.header('Filter Data')
selected_stations = st.sidebar.multiselect(
    'Pilih Stasiun',
    options=df['station'].unique(),
    default=[df['station'].unique()[0]] if len(df['station'].unique()) > 0 else []
)

# Handle date input dengan aman
try:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
except AttributeError:
    st.error("Data tanggal tidak valid")
    st.stop()

start_date = st.sidebar.date_input('Tanggal Awal', min_date)
end_date = st.sidebar.date_input('Tanggal Akhir', max_date)

# Filter Data
filtered_df = df[
    (df['station'].isin(selected_stations)) &
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date))
]

# Visualisasi Tren PM2.5
st.subheader('Tren PM2.5 per Stasiun')
if not filtered_df.empty and not filtered_df['date'].isnull().all():
    fig, ax = plt.subplots(figsize=(12,6))
    for station in selected_stations:
        station_data = filtered_df[filtered_df['station'] == station]
        if not station_data.empty:
            monthly_avg = station_data.resample('M', on='date')['PM2.5'].mean()
            ax.plot(monthly_avg.index, monthly_avg, label=station)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('Tahun')
    plt.ylabel('PM2.5 (µg/m³)')
    st.pyplot(fig)
else:
    st.write('Tidak ada data yang dipilih atau data tanggal tidak valid.')

# Visualisasi lainnya tetap sama...

st.subheader('Korelasi PM2.5 dengan Faktor Meteorologi')
corr_matrix = filtered_df[['PM2.5', 'TEMP', 'PRES', 'WSPM', 'DEWP']].corr()
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader('Klasterisasi Stasiun Berdasarkan PM2.5')
station_avg_pm = df.groupby('station')['PM2.5'].mean().reset_index()
station_avg_pm['cluster'] = pd.qcut(station_avg_pm['PM2.5'], q=3, labels=['Rendah', 'Sedang', 'Tinggi'])
fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(x='PM2.5', y='station', data=station_avg_pm.sort_values('PM2.5'), hue='cluster', dodge=False, ax=ax)
plt.xlabel('Rata-Rata PM2.5 (µg/m³)')
plt.ylabel('Stasiun')
st.pyplot(fig)

st.markdown(
    """
# Kesimpulan

1. **Tren PM2.5** menunjukkan pola musiman dengan puncak di musim dingin karena peningkatan penggunaan pemanas batubara dan kondisi inversi atmosfer.
2. **Faktor meteorologi** yang signifikan:
   - Suhu rendah berkorelasi dengan polusi tinggi.
   - Kecepatan angin rendah berhubungan dengan akumulasi polutan.
3. **Rekomendasi**:
   - Fokus kontrol polusi di stasiun dengan risiko tinggi.
   - Penguatan regulasi emisi selama musim dingin.
   - Sistem peringatan dini berbasis prediksi meteorologi.
"""
)
