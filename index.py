import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import re
import time
from datetime import date
import google.generativeai as genai

# Set layout halaman agar lebar
st.set_page_config(page_title="Analisis Komplain BRImo", layout="wide")

# --- INJEKSI CSS TEMA BIRU ---
st.markdown("""
    <style>
    thead tr th {
        background-color: #E0F2FE !important;
        color: #0369A1 !important;
        font-size: 16px !important;
        border-bottom: 2px solid #7DD3FC !important;
    }
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th, td {
        border: 1px solid #BAE6FD !important;
    }
    </style>
""", unsafe_allow_html=True)

def blue_divider():
    st.markdown("<hr style='border: 1.5px solid #7DD3FC; border-radius: 5px; margin-top: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# Konfigurasi Kategori 
# --- LOAD KONFIGURASI DARI JSON ---
CONFIG_FILE = 'categories_config.json'
UNCATEGORIZED_FILE = 'uncategorized_complaints.csv'

def load_categories():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_categories(config_data):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

# Load konfigurasi awal
categories_config = load_categories()

def categorize_review(text):
    text_lower = str(text).lower()
    for category, data in categories_config.items():
        for kw in data['keywords']:
            if kw in text_lower:
                return category
    return "Lainnya"

def parse_version(ver_str):
    nums = re.findall(r'\d+', str(ver_str))
    return tuple(int(n) for n in nums) if nums else (0,)

def update_uncategorized_file(df):
    uncat_df = df[
        (df['Kategori'] == 'Lainnya') &
        (df['Star Rating'] <= 3) &
        (df['Review Text'].apply(is_valid_complaint))
    ]

    uncat_df.to_csv(UNCATEGORIZED_FILE, index=True)
    return len(uncat_df)

def is_valid_complaint(review_text):
    if pd.isna(review_text):
        return False

    text = str(review_text).strip()

    if not text:
        return False

    meaningful_text = re.sub(r'[^A-Za-z0-9À-ÿ\s]', '', text)
    char_count = len(re.sub(r'\s+', '', meaningful_text))

    return char_count >= 5

# --- INISIALISASI DATA DI SESSION STATE ---
@st.cache_data
def load_data_initial(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-16')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv('reviews_brimo_202611.csv', encoding='utf-16')

    df = df.dropna(subset=['Review Text', 'Star Rating'])

    df['Kategori'] = df['Review Text'].apply(categorize_review)

    df['Date'] = pd.to_datetime(
        df['Review Submit Date and Time'],
        errors='coerce'
    )

    df = df.dropna(subset=['Date'])

    return df

uploaded_file = st.sidebar.file_uploader(
    "Upload file review",
    type=["csv"]
)

if uploaded_file:
    st.sidebar.success(
        f"Loaded: {uploaded_file.name}"
    )
else:
    st.sidebar.info(
        "Using default dataset"
    )

# Menyimpan df di session_state agar hasil AI tersimpan dan tidak reset saat tanggal diubah
if ('current_file' not in st.session_state
    or st.session_state.current_file != (
        uploaded_file.name if uploaded_file else "default"
    )
):
    st.session_state.current_file = (
        uploaded_file.name if uploaded_file else "default"
    )

    st.session_state.raw_df = load_data_initial(
        uploaded_file
    ).copy()

# Referensi ke data yang bisa dimodifikasi oleh AI
raw_df = st.session_state.raw_df

# --- SIDEBAR: FILTER TANGGAL ---
st.sidebar.markdown("<h2 style='color: #0369A1;'>🔍 Filter Data</h2>", unsafe_allow_html=True)
min_data_date = raw_df['Date'].min().date()
max_data_date = raw_df['Date'].max().date()
target_start = date(max_data_date.year, 10, 11)
default_start = max(min_data_date, min(target_start, max_data_date))

selected_dates = st.sidebar.date_input(
    "Rentang Tanggal Ulasan",
    value=(default_start, max_data_date),
    min_value=min_data_date,
    max_value=max_data_date
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
elif isinstance(selected_dates, tuple) and len(selected_dates) == 1:
    start_date = end_date = selected_dates[0]
else:
    start_date = end_date = selected_dates

# Update view df sesuai filter
df = raw_df[(raw_df['Date'].dt.date >= start_date) & (raw_df['Date'].dt.date <= end_date)].copy()

# --- SIDEBAR: AI SMART CATEGORIZATION ---
st.sidebar.markdown("<br><h2 style='color: #0369A1;'>🧠 AI Smart Categorization</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 13px; color: #555;'>Gunakan Gemini AI untuk memproses keluhan 'Lainnya' dan mempelajari kategori kendala baru secara otomatis.</p>", unsafe_allow_html=True)
st.sidebar.markdown(
    "<h2 style='color: #0369A1;'>📂 Upload Review CSV</h2>",
    unsafe_allow_html=True
)


# Hitung jumlah file uncategorized saat ini
uncat_count = 0
if os.path.exists(UNCATEGORIZED_FILE):
    uncat_df_temp = pd.read_csv(UNCATEGORIZED_FILE, index_col=0)
    uncat_count = len(uncat_df_temp)

st.sidebar.info(f"📂 Ada **{uncat_count}** keluhan siap dipelajari AI.")

api_key = st.sidebar.text_input("Google Gemini API Key", type="password")

if st.sidebar.button("🚀 Proses 'Lainnya' dengan AI"):
    if not api_key:
        st.sidebar.error("API Key wajib diisi untuk menggunakan AI!")
    else:
        try:
            genai.configure(api_key=api_key)
            
            # Auto-detect available models to prevent 404 errors
            # available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # print(available_models)
            target_model_name = 'models/gemini-3.5-flash-lite' # Safe default
            # for m in ['models/gemini-1.5-flash', 'models/gemini-1.5-flash-latest', 'models/gemini-1.5-pro-latest']:
            #     if m in available_models:
            #         target_model_name = m
            #         break
            
            model = genai.GenerativeModel(target_model_name)
                        
                        # Load uncategorized file
            uncategorized_df = pd.read_csv(UNCATEGORIZED_FILE, index_col=0)
                        
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            current_config = load_categories()
            
            # Konversi dataframe ke list of dicts untuk diproses ber-batch (50 data per batch)
            records = [{"id": idx, "text": row['Review Text']} for idx, row in uncategorized_df.iterrows()]
            batch_size = 50 
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]
                batch_json = json.dumps(batch, ensure_ascii=False)
                
                prompt = f"""
                Anda adalah analis aplikasi perbankan. Kategorikan daftar keluhan pelanggan berikut yang dikirim dalam format JSON.
                
                Kategori Utama saat ini:
                {', '.join(current_config.keys())}
                
                Input Keluhan:
                {batch_json}
                
                Instruksi:
                1. Untuk SETIAP ID, tentukan 'kategori' yang paling tepat (Pilih dari Kategori Utama yang ada, ATAU buat Kategori Baru maks 4 kata jika benar-benar isu baru).
                2. Ekstrak 1 'keyword_baru' spesifik (maks 2 kata, huruf kecil) sebagai akar masalah (misal: "qris", "fingerprint", "otp").
                3. Jika ulasan tidak memiliki arti (spam), set kategori = "Spam / Tidak Jelas" dan keyword_baru = "spam".
                
                Balas HANYA dengan array JSON murni berisikan objek dengan keys: "id", "kategori", "keyword_baru". Tanpa penjelasan tambahan, tanpa formatting markdown ```json.
                Contoh Output yang valid:
                [
                    {{"id": 105, "kategori": "Login, registrasi & akses akun", "keyword_baru": "otp"}},
                    {{"id": 201, "kategori": "Transfer & BI-FAST", "keyword_baru": "pending"}}
                ]
                """
                
                try:
                    response = model.generate_content(prompt)
                    result_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    ai_results = json.loads(result_text)
                    
                    # Update config dan main DataFrame berdasarkan balasan AI
                    for res in ai_results:
                        idx = res.get("id")
                        final_category = res.get("kategori", "Lainnya")
                        new_keyword = res.get("keyword_baru", "").lower()
                        
                        if final_category != "Spam / Tidak Jelas" and new_keyword:
                            if final_category in current_config:
                                if new_keyword not in current_config[final_category]['keywords']:
                                    current_config[final_category]['keywords'].append(new_keyword)
                            else:
                                current_config[final_category] = {
                                    "keywords": [new_keyword],
                                    "desc": ["✨ Kategori baru dipelajari oleh AI."]
                                }
                        
                        if idx in st.session_state.raw_df.index:
                            st.session_state.raw_df.loc[idx, 'Kategori'] = final_category

                except Exception as e:
                    print(f"Error AI batch {i}: {e}")
                    time.sleep(2)
                    
                progress = min((i + batch_size) / len(records), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Memproses {min(i+batch_size, len(records))}/{len(records)} ulasan secara batch...")
                time.sleep(2) 
            
            # Simpan pembaruan JSON
            save_categories(current_config)
            categories_config = current_config 
            
            # Re-evaluasi seluruh raw_df menggunakan dictionary terbaru 
            def re_categorize(row):
                if row['Kategori'] == 'Lainnya':
                    return categorize_review(row['Review Text'])
                return row['Kategori']
            
            st.session_state.raw_df['Kategori'] = st.session_state.raw_df.apply(re_categorize, axis=1)
            
            # Perbarui file uncategorized
            sisa_uncat = update_uncategorized_file(st.session_state.raw_df)
            
            st.sidebar.success(f"Selesai! {len(records)} ulasan berhasil dipelajari AI. Tersisa {sisa_uncat} Lainnya.")
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.sidebar.error(f"Gagal memproses AI: {e}")

# --- HEADER ---
st.markdown("<h1 style='color: #075985;'>📊 Analisis Komplain BRImo di Google Playstore</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='color: #0284C7; margin-top: -15px;'>Periode: {start_date.strftime('%d %B %Y')} - {end_date.strftime('%d %B %Y')}</h3>", unsafe_allow_html=True)
blue_divider()

if df.empty:
    st.warning("Tidak ada data ulasan pada rentang tanggal yang dipilih.")
    st.stop()

# --- TOP ROW: 3 COLUMNS ---
col1, col2, col3 = st.columns(3)

total_reviews = len(df)
complaints_df = df[
    (df['Star Rating'] <= 3) &
    (df['Review Text'].apply(is_valid_complaint))
]

with col1:
    rating_counts = df['Star Rating'].value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)
    star_labels = [f"{'⭐' * int(r)}" for r in rating_counts.index]
    
    color_map = {
        "⭐⭐⭐⭐⭐": "#28a745", "⭐⭐⭐⭐": "#85e085", "⭐⭐⭐": "#ffc107", "⭐⭐": "#fd7e14", "⭐": "#dc3545"
    }

    fig = px.pie(values=rating_counts.values, names=star_labels, hole=0.5, color=star_labels, color_discrete_map=color_map, category_orders={"names": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐"]})
    fig.update_layout(title={"text": "Distribusi Star Rating", "font": {"size": 18, "color": "#075985"}}, margin=dict(t=40, b=20, l=10, r=10), showlegend=False)
    fig.update_traces(textinfo='label+percent', textfont_size=13)
    st.plotly_chart(fig, width='stretch')
    
    summary_html = "<div style='background-color: #F0F9FF; border: 1px solid #BAE6FD; padding: 15px; border-radius: 10px; font-size: 15px; line-height: 1.6; text-align: center; margin-top: -20px; color: #0C4A6E;'>"
    for r in range(5, 0, -1):
        count = rating_counts.get(r, 0)
        pct = (count / total_reviews) * 100 if total_reviews > 0 else 0
        summary_html += f"<b>{'⭐' * r}</b> : {count:,} ulasan ({pct:.2f}%)<br>"
    summary_html += "</div>"
    st.markdown(summary_html, unsafe_allow_html=True)

with col2:
    st.markdown("<h3 style='color: #075985;'>🛒 Produk yang Sering Dikomplain</h3>", unsafe_allow_html=True)
    text_corpus = " ".join(complaints_df['Review Text'].astype(str).tolist()).lower()
    potential_products = ['emas', 'pdam', 'briva', 'qris', 'listrik', 'token', 'asuransi', 'pinjaman', 'ceria', 'kartu kredit', 'deposito', 'tiket', 'kai', 'investasi', 'bpjs', 'pajak', 'internet', 'pbb', 'donasi', 'valas']
    
    product_counts = []
    for prod in potential_products:
        count = len(re.findall(rf"\b{prod}\b", text_corpus))
        if count > 0:
            product_counts.append({"Produk": prod.title(), "Jumlah Komplain": count})
    
    if product_counts:
        prod_df = pd.DataFrame(product_counts).sort_values(by="Jumlah Komplain", ascending=False).reset_index(drop=True)
        prod_df.index = prod_df.index + 1
        styled_prod = prod_df.style.background_gradient(cmap='Blues', subset=['Jumlah Komplain'])
        st.dataframe(styled_prod, width='stretch')
    else:
        st.info("Belum ada produk terdeteksi.")

with col3:
    st.markdown("<h3 style='color: #075985;'>📱 Komplain Berdasarkan Versi Aplikasi</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #0284C7; font-size: 14px; margin-top: -15px;'>10 versi terbaru (diurutkan dari versi paling baru)</p>", unsafe_allow_html=True)

    ver_complaints = df[df['Star Rating'] <= 3]
    
    if not ver_complaints.empty:
        ver_stats = ver_complaints.groupby('App Version Name').agg(Jumlah_Komplain=('Star Rating', 'count')).reset_index()
        ver_stats['Persentase'] = (ver_stats['Jumlah_Komplain'] / len(ver_complaints) * 100).round(2).astype(str) + "%"
        ver_stats['ver_tuple'] = ver_stats['App Version Name'].apply(parse_version)
        ver_stats = ver_stats.sort_values(by='ver_tuple', ascending=False).head(10)
        
        ver_display = ver_stats.set_index('App Version Name')[['Jumlah_Komplain', 'Persentase']]
        styled_ver = ver_display.style.background_gradient(cmap='Blues', subset=['Jumlah_Komplain'])
        st.dataframe(styled_ver, width='stretch')
    else:
        st.info("Tidak ada data komplain.")

blue_divider()

# --- SECTION 2: TABEL TOP 5 KENDALA PER RATING ---
st.markdown("<h3 style='color: #075985;'>🚨 Highlight Top 5 Kendala per Rating</h3>", unsafe_allow_html=True)

valid_complaints = complaints_df[~complaints_df['Kategori'].isin(['Lainnya', 'Spam / Tidak Jelas'])]
top_5_cats = valid_complaints['Kategori'].value_counts().head(5).index

summary_rows = []
for cat in top_5_cats:
    cat_df = valid_complaints[valid_complaints['Kategori'] == cat]
    summary_rows.append({
        "Kendala": cat,
        "⭐ (Rating 1)": len(cat_df[cat_df['Star Rating'] == 1]),
        "⭐⭐ (Rating 2)": len(cat_df[cat_df['Star Rating'] == 2]),
        "⭐⭐⭐ (Rating 3)": len(cat_df[cat_df['Star Rating'] == 3])
    })

if summary_rows:
    top_5_table_df = pd.DataFrame(summary_rows)
    styled_top_5 = top_5_table_df.style.background_gradient(cmap='Blues', subset=['⭐ (Rating 1)', '⭐⭐ (Rating 2)', '⭐⭐⭐ (Rating 3)'])
    st.dataframe(styled_top_5, width='stretch', hide_index=True)
else:
    st.info("Belum ada kendala terkategori pada periode ini.")

blue_divider()
st.markdown("<br>", unsafe_allow_html=True)

# --- SECTION 3: TABEL RINCIAN KATEGORI SELURUH KOMPLAIN ---
st.markdown("<h3 style='color: #075985;'>📑 Rincian Kategori Seluruh Komplain</h3>", unsafe_allow_html=True)

all_complaints = complaints_df
category_counts = all_complaints['Kategori'].value_counts()
detail_rows = []

for cat in category_counts.index:
    if cat in ["Lainnya", "Spam / Tidak Jelas"]:
        continue
        
    count = category_counts[cat]
    isi_komplain = " ".join(categories_config[cat]['desc'])
        
    contoh = all_complaints[all_complaints['Kategori'] == cat]['Review Text'].dropna().iloc[0]
    
    display_cat = cat
    if "✨ Kategori baru dipelajari oleh AI." in isi_komplain:
        display_cat = f"✨ {cat}"
        
    detail_rows.append({
        "Jenis Komplain": display_cat,
        "Isi Komplain": isi_komplain,
        "Jumlah": count,
        "Contoh": f'"{contoh}"'
    })

if "Lainnya" in category_counts:
    count_lainnya = category_counts["Lainnya"]
    contoh_lainnya = all_complaints[all_complaints['Kategori'] == "Lainnya"]['Review Text'].dropna().iloc[0] if count_lainnya > 0 else "Tidak ada."
    detail_rows.append({
        "Jenis Komplain": "Komplain Lainnya (Belum Terkategori)",
        "Isi Komplain": "Kasus spesifik di luar kategori utama yang belum diproses AI.",
        "Jumlah": count_lainnya,
        "Contoh": f'"{contoh_lainnya}"'
    })

if "Spam / Tidak Jelas" in category_counts:
    count_spam = category_counts["Spam / Tidak Jelas"]
    contoh_spam = all_complaints[all_complaints['Kategori'] == "Spam / Tidak Jelas"]['Review Text'].dropna().iloc[0] if count_spam > 0 else "Tidak ada."
    detail_rows.append({
        "Jenis Komplain": "🗑️ Spam / Tidak Jelas",
        "Isi Komplain": "Ulasan yang tidak memiliki keluhan bermakna, spam, atau hanya makian.",
        "Jumlah": count_spam,
        "Contoh": f'"{contoh_spam}"'
    })

if detail_rows:
    detail_df = pd.DataFrame(detail_rows)
    st.table(detail_df)