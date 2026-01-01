import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import warnings
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Global Brand Colors (New Matching Gradient Palette)
KERATONIAN_COLORS = ['#845EC2', '#2C73D2', '#0081CF', '#0089BA', '#008E9B', '#00BF7A']
KERATONIAN_GRADIENT = [[0, '#845EC2'], [0.5, '#0089BA'], [1, '#00BF7A']]
GLOBAL_GRADIENT_CSS = "linear-gradient(90deg, #845EC2 0%, #00BF7A 100%)"

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="KERATONIAN Business Performance Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    /* Main Layout */
    .main {
        padding-top: 5px !important;
    }
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
    
    /* Premium Header */
    .premium-header {
        background: linear-gradient(90deg, #845EC2 0%, #00BF7A 100%);
        padding: 25px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title-container {
        display: flex;
        align-items: center;
    }
    .header-icon {
        font-size: 2.5rem;
        margin-right: 15px;
    }
    .header-text h1 {
        margin: 0;
        padding: 0;
        color: white;
        text-align: left;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .header-text p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .quarter-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 6px 15px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(5px);
    }
    
    /* Typography Overrides */
    h2, h3 {
        padding-top: 1rem !important;
        margin-top: 0 !important;
    }
    .stDivider {
        margin-top: 1rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Sidebar Styling */
    /* Sidebar Styling - Reverted to White Background */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #eee;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2C73D2 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 5px !important;
    }
    [data-testid="stSidebar"] label {
        color: #444 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Multiselect Chips (Eye-catching colors) */
    span[data-baseweb="tag"] {
        background-color: #845EC2 !important;
        color: white !important;
    }
    .sidebar-logo-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 85%;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Sidebar Widget Styling */
    div[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
    }
    
    /* Insight Cards Colors (Synced with Palette) */
    .insight-card[style*="border-left-color: #5D3A1A"] { border-left-color: #845EC2 !important; }
    .insight-card[style*="border-left-color: #A67C52"] { border-left-color: #2C73D2 !important; }
    .insight-card[style*="border-left-color: #8B5E3C"] { border-left-color: #0089BA !important; }
    .insight-card[style*="border-left-color: #4A3014"] { border-left-color: #00BF7A !important; }
    
    /* Welcome Setup Premium Revamp */
    .stApp {
        background-color: #ffffff;
    }
    .welcome-container {
        display: none; /* Hide the marker div */
    }
    /* Target the main container on the landing page specifically */
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) {
        background: white;
        padding: 0px;
        border-radius: 24px;
        box-shadow: 0 20px 50px rgba(132, 94, 194, 0.15);
        border: 1px solid #E0E0E0;
        margin-top: 30px;
        overflow: hidden; /* Ensure image doesn't bleed out */
    }
    .setup-logo {
        width: 150px;
        margin-bottom: 20px;
    }
    .setup-title {
        color: #845EC2;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 15px;
        letter-spacing: -1px;
    }
    .setup-subtitle {
        color: #2C73D2;
        font-size: 1.1rem;
        margin-bottom: 40px;
        opacity: 0.8;
    }
    .setup-input-label {
        color: #845EC2;
        font-weight: 700;
        text-align: left;
        margin-bottom: 10px;
        display: block;
    }
    .stMultiSelect, .stSelectbox {
        margin-bottom: 20px;
    }
    .enter-button button {
        background: linear-gradient(90deg, #845EC2 0%, #00BF7A 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 15px 40px !important;
        font-size: 1.2rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(132, 94, 194, 0.2) !important;
        transition: transform 0.3s ease !important;
    }
    .enter-button button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(0, 191, 122, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def format_rupiah(value):
    """Format angka ke format Rupiah (Rp) - versi Indonesia"""
    if value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} Miliar"
    elif value >= 1_000_000:
        return f"Rp {value/1_000_000:.1f} Juta"
    elif value >= 1_000:
        return f"Rp {value/1_000:.0f} Ribu"
    else:
        return f"Rp {value:,.0f}"

def format_axis_label(value):
    """Format angka untuk axis label di chart - tanpa 'Rp'"""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}M (Miliar)"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}J (Juta)"
    elif value >= 1_000:
        return f"{value/1_000:.0f}R (Ribu)"
    else:
        return f"{value:,.0f}"

# ============================================
# LOAD & CACHE DATA
# ============================================

@st.cache_data
def load_data(file_path):
    """Load dan preprocess data"""
    df = pd.read_csv(file_path, sep=';')
    
    # Parse dates - FINAL FIX: handle datetime with time component
    df['Tanggal Order'] = pd.to_datetime(df['Tanggal Order'], format='mixed', dayfirst=False)
    df = df.dropna(subset=['Tanggal Order'])
    
    # Extract time components and standardize to English for consistent ordering
    df['Tahun'] = df['Tanggal Order'].dt.year
    df['Bulan'] = df['Tanggal Order'].dt.month
    df['Bulan_Nama'] = df['Tanggal Order'].dt.strftime('%B')
    df['Kuartal'] = df['Tanggal Order'].dt.quarter
    df['Hari_Minggu'] = df['Tanggal Order'].dt.day_name()
    df['Hour'] = df['Tanggal Order'].dt.hour
    
    # Standardize Month Names to English if locale differs
    month_map = {
        'Januari': 'January', 'Februari': 'February', 'Maret': 'March', 'April': 'April',
        'Mei': 'May', 'Juni': 'June', 'Juli': 'July', 'Agustus': 'August',
        'September': 'September', 'Oktober': 'October', 'November': 'November', 'Desember': 'December'
    }
    df['Bulan_Nama'] = df['Bulan_Nama'].map(lambda x: month_map.get(x, x))
    
    # Standardize Day Names to English if locale differs
    day_map = {
        'Senin': 'Monday', 'Selasa': 'Tuesday', 'Rabu': 'Wednesday', 'Kamis': 'Thursday',
        'Jumat': 'Friday', 'Sabtu': 'Saturday', 'Minggu': 'Sunday'
    }
    df['Hari_Minggu'] = df['Hari_Minggu'].map(lambda x: day_map.get(x, x))
    
    return df

@st.cache_resource
def train_prediction_model(df):
    """Train Random Forest model based on notebook logic"""
    # Create copy for training
    train_df = df.copy()
    
    # Define Target: High Value (1) if > median, else Low Value (0)
    median_val = train_df['Total Penjualan'].median()
    train_df['Kategori_Transaksi'] = (train_df['Total Penjualan'] > median_val).astype(int)
    
    # Feature Engineering
    # Features: Qty, Harga Jual, Total Diskon, Provinsi, Channel, Pesanan
    features = ['Qty', 'Harga Jual', 'Total Diskon', 'Provinsi', 'Channel', 'Pesanan']
    X = train_df[features].copy()
    y = train_df['Kategori_Transaksi']
    
    # Encoding categorical features
    encoders = {}
    cat_cols = ['Provinsi', 'Channel', 'Pesanan']
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, encoders, acc, median_val

# Load data
try:
    df = load_data('Keratonian_anomalies_marked(CLEAN) (1)(Keratonian_anomalies_marked(CLE).csv')
except FileNotFoundError:
    st.error("❌ File 'Keratonian_anomalies_marked_CLEAN.csv' tidak ditemukan!")
    st.info("Pastikan file CSV ada di folder yang sama dengan script ini")
    st.stop()

# Train Prediction Model
model, encoders, model_acc, median_threshold = train_prediction_model(df)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'dashboard_ready' not in st.session_state:
    st.session_state.dashboard_ready = False
    st.session_state.selected_tahun = [sorted(df['Tahun'].unique())[-1]] # Default to latest year
    st.session_state.selected_quarter = None

# ============================================
# WELCOME/SETUP PAGE
# ============================================

# ============================================
# WELCOME/SETUP PAGE (PREMIUM REVAMP)
# ============================================

if not st.session_state.dashboard_ready:
    # Spacer at the top
    # Spacer at the top
    st.write("<br>", unsafe_allow_html=True)
    
    # Main Card Container with marker
    st.markdown("<div class='split-hero-card'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image("assets/image (1).png", use_container_width=True)
    
    with col2:
        st.markdown("<div style='padding: 30px 20px;'>", unsafe_allow_html=True)
        
        # Logo & Branding
        st.image("assets/logo_official.svg", width=200)
        st.markdown("<h1 class='setup-title' style='font-size: 2rem;'>BUSINESS INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.markdown("<p class='setup-subtitle' style='font-size: 1rem; margin-bottom: 20px;'>Siapkan periode data untuk eksplorasi performa bisnis Keratonian.</p>", unsafe_allow_html=True)
        
        st.markdown("<hr style='border: 0.5px solid #E5D3B3; margin-top: 15px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        # Setup form
        tahun_list = sorted(df['Tahun'].unique())
        
        st.markdown("<span class='setup-input-label'>📅 Pilih Tahun</span>", unsafe_allow_html=True)
        selected_tahun = st.multiselect(
            "Tahun", 
            tahun_list, 
            default=st.session_state.selected_tahun,
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        
        st.markdown("<span class='setup-input-label'>📊 Pilih Kuartal</span>", unsafe_allow_html=True)
        selected_quarter_option = st.selectbox(
            "Kuartal", 
            ["Semua (Full Year)", "Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Okt-Des)"],
            label_visibility="collapsed"
        )
        
        # Map choice
        if selected_quarter_option == "Semua (Full Year)":
            selected_quarter = None
        else:
            selected_quarter = int(selected_quarter_option[1])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enter Action
        st.markdown("<div class='enter-button'>", unsafe_allow_html=True)
        if st.button("🚀 MULAI ANALISIS SEKARANG"):
            if not selected_tahun:
                st.error("Silakan pilih minimal satu tahun!")
            else:
                st.session_state.selected_tahun = selected_tahun
                st.session_state.selected_quarter = selected_quarter
                st.session_state.dashboard_ready = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<p style='margin-top: 15px; color: #5D3A1A; font-size: 0.8rem; font-style: italic;'>Atau gunakan akses cepat:</p>", unsafe_allow_html=True)
        
        if st.button("🌟 FULL YEAR ANALYSIS", use_container_width=True):
            st.session_state.selected_tahun = tahun_list
            st.session_state.selected_quarter = None
            st.session_state.dashboard_ready = True
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True) # End Right Column Padding

# ============================================
# MAIN DASHBOARD (After Setup)
# ============================================

else:
    # Get selected values from session state
    selected_tahun = st.session_state.selected_tahun
    selected_quarter = st.session_state.selected_quarter
    
    # Prepare data based on selection
    dashboard_df = df[df['Tahun'].isin(selected_tahun)].copy()
    
    years_str = ", ".join(map(str, sorted(selected_tahun)))
    if selected_quarter is not None:
        dashboard_df = dashboard_df[dashboard_df['Kuartal'] == selected_quarter]
        quarter_label = f"Q{selected_quarter} ({years_str})"
    else:
        quarter_label = f"Full Year ({years_str})"
    
    # Back button and Logo in sidebar
    st.sidebar.image("assets/logo_official.svg", use_container_width=True)
    
    if st.sidebar.button("🔙 Kembali ke Setup", use_container_width=True):
        st.session_state.dashboard_ready = False
        st.rerun()
    
    st.sidebar.divider()
    
    # SIDEBAR FILTERS
    st.sidebar.title("🎛️ FILTER TAMBAHAN")
    
    # Channel Filter
    channels = sorted([c for c in dashboard_df['Channel'].unique() if c != 'Sample'])
    selected_channels = st.sidebar.multiselect(
        "📱 Pilih Channel:",
        channels,
        default=channels if len(channels) <= 5 else channels[:3]
    )
    
    # Category Filter
    categories = sorted(dashboard_df['Kategori'].unique())
    selected_categories = st.sidebar.multiselect(
        "🏷️ Pilih Kategori:",
        categories,
        default=categories
    )
    
    # Reset Button
    if st.sidebar.button("🔄 Reset Filter", use_container_width=True):
        st.rerun()
    
    # Apply Filters
    filtered_df = dashboard_df[
        (dashboard_df['Channel'].isin(selected_channels)) &
        (dashboard_df['Kategori'].isin(selected_categories))
    ].copy()
    
    st.sidebar.divider()
    st.sidebar.metric("📊 Data Ditampilkan", f"{len(filtered_df):,} transaksi")
    
    # ============================================
    # MAIN HEADER (PREMIUM REVAMP)
    # ============================================
    
    st.markdown(f"""
        <div class="premium-header">
            <div class="header-title-container">
                <div class="header-icon">📊</div>
                <div class="header-text">
                    <h1>KERATONIAN SALES DASHBOARD</h1>
                    <p>Business Insights & Analytics Platform</p>
                </div>
            </div>
            <div class="quarter-badge">
                📍 {quarter_label}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # KEY METRICS
    # ============================================
    
    if len(filtered_df) > 0:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            total_revenue = filtered_df['Total Penjualan'].sum()
            st.metric(
                "💰 Total Revenue",
                format_rupiah(total_revenue),
                delta=f"{len(filtered_df):,} transaksi"
            )
        
        with metric_col2:
            total_qty = filtered_df['Qty'].sum()
            avg_transaction = filtered_df['Total Penjualan'].mean()
            st.metric(
                "📦 Total Unit",
                f"{total_qty:,.0f}",
                delta=f"Avg: {format_rupiah(avg_transaction)}"
            )
        
        with metric_col3:
            unique_cust = filtered_df['Cust'].nunique()
            st.metric(
                "👥 Customers",
                f"{unique_cust:,}",
                delta=f"Channel: {len(selected_channels)}"
            )
        
        with metric_col4:
            avg_transaction = filtered_df['Total Penjualan'].mean()
            max_transaction = filtered_df['Total Penjualan'].max()
            st.metric(
                "📊 Avg Transaction",
                format_rupiah(avg_transaction),
                delta=f"Max: {format_rupiah(max_transaction)}"
            )
        
        st.divider()
        
        # ============================================
        # TABS
        # ============================================
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📋 Overview",
            "🏆 Best Seller",
            "⏰ Prime Time",
            "👥 Customers",
            "🗺️ Geographic",
            "🔮 Prediction",
            "📦 Stock Recommendation",
            "📥 Export"
        ])
        
        # ============================================
        # TAB 1: OVERVIEW
        # ============================================
        
        with tab1:
            st.subheader("📊 Sales Overview")
            
            # --- LOCAL CUSTOMER TYPE FILTER ---
            overview_cust_filter = st.radio(
                "👤 Tipe Pelanggan (Overview):",
                ["Semua", "End User", "Reseller"],
                horizontal=True,
                key="ov_cust_filter"
            )
            
            if overview_cust_filter != "Semua":
                ov_df = filtered_df[filtered_df['Kategori Channel'] == overview_cust_filter]
            else:
                ov_df = filtered_df.copy()
                
            if len(ov_df) == 0:
                st.warning("Tidak ada data untuk filter yang dipilih.")
                st.stop()

            # Using ov_df for overview metrics
            ov_metric_col1, ov_metric_col2, ov_metric_col3, ov_metric_col4 = st.columns(4)
            
            with ov_metric_col1:
                total_revenue = ov_df['Total Penjualan'].sum()
                st.metric("💰 Total Revenue", format_rupiah(total_revenue))
            
            with ov_metric_col2:
                total_qty = ov_df['Qty'].sum()
                st.metric("📦 Total Unit", f"{total_qty:,.0f}")
            
            with ov_metric_col3:
                unique_cust = ov_df['Cust'].nunique()
                st.metric("👥 Customers", f"{unique_cust:,}")
            
            with ov_metric_col4:
                st.metric("📦 Total Baris", f"{len(ov_df):,}")
            
            st.divider()
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("### Top 5 Produk by Quantity")
                top_5_qty = ov_df.groupby('Pesanan')['Qty'].sum().sort_values(ascending=False).head(5)
                fig_qty = px.bar(
                    x=top_5_qty.index,
                    y=top_5_qty.values,
                    labels={'x': 'Produk', 'y': 'Total Qty'},
                    color=top_5_qty.values,
                    color_continuous_scale=KERATONIAN_GRADIENT
                )
                fig_qty.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_qty, use_container_width=True)
                
            with col_chart2:
                st.write("### Revenue by Channel")
                rev_channel = ov_df.groupby('Channel')['Total Penjualan'].sum()
                fig_channel = px.pie(
                    values=rev_channel.values,
                    names=rev_channel.index,
                    hole=0.4,
                    color_discrete_sequence=KERATONIAN_COLORS
                )
                fig_channel.update_layout(height=350)
                st.plotly_chart(fig_channel, use_container_width=True)
            
            st.divider()
            
            # --- AI DATA INSIGHTS (REAMPED) ---
            st.write("### 🤖 Data Insights (AI Generated)")
            
            # Logic for Dynamic Insights
            top_channel = ov_df.groupby('Channel')['Total Penjualan'].sum().idxmax() if not ov_df.empty else "N/A"
            top_product = ov_df.groupby('Pesanan')['Qty'].sum().idxmax() if not ov_df.empty else "N/A"
            
            # Custom CSS for Eye-Catching Cards
            st.markdown("""
                <style>
                .insight-card {
                    background: #fdfdfd;
                    padding: 20px;
                    border-radius: 12px;
                    border-left: 5px solid #ff4b4b;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
                    margin-bottom: 15px;
                    height: 140px;
                }
                .insight-title {
                    font-size: 1.1rem;
                    font-weight: bold;
                    color: #31333F;
                    margin-bottom: 8px;
                    display: flex;
                    align-items: center;
                }
                .insight-icon {
                    font-size: 1.5rem;
                    margin-right: 10px;
                }
                .insight-text {
                    font-size: 0.9rem;
                    color: #555;
                    line-height: 1.4;
                }
                </style>
            """, unsafe_allow_html=True)
            
            ins_col1, ins_col2 = st.columns(2)
            
            with ins_col1:
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #5D3A1A;">
                    <div class="insight-title"><span class="insight-icon">📱</span> Dominasi Channel</div>
                    <div class="insight-text">
                        Channel <b>{top_channel}</b> adalah kontributor pendapatan utama. Disarankan untuk memperkuat strategi promosi khusus di platform ini.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #A67C52;">
                    <div class="insight-title"><span class="insight-icon">⭐</span> Star Product</div>
                    <div class="insight-text">
                        Produk <b>{top_product}</b> memiliki kecepatan penjualan (velocity) tertinggi. Pastikan ketersediaan stok di gudang tetap aman.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with ins_col2:
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #8B5E3C;">
                    <div class="insight-title"><span class="insight-icon">📦</span> Strategi Cross-Selling</div>
                    <div class="insight-text">
                        Analisis menunjukkan pembeli <i>Hampers</i> seringkali tertarik pada <i>Tatakan Dupa</i>. Sangat potensial untuk promo paket bundling.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #4A3014;">
                    <div class="insight-title"><span class="insight-icon">🔮</span> Predictor Status</div>
                    <div class="insight-text">
                        Model AI kami memiliki akurasi <b>{model_acc:.1%}</b> dalam memprediksi nilai transaksi. Gunakan tab <b>Prediction</b> untuk simulasi.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # --- DRILL DOWN / DIAGNOSTIC SECTION FROM 4_1.IPYNB ---
            with st.expander("🔍 Deep Dive: Analisis Korelasi & Faktor Penentu", expanded=False):
                st.markdown("Analisis korelasi membantu kita memahami hubungan antara variabel (misal: apakah diskon benar-benar meningkatkan kuantitas?).")
                
                # Correlation Data Calculation
                numeric_cols = ['Qty', 'Harga Jual', 'Total Diskon', 'Total Penjualan']
                corr_matrix = ov_df[numeric_cols].corr()
                
                # Plotly Heatmap
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    aspect="auto",
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    labels=dict(color="Correlation Score")
                )
                fig_corr.update_layout(
                    title="Matriks Korelasi (Diagnostic Analysis)",
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                st.info("""
                **Interpretasi Korelasi:**
                - **Dekat +1.0**: Hubungan searah yang kuat (misal: Qty naik, Penjualan pasti naik).
                - **Dekat -1.0**: Hubungan berlawanan arah (misal: Harga naik, Qty turun).
                - **Dekat 0.0**: Tidak ada hubungan signifikan antara variabel tersebut.
                """)
                
                st.divider()
                st.markdown("**Drill-Down: Perilaku Segmen Pelanggan**")
                
                # Cross-segmentation calculation (Top 5 Products vs Customer Type)
                seg_drill = ov_df.groupby(['Kategori Channel', 'Pesanan'])['Qty'].sum().reset_index()
                top_5_prods = ov_df.groupby('Pesanan')['Qty'].sum().nlargest(5).index
                seg_drill = seg_drill[seg_drill['Pesanan'].isin(top_5_prods)]
                
                fig_seg_drill = px.bar(
                    seg_drill,
                    x='Pesanan',
                    y='Qty',
                    color='Kategori Channel',
                    barmode='group',
                    title="Proporsi Produk Terlaris per Segmen Pelanggan",
                    color_discrete_sequence=KERATONIAN_COLORS
                )
                fig_seg_drill.update_layout(height=400)
                st.plotly_chart(fig_seg_drill, use_container_width=True)
                
                st.write("**📝 Ringkasan Karakteristik Data (Statistik Deskriptif):**")
                st.markdown(f"""
                Bagian ini menjelaskan profil data Anda secara keseluruhan untuk periode yang dipilih:
                - **Rata-rata Penjualan**: Transaksi rata-rata bernilai **{format_rupiah(ov_df['Total Penjualan'].mean())}**.
                - **Volume Terbanyak**: Sebagian besar transaksi memiliki kuantitas **{ov_df['Qty'].mode()[0]} unit**.
                - **Rentang Harga**: Produk Anda dijual mulai dari **{format_rupiah(ov_df['Harga Jual'].min())}** hingga **{format_rupiah(ov_df['Harga Jual'].max())}**.
                - **Total Data**: Analisis ini didasarkan pada **{len(ov_df):,} baris transaksi**.
                """)
                
                with st.expander("📊 Lihat Detail Teknis (Tabel Deskriptif)"):
                    st.dataframe(ov_df[numeric_cols].describe().T, use_container_width=True)
        
        # ============================================
        # TAB 2: BEST SELLER
        # ============================================
        
        with tab2:
            st.subheader("🏆 Best Seller Analysis")
            
            # --- LOCAL FILTERS ---
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                product_segment = st.radio(
                    "🔍 Pilih Segmentasi Produk:",
                    ["Semua", "DKK (Kerucut)", "DKS (Stick)"],
                    horizontal=True,
                    key="bs_prod_seg"
                )
            with col_filt2:
                bs_cust_filter = st.selectbox(
                    "👤 Tipe Pelanggan:",
                    ["Semua", "End User", "Reseller"],
                    key="bs_cust_filt"
                )
            
            # Apply local filters
            seller_df = filtered_df.copy()
            if product_segment == "DKK (Kerucut)":
                seller_df = seller_df[seller_df['Pesanan'].str.contains('DKK', case=False, na=False)]
            elif product_segment == "DKS (Stick)":
                seller_df = seller_df[seller_df['Pesanan'].str.contains('DKS', case=False, na=False)]
                
            if bs_cust_filter != "Semua":
                seller_df = seller_df[seller_df['Kategori Channel'] == bs_cust_filter]
            
            if len(seller_df) == 0:
                st.warning("Tidak ada data untuk filter yang dipilih.")
                st.stop()
            
            best_qty = seller_df.groupby('Pesanan').agg({
                'Qty': 'sum',
                'Total Penjualan': 'sum',
                'Pesanan': 'count'
            }).sort_values('Qty', ascending=False)
            best_qty.columns = ['Unit', 'Revenue', 'Transaksi']
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### By Quantity")
                fig, ax = plt.subplots(figsize=(10, 6))
                # Use primary brand color for bars: #845EC2
                best_qty['Unit'].head(8).plot(kind='bar', ax=ax, color='#845EC2')
                ax.set_title('Top 8 Produk', fontweight='bold')
                ax.set_ylabel('Unit Terjual')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig, use_container_width=True)
            
            with col_right:
                st.write("### By Revenue ")
                best_revenue = seller_df.groupby('Pesanan')['Total Penjualan'].sum().sort_values(ascending=False).head(8)
                
                # Buat Plotly chart dengan custom hover
                fig = go.Figure(data=[
                    go.Bar(
                        x=best_revenue.values,
                        y=best_revenue.index,
                        orientation='h',
                        # Update colorscale to brand gradient
                        marker=dict(color=best_revenue.values, colorscale=KERATONIAN_GRADIENT),
                        hovertemplate='<b>%{y}</b><br>Revenue: %{text}<extra></extra>',
                        text=[format_rupiah(v) for v in best_revenue.values]
                    )
                ])
                fig.update_layout(
                    title='Top 8 Produk by Revenue',
                    xaxis_title='Revenue (Rp)',
                    yaxis_title='Produk',
                    height=400,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Format revenue column for display
            best_qty_display = best_qty.copy()
            best_qty_display['Revenue'] = best_qty_display['Revenue'].apply(format_rupiah)
            st.dataframe(best_qty_display.head(10), use_container_width=True)
        
        # ============================================
        # TAB 3: PRIME TIME
        # ============================================
        
        with tab3:
            st.subheader("⏰ Prime Time Analysis")
            
            # --- INTERACTIVE MULTI-YEAR TREND (AS REQUESTED) ---
            st.write("### 📈 Tren Penjualan Interaktif (Multi-Tahun)")
            st.markdown("Bandingkan jumlah pesanan antar tahun untuk melihat pola prime time.")
            
            # Sub-filters within the tab for visualization
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                trend_cust_type = st.selectbox(
                    "👤 Pilih Tipe Pelanggan:",
                    ["End User", "Reseller"],
                    key="trend_cust_type"
                )
            with col_f2:
                trend_metric = st.radio(
                    "📊 Metrik:",
                    ["Jumlah Pesanan (Unit)", "Total Revenue (Rp)"],
                    horizontal=True,
                    key="trend_metric"
                )

            # Map months to Indonesian short names
            indo_months = {
                'January': 'Jan', 'February': 'Feb', 'March': 'Mar', 'April': 'Apr',
                'May': 'Mei', 'June': 'Jun', 'July': 'Jul', 'August': 'Agu',
                'September': 'Sep', 'October': 'Okt', 'November': 'Nov', 'December': 'Des'
            }

            # For multi-year we need data outside the current dashboard_df year filter
            # BUT we apply the other sidebar filters (Channel, Category)
            multi_year_df = df[
                (df['Channel'].isin(selected_channels)) &
                (df['Kategori'].isin(selected_categories)) &
                (df['Kategori Channel'] == trend_cust_type)
            ].copy()

            if len(multi_year_df) > 0:
                metric_col = 'Qty' if "Unit" in trend_metric else 'Total Penjualan'
                
                # Group by Year and month
                trend_data = multi_year_df.groupby(['Tahun', 'Bulan', 'Bulan_Nama']).agg({
                    metric_col: 'sum'
                }).reset_index()
                
                # Sort by month number
                trend_data = trend_data.sort_values(['Tahun', 'Bulan'])
                
                # Map to Indonesian
                trend_data['Bulan_Label'] = trend_data['Bulan_Nama'].map(indo_months)
                
                # Convert Tahun to string to remove comma in legend
                trend_data['Tahun'] = trend_data['Tahun'].astype(str)
                
                fig = px.line(
                    trend_data,
                    x='Bulan_Label',
                    y=metric_col,
                    color='Tahun',
                    markers=True,
                    labels={metric_col: 'Value', 'Bulan_Label': 'Bulan'},
                    title=f"Tren {trend_metric} - {trend_cust_type}",
                    color_discrete_sequence=KERATONIAN_COLORS
                )
                
                fig.update_layout(
                    height=500,
                    xaxis={'categoryorder':'array', 'categoryarray':['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']},
                    hovermode='x unified'
                )
                
                if "Revenue" in trend_metric:
                    fig.update_traces(hovertemplate='<b>%{x}</b><br>Revenue: Rp %{y:,.0f}<extra></extra>')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada data untuk kombinasi filter ini.")

            st.divider()
            
            # --- MULTI-YEAR DAILY COMPARISON ---
            st.write("### 📅 Penjualan Berdasarkan Hari (Perbandingan Tahunan)")
            st.markdown("Analisis hari teramai dalam seminggu untuk setiap tahun yang dipilih.")
            
            # Map days to Indonesian
            day_map = {
                'Monday': 'Senin', 'Tuesday': 'Selasa', 'Wednesday': 'Rabu',
                'Thursday': 'Kamis', 'Friday': 'Jumat', 'Saturday': 'Sabtu', 'Sunday': 'Minggu'
            }
            day_order_id = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
            
            daily_data = filtered_df.copy()
            daily_data['Hari_Label'] = daily_data['Hari_Minggu'].map(day_map)
            
            # Group by Year and Day
            weekly_comp = daily_data.groupby(['Tahun', 'Hari_Label'])['Total Penjualan'].sum().reset_index()
            
            # Convert Tahun to string to remove comma in legend
            weekly_comp['Tahun'] = weekly_comp['Tahun'].astype(str)
            
            fig_daily = px.bar(
                weekly_comp,
                x='Hari_Label',
                y='Total Penjualan',
                color='Tahun',
                barmode='group',
                title='Perbandingan Penjualan Per Hari',
                labels={'Hari_Label': 'Hari', 'Total Penjualan': 'Revenue (Rp)'},
                color_discrete_sequence=KERATONIAN_COLORS
            )
            
            fig_daily.update_layout(
                height=450,
                xaxis={'categoryorder':'array', 'categoryarray': day_order_id},
                hovermode='x unified'
            )
            fig_daily.update_traces(hovertemplate='<b>%{x}</b><br>Revenue: Rp %{y:,.0f}<extra></extra>')
            
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # ============================================
        # TAB 4: CUSTOMER ANALYSIS
        # ============================================
        
        with tab4:
            st.subheader("👥 Customer Segmentation")
            
            # --- CUSTOMER TYPE FILTER (AS REQUESTED) ---
            cust_filter = st.radio(
                "🎯 Filter Tipe Pelanggan:",
                ["Semua", "End-User", "Reseller"],
                horizontal=True,
                key="cust_seg_filter"
            )
            
            if cust_filter == "End-User":
                cust_df = filtered_df[filtered_df['Kategori Channel'] == 'End User']
            elif cust_filter == "Reseller":
                cust_df = filtered_df[filtered_df['Kategori Channel'] == 'Reseller']
            else:
                cust_df = filtered_df.copy()
            
            if len(cust_df) == 0:
                st.warning(f"Tidak ada data untuk kategori {cust_filter} pada periode ini.")
                st.stop()

            cust_analysis = cust_df.groupby('Cust').agg({
                'Pesanan': 'count',
                'Total Penjualan': 'sum',
                'Kategori Channel': 'first'
            }).sort_values('Total Penjualan', ascending=False)
            cust_analysis.columns = ['Frequency', 'Monetary', 'Tipe Pelanggan']
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### Top 10 by Frequency")
                st.caption("Menampilkan pelanggan yang paling sering berbelanja (berdasarkan jumlah transaksi).")
                top_freq = cust_analysis['Frequency'].nlargest(10).reset_index()
                
                fig_freq = px.bar(
                    top_freq,
                    x='Frequency',
                    y='Cust',
                    orientation='h',
                    color='Frequency',
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    text_auto=True
                )
                fig_freq.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_freq, use_container_width=True)
            
            with col_right:
                st.write("### Top 10 by Spending")
                st.caption("Menampilkan pelanggan dengan total pengeluaran/belanja terbesar (dalam Rupiah).")
                top_monetary = cust_analysis['Monetary'].nlargest(10).reset_index()
                
                fig_spent = px.bar(
                    top_monetary,
                    x='Monetary',
                    y='Cust',
                    orientation='h',
                    color='Monetary',
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    text_auto='.2s'
                )
                fig_spent.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_spent, use_container_width=True)
            
            st.divider()
            st.write("### Customer Type Distribution")
            cust_type = filtered_df.groupby('Kategori Channel')['Total Penjualan'].sum().reset_index()
            
            fig_type = px.bar(
                cust_type,
                x='Kategori Channel',
                y='Total Penjualan',
                color='Kategori Channel',
                color_discrete_sequence=KERATONIAN_COLORS,
                text_auto='.2s'
            )
            fig_type.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_type, use_container_width=True)
            
            st.write("### Detail Transaksi Customer")
            # Format monetary column for display
            cust_analysis_display = cust_analysis[['Tipe Pelanggan', 'Frequency', 'Monetary']].copy()
            cust_analysis_display['Monetary'] = cust_analysis_display['Monetary'].apply(format_rupiah)
            st.dataframe(cust_analysis_display.head(15), use_container_width=True)
        
        # ============================================
        # TAB 5: GEOGRAPHIC
        # ============================================
        
        with tab5:
            st.subheader("🗺️ Geographic Sales Distribution")
            
            # --- LOCAL CUSTOMER TYPE FILTER ---
            geo_cust_filter = st.radio(
                "👤 Tipe Pelanggan (Geographic)",
                ["Semua", "End User", "Reseller"],
                horizontal=True,
                key="geo_cust_filter"
            )
            
            geo_df = filtered_df.copy()
            if geo_cust_filter != "Semua":
                geo_df = geo_df[geo_df['Kategori Channel'] == geo_cust_filter]
                
            if len(geo_df) == 0:
                st.warning("Tidak ada data untuk filter yang dipilih.")
                st.stop()

            st.markdown("Distribusi pendapatan berdasarkan wilayah (Provinsi & Kabupaten/Kota).")
            
            col_geo1, col_geo2 = st.columns(2)
            
            with col_geo1:
                st.write("### Top 10 Provinces")
                by_province = geo_df.groupby('Provinsi')['Total Penjualan'].sum().sort_values(ascending=False).head(10).reset_index()
                
                fig_prov = px.bar(
                    by_province,
                    x='Total Penjualan',
                    y='Provinsi',
                    orientation='h',
                    color='Total Penjualan',
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    text_auto='.2s'
                )
                fig_prov.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_prov, use_container_width=True)
            
            with col_geo2:
                st.write("### Top 10 Districts")
                by_district = geo_df.groupby('Daerah')['Total Penjualan'].sum().sort_values(ascending=False).head(10).reset_index()
                
                fig_dist = px.bar(
                    by_district,
                    x='Total Penjualan',
                    y='Daerah',
                    orientation='h',
                    color='Total Penjualan',
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    text_auto='.2s'
                )
                fig_dist.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    height=450,
                    margin=dict(l=20, r=20, t=30, b=20),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # ============================================
        # TAB 6: PREDICTION
        # ============================================
        
        with tab6:
            st.subheader("🔮 Transaction Value Prediction")
            st.markdown(f"**Model Accuracy:** `{model_acc:.2%}`")
            st.markdown("Prediksi apakah transaksi akan masuk kategori **High Value** (> median) atau **Low Value**.")
            
            with st.form("prediction_form"):
                col_pred1, col_pred2 = st.columns(2)
                
                with col_pred1:
                    # Numeric Inputs
                    qty_input = st.number_input("Quantity", min_value=1, value=1)
                    price_input = st.number_input("Harga Jual", min_value=0, value=25000)
                    discount_input = st.number_input("Total Diskon", min_value=0, value=0)
                
                with col_pred2:
                    # Categorical Inputs
                    provinsi_opt = sorted(df['Provinsi'].astype(str).unique())
                    provinsi_input = st.selectbox("Provinsi", provinsi_opt)
                    
                    channel_opt = sorted(df['Channel'].astype(str).unique())
                    channel_input = st.selectbox("Channel", channel_opt)
                    
                    pesanan_opt = sorted(df['Pesanan'].astype(str).unique())
                    pesanan_input = st.selectbox("Product (Pesanan)", pesanan_opt)
                
                predict_btn = st.form_submit_button("🔮 Predict Value", use_container_width=True)
                
                if predict_btn:
                    try:
                        # Prepare input data
                        input_data = pd.DataFrame([{
                            'Qty': qty_input,
                            'Harga Jual': price_input,
                            'Total Diskon': discount_input,
                            'Provinsi': encoders['Provinsi'].transform([provinsi_input])[0],
                            'Channel': encoders['Channel'].transform([channel_input])[0],
                            'Pesanan': encoders['Pesanan'].transform([pesanan_input])[0]
                        }])
                        
                        # Prediction
                        prediction = model.predict(input_data)[0]
                        prob = model.predict_proba(input_data)[0][prediction]
                        
                        if prediction == 1:
                            st.success(f"### Result: 💎 High Value Transaction")
                            st.write(f"Confidence: {prob:.2%}")
                        else:
                            st.info(f"### Result: 📦 Low Value Transaction")
                            st.write(f"Confidence: {prob:.2%}")
                            
                        st.caption(f"Median Threshold: {format_rupiah(median_threshold)}")
                    except Exception as e:
                        st.caption(f"Median Threshold: {format_rupiah(median_threshold)}")
                    except Exception as e:
                        st.error(f"Error in prediction: {e}")

            st.divider()
            
            # --- FORECASTING SECTION ---
            st.subheader("📊 Analisis Tren & Forecasting (Beta)")
            st.info("Visualisasi tren penjualan kumulatif berdasarkan data historis, ditambah dengan proyeksi tren sederhana untuk 3 bulan ke depan.")
            
            # 1. Prepare Data for Time Series
            forecast_df = filtered_df.copy()
            # Ensure sorting by date
            forecast_df = forecast_df.sort_values('Tanggal Order')
            
            # Resample to Monthly Sales
            monthly_sales = forecast_df.set_index('Tanggal Order').resample('M')['Total Penjualan'].sum().reset_index()
            monthly_sales['Bulan'] = monthly_sales['Tanggal Order'].dt.strftime('%b %Y')
            monthly_sales['Type'] = 'Historical'
            
            if len(monthly_sales) > 1:
                # 2. Calculate Simple Moving Average (Trend) - 3 Month Window
                monthly_sales['MA_3'] = monthly_sales['Total Penjualan'].rolling(window=3, min_periods=1).mean()
                
                # 3. Generate Simple Forecast (Linear Projection based on last avg growth)
                last_date = monthly_sales['Tanggal Order'].iloc[-1]
                last_ma = monthly_sales['MA_3'].iloc[-1]
                
                # Calculate average monthly growth rate from the last 6 months (or less)
                if len(monthly_sales) >= 4:
                    recent = monthly_sales.tail(4)
                    start_val = recent['MA_3'].iloc[0]
                    end_val = recent['MA_3'].iloc[-1]
                    growth_rate = (end_val - start_val) / 3 # Simple slope
                else:
                    growth_rate = 0 # Flat forecast if not enough data
                
                future_dates = []
                future_vals = []
                
                for i in range(1, 4): # Forecast next 3 months
                    next_date = last_date + pd.DateOffset(months=i)
                    next_val = last_ma + (growth_rate * i)
                    future_dates.append(next_date)
                    future_vals.append(max(0, next_val)) # No negative sales directly
                
                forecast_data = pd.DataFrame({
                    'Tanggal Order': future_dates,
                    'Total Penjualan': future_vals,
                    'Type': 'Forecast (Est)'
                })
                forecast_data['Bulan'] = forecast_data['Tanggal Order'].dt.strftime('%b %Y')
                
                # Combine
                combined_forecast = pd.concat([monthly_sales, forecast_data], ignore_index=True)
                
                # Plot
                fig_forecast = px.line(
                    combined_forecast,
                    x='Bulan',
                    y='Total Penjualan',
                    color='Type',
                    markers=True,
                    title='Tren Penjualan Historis & Prediksi Terusan (Forecast 3 Bulan Depan)',
                    labels={'Total Penjualan': 'Sales Revenue', 'Bulan': 'Bulan'},
                    color_discrete_map={'Historical': '#845EC2', 'Forecast (Est)': '#00BF7A'}
                )
                
                # Add trendline styling
                fig_forecast.update_traces(line=dict(width=3))
                # Make forecast dashed
                fig_forecast.for_each_trace(
                    lambda trace: trace.update(line=dict(dash='dot')) if trace.name == 'Forecast (Est)' else trace.update(line=dict(dash='solid'))
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Narasi Forecast
                if growth_rate > 0:
                    st.success("📈 **Tren Positif:** Berdasarkan data terakhir, penjualan menunjukkan tren kenaikan rata-rata. Proyeksi 3 bulan ke depan optimis.")
                elif growth_rate < 0:
                    st.warning("📉 **Tren Menurun:** Terdeteksi penurunan pada rata-rata penjualan 3 bulan terakhir. Perlu strategi boosting untuk bulan depan.")
                else:
                    st.info("➡️ **Tren Stabil:** Penjualan relatif stabil. Pertahankan performa.")
                    
            else:
                st.warning("⚠️ Data belum cukup untuk melakukan forecasting (Minimal butuh > 1 bulan data).")


        # ============================================
        # TAB 7: STOCK RECOMMENDATION
        # ============================================
        
        with tab7:
            st.subheader("📦 REKOMENDASI AKSI GUDANG (ACTION PLAN)")
            
            # --- GLOSSARY / KETERANGAN ---
            with st.expander("ℹ️ Lihat Keterangan Pengelompokan", expanded=True):
                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.markdown("""
                    **🔥 High (Ramai)**
                    *   **Kriteria:** Penjualan > 20 unit.
                    *   **Aksi:** Stok melimpah, prioritaskan ketersediaan.
                    """)
                with col_g2:
                    st.markdown("""
                    **⚖️ Medium (Normal)**
                    *   **Kriteria:** Penjualan 11 - 20 unit.
                    *   **Aksi:** Stok standar, monitor berkala.
                    """)
                with col_g3:
                    st.markdown("""
                    **❄️ Low (Sepi)**
                    *   **Kriteria:** Penjualan ≤ 10 unit.
                    *   **Aksi:** Stok minimal, produksi sesuai pesanan.
                    """)
            
            st.divider()
            
            # --- LOCAL FILTERS ---
            col_st1, col_st2 = st.columns(2)
            with col_st1:
                product_segment_stock = st.radio(
                    "🔍 Pilih Segmentasi Produk:",
                    ["Semua", "DKK (Kerucut)", "DKS (Stick)"],
                    horizontal=True,
                    key="segment_stock"
                )
            with col_st2:
                stock_cust_filter = st.selectbox(
                    "👤 Tipe Pelanggan:",
                    ["Semua", "End User", "Reseller"],
                    key="stock_cust_filt"
                )
            
            # Apply local filters
            stock_df_filtered = filtered_df.copy()
            if product_segment_stock == "DKK (Kerucut)":
                stock_df_filtered = stock_df_filtered[stock_df_filtered['Pesanan'].str.contains('DKK', case=False, na=False)]
            elif product_segment_stock == "DKS (Stick)":
                stock_df_filtered = stock_df_filtered[stock_df_filtered['Pesanan'].str.contains('DKS', case=False, na=False)]
                
            if stock_cust_filter != "Semua":
                stock_df_filtered = stock_df_filtered[stock_df_filtered['Kategori Channel'] == stock_cust_filter]
            
            if len(stock_df_filtered) == 0:
                st.warning("Tidak ada data untuk filter yang dipilih.")
                st.stop()

            # 1. Hitung Qty per Produk
            stock_data = stock_df_filtered.groupby('Pesanan').agg({
                'Qty': 'sum'
            }).reset_index()
            
            # 2. Klasifikasi Kondisi Pasar (Logic from Notebook)
            # High (Ramai): Qty > 20
            # Medium (Normal): 10 < Qty <= 20
            # Low (Sepi): Qty <= 10
            
            def classify_market(qty):
                if qty > 20:
                    return "3. High (Ramai)"
                elif qty > 10:
                    return "2. Medium (Normal)"
                else:
                    return "1. Low (Sepi)"
            
            def get_recommendation(market):
                if market == "3. High (Ramai)":
                    return "URGENT! Stok Melimpah (>20)"
                elif market == "2. Medium (Normal)":
                    return "Siaga, Stok Standar (Max 20)"
                else:
                    return "Stok Minimal (Sesuai Pesanan)"
            
            stock_data['Kondisi Pasar'] = stock_data['Qty'].apply(classify_market)
            stock_data['REKOMENDASI GUDANG'] = stock_data['Kondisi Pasar'].apply(get_recommendation)
            
            # 3. Sort by condition
            stock_data = stock_data.sort_values('Qty', ascending=False)
            
            # Display metrics for summary
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("🔥 High Demand", len(stock_data[stock_data['Kondisi Pasar'] == "3. High (Ramai)"]))
            with m_col2:
                st.metric("⚖️ Normal Demand", len(stock_data[stock_data['Kondisi Pasar'] == "2. Medium (Normal)"]))
            with m_col3:
                st.metric("❄️ Low Demand", len(stock_data[stock_data['Kondisi Pasar'] == "1. Low (Sepi)"]))
            
            st.divider()
            
            # Styling the dataframe (Updated with Brand Colors)
            def color_market(val):
                if "High" in val: color = '#845EC2' # Purple (Primary)
                elif "Medium" in val: color = '#2C73D2' # Blue
                else: color = '#00BF7A' # Teal
                return f'background-color: {color}; color: white; font-weight: bold'

            def color_rec(val):
                if "URGENT" in val: color = '#845EC2'
                elif "Siaga" in val: color = '#2C73D2'
                else: color = '#00BF7A'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                stock_data.style.applymap(color_market, subset=['Kondisi Pasar'])
                .applymap(color_rec, subset=['REKOMENDASI GUDANG']),
                use_container_width=True,
                hide_index=True
            )

        # ============================================
        # TAB 8: EXPORT
        # ============================================
        
        with tab8:
            st.subheader("📥 Export Data")
            
            st.write("Download hasil analisis dalam format Excel atau CSV:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Excel Export
                st.write("**📊 Export ke Excel**")
                output = io.BytesIO()
                
                export_sheets = {
                    'Summary': pd.DataFrame({
                        'Metric': ['Total Revenue', 'Total Units', 'Total Transactions', 'Unique Customers'],
                        'Value': [format_rupiah(filtered_df['Total Penjualan'].sum()), 
                                 f"{filtered_df['Qty'].sum():,}", 
                                 f"{len(filtered_df):,}",
                                 f"{filtered_df['Cust'].nunique():,}"]
                    }),
                    'Best_Seller': filtered_df.groupby('Pesanan').agg({'Qty': 'sum', 'Total Penjualan': 'sum'}).sort_values('Qty', ascending=False),
                    'Top_Customers': filtered_df.groupby('Cust').agg({'Pesanan': 'count', 'Total Penjualan': 'sum'}).sort_values('Total Penjualan', ascending=False),
                    'By_Province': filtered_df.groupby('Provinsi').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}).sort_values('Total Penjualan', ascending=False),
                    'By_Channel': filtered_df.groupby('Channel').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}),
                }
                
                try:
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        for sheet_name, df_sheet in export_sheets.items():
                            df_sheet.to_excel(writer, sheet_name=sheet_name[:31])
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="⬇️ Download Excel",
                        data=excel_data,
                        file_name=f"Dashboard_Keratonian_{selected_tahun}_{quarter_label.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                except OSError as e:
                    if e.errno == 28: # No space left on device
                        st.error("❌ Gagal membuat Excel: Penyimpanan Penuh (Disk Full).")
                        st.info("⚠️ Silakan gunakan opsi **Download CSV** di sebelah kanan, karena file CSV jauh lebih ringan dan tidak memerlukan ruang temporary yang besar.")
                    else:
                        st.error(f"❌ Gagal membuat Excel (IO Error): {e}")
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat export Excel: {e}")
            
            with col2:
                # CSV Export
                st.write("**📄 Export ke CSV**")
                csv_data = filtered_df[['Tanggal Order', 'Channel', 'Cust', 'Pesanan', 'Qty', 'Total Penjualan', 'Provinsi']].to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"Dashboard_Keratonian_{selected_tahun}_{quarter_label.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.divider()
            st.write("### Preview Data (100 rows)")
            
            # Format revenue and date for preview
            preview_df = filtered_df[['Tanggal Order', 'Channel', 'Cust', 'Pesanan', 'Qty', 'Total Penjualan', 'Provinsi', 'Daerah']].head(100).copy()
            preview_df['Tanggal Order'] = pd.to_datetime(preview_df['Tanggal Order']).dt.date
            preview_df['Total Penjualan'] = preview_df['Total Penjualan'].apply(format_rupiah)
            st.dataframe(preview_df, use_container_width=True)
    
    else:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Coba ubah filter!")
    
    # ============================================
    # FOOTER
    # ============================================
    
    st.divider()
    st.markdown("""
        <div style='text-align: center; color: gray; font-size: 11px; margin-top: 2rem;'>
        <p>Dashboard Keratonian © 2025 | Interactive Sales Analytics</p>
        </div>
        """, unsafe_allow_html=True)
