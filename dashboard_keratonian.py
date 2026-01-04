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
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Global Brand Colors (New Brown Gradient Palette)
# Palette: Cinnamon, Brown, Tortilla, Russet, Coffee, Brunette
KERATONIAN_COLORS = ['#622A0F', '#7C4700', '#997950', '#7F461B', '#4B3619', '#3A1F04', '#795C32', '#5C2C06']
KERATONIAN_GRADIENT = [[0, '#3A1F04'], [0.5, '#7C4700'], [1, '#997950']]
GLOBAL_GRADIENT_CSS = "linear-gradient(90deg, #622A0F 0%, #997950 100%)"

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
        background: linear-gradient(90deg, #622A0F 0%, #997950 100%);
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
        color: #7C4700 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        margin-bottom: 5px !important;
    }
    [data-testid="stSidebar"] label {
        color: #444 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Multiselect Chips (Diverse Brown Palette) */
    span[data-baseweb="tag"]:nth-of-type(4n+1) { background-color: #622A0F !important; color: white !important; }
    span[data-baseweb="tag"]:nth-of-type(4n+2) { background-color: #7C4700 !important; color: white !important; }
    span[data-baseweb="tag"]:nth-of-type(4n+3) { background-color: #997950 !important; color: white !important; }
    span[data-baseweb="tag"]:nth-of-type(4n+4) { background-color: #5C2C06 !important; color: white !important; }
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
    .insight-card[style*="border-left-color: #5D3A1A"] { border-left-color: #622A0F !important; }
    .insight-card[style*="border-left-color: #A67C52"] { border-left-color: #997950 !important; }
    .insight-card[style*="border-left-color: #8B5E3C"] { border-left-color: #7C4700 !important; }
    .insight-card[style*="border-left-color: #4A3014"] { border-left-color: #3A1F04 !important; }
    
    /* Welcome Setup Premium Revamp */
    .stApp {
        background-color: #ffffff;
    }
    .welcome-container {
        display: none;
    }
    
    /* Main Welcome Card */
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) {
        background: white;
        padding: 0px !important;
        margin: 0px !important;
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(98, 42, 15, 0.12);
        border: 1px solid #E0E0E0;
        margin-top: 20px !important;
        overflow: hidden;
    }
    
    /* AGGRESSIVE: Remove ALL padding/margin from columns */
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Target EVERY nested element in left column */
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:first-child,
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:first-child *,
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:first-child > div,
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:first-child > div > div,
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:first-child > div > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Image Container - Full Height with Flex */
    .hero-image-column {
        padding: 0 !important;
        margin: 0 !important;
        line-height: 0 !important;
        height: 100% !important;
        display: flex !important;
        width: 100% !important;
    }
    .hero-image-column img {
        object-fit: cover !important;
        width: 100% !important;
        height: 100% !important;
        min-height: 650px !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        flex: 1 !important;
    }
    
    /* Right Column - Form Side */
    div[data-testid="stVerticalBlock"]:has(> div > div > .split-hero-card) div[data-testid="stColumn"]:last-child {
        background: white;
        display: flex;
        align-items: center;
        padding: 0 !important;
    }
    
    /* Form Container */
    .form-container {
        padding: 40px 35px;
        width: 100%;
    }
    .setup-logo {
        width: 150px;
        margin-bottom: 20px;
    }
    .setup-title {
        color: #622A0F;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 15px;
        letter-spacing: -1px;
    }
    .setup-subtitle {
        color: #333333;
        font-size: 1.1rem;
        margin-bottom: 40px;
        opacity: 0.9;
    }
    .setup-input-label {
        color: #622A0F;
        font-weight: 700;
        text-align: left;
        margin-bottom: 10px;
        display: block;
    }
    .stMultiSelect, .stSelectbox {
        margin-bottom: 20px;
    }
    .enter-button button {
        background: linear-gradient(90deg, #622A0F 0%, #997950 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 15px 40px !important;
        font-size: 1.2rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 10px 20px rgba(98, 42, 15, 0.2) !important;
        transition: transform 0.3s ease !important;
    }
    .enter-button button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(153, 121, 80, 0.3) !important;
    }
    
    /* Tab Styling - Brown Active Indicator */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom-color: #622A0F !important;
        color: #622A0F !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: #7C4700 !important;
    }
    
    /* Override Streamlit Red Accents to Brown */
    .stAlert, [data-testid="stNotification"] {
        border-left-color: #622A0F !important;
    }
    a, a:hover {
        color: #622A0F !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================


def format_rupiah(value):
    """Format angka ke format Rupiah (Rp) - versi Indonesia dengan singkatan konsisten"""
    if value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} M"  # M = Miliar
    elif value >= 1_000_000:
        return f"Rp {value/1_000_000:.1f} Jt"  # Jt = Juta
    elif value >= 1_000:
        return f"Rp {value/1_000:.0f} Rb"  # Rb = Ribu
    else:
        return f"Rp {value:,.0f}"

def format_axis_label(value):
    """Format angka untuk axis label di chart - tanpa 'Rp' tapi dengan keterangan"""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f} M"  # M = Miliar
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f} Jt"  # Jt = Juta
    elif value >= 1_000:
        return f"{value/1_000:.0f} Rb"  # Rb = Ribu
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
    """Train Random Forest models for Transaction Value and Churn Risk"""
    train_df = df.copy()
    
    # 1. VALUE PREDICTION MODEL (Existing)
    median_val = train_df['Total Penjualan'].median()
    train_df['Kategori_Transaksi'] = (train_df['Total Penjualan'] > median_val).astype(int)
    
    # 2. CHURN PREDICTION MODEL (New Logic)
    # Define Churn: Customer has not ordered in the last 6 months (relative to dataset max date)
    max_date = train_df['Tanggal Order'].max()
    cust_last_date = train_df.groupby('Cust')['Tanggal Order'].transform('max')
    # Use 180 days as threshold for churn
    train_df['Days_Since_Last'] = (max_date - cust_last_date).dt.days
    train_df['is_churn'] = (train_df['Days_Since_Last'] > 180).astype(int)
    
    # Feature Engineering
    features = ['Qty', 'Harga Jual', 'Total Diskon', 'Provinsi', 'Channel', 'Pesanan']
    X = train_df[features].copy()
    y_value = train_df['Kategori_Transaksi']
    y_churn = train_df['is_churn']
    
    # Encoding categorical features
    encoders = {}
    cat_cols = ['Provinsi', 'Channel', 'Pesanan']
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    # Split data for Value Model
    X_train_v, X_test_v, y_train_v, y_test_v = train_test_split(X, y_value, test_size=0.2, random_state=42)
    model_value = RandomForestClassifier(n_estimators=100, random_state=42)
    model_value.fit(X_train_v, y_train_v)
    acc_value = accuracy_score(y_test_v, model_value.predict(X_test_v))
    
    # Split data for Churn Model
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_churn, test_size=0.2, random_state=42)
    model_churn = RandomForestClassifier(n_estimators=100, random_state=42)
    model_churn.fit(X_train_c, y_train_c)
    acc_churn = accuracy_score(y_test_c, model_churn.predict(X_test_c))
    
    return {
        'value_model': model_value,
        'churn_model': model_churn,
        'encoders': encoders,
        'acc_value': acc_value,
        'acc_churn': acc_churn,
        'median_threshold': median_val
    }

# Load data
try:
    df = load_data('Keratonian_anomalies_marked(CLEAN) (1)(Keratonian_anomalies_marked(CLE).csv')
except FileNotFoundError:
    st.error("❌ File 'Keratonian_anomalies_marked_CLEAN.csv' tidak ditemukan!")
    st.info("Pastikan file CSV ada di folder yang sama dengan script ini")
    st.stop()

# Train Prediction Models
trained_assets = train_prediction_model(df)
model = trained_assets['value_model']
churn_model = trained_assets['churn_model']
encoders = trained_assets['encoders']
model_acc = trained_assets['acc_value']
churn_acc = trained_assets['acc_churn']
median_threshold = trained_assets['median_threshold']

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
    # Main Card Container with marker
    st.markdown("<div class='split-hero-card'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='hero-image-column' style='padding:0!important;margin:0!important;line-height:0;height:100%;display:flex;width:100%;'>", unsafe_allow_html=True)
        st.image("assets/keratonian.jpeg", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        
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
            
        st.markdown("</div>", unsafe_allow_html=True) # End form-container

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
    st.sidebar.title("FILTER TAMBAHAN")
    
    # Channel Filter
    channels = sorted([c for c in dashboard_df['Channel'].unique() if c != 'Sample'])
    
    # Initialize session state for filters if not exists
    if 'chan_multiselect' not in st.session_state:
        st.session_state.chan_multiselect = channels if len(channels) <= 5 else channels[:3]
    if 'cat_multiselect' not in st.session_state:
        st.session_state.cat_multiselect = categories if 'categories' in locals() else sorted(dashboard_df['Kategori'].unique())

    # Select All Channel logic
    if st.sidebar.button("✅ Select All Channel", key="sel_all_chan"):
        st.session_state.chan_multiselect = channels
        st.rerun()

    selected_channels = st.sidebar.multiselect(
        "📱 Pilih Channel:",
        channels,
        key="chan_multiselect"
    )
    
    # Category Filter
    categories = sorted(dashboard_df['Kategori'].unique())
    
    # Select All Category logic
    if st.sidebar.button("✅ Select All Kategori", key="sel_all_cat"):
        st.session_state.cat_multiselect = categories
        st.rerun()

    selected_categories = st.sidebar.multiselect(
        "🏷️ Pilih Kategori:",
        categories,
        key="cat_multiselect"
    )
    
    # Reset Button
    if st.sidebar.button(" Reset Filter", use_container_width=True):
        st.rerun()
    
    # Apply Filters
    filtered_df = dashboard_df[
        (dashboard_df['Channel'].isin(selected_channels)) &
        (dashboard_df['Kategori'].isin(selected_categories))
    ].copy()
    
    st.sidebar.divider()
    st.sidebar.metric("Data Ditampilkan", f"{len(filtered_df):,} transaksi")
    
    # Website Button - Premium Design
    st.sidebar.markdown("""
        <style>
        .website-container {
            margin-top: 20px;
            margin-bottom: 10px;
        }
        .website-button {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 16px 24px;
            background: linear-gradient(135deg, #622A0F 0%, #7C4700 50%, #997950 100%);
            color: white !important;
            text-decoration: none !important;
            border-radius: 12px;
            font-weight: 700;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(98, 42, 15, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        .website-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        .website-button:hover::before {
            left: 100%;
        }
        .website-button:hover {
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 25px rgba(98, 42, 15, 0.4);
            text-decoration: none;
            color: white !important;
        }
        .website-button:active {
            transform: translateY(-1px) scale(0.98);
        }
        .website-icon {
            font-size: 1.3rem;
            filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
        }
        .website-text {
            font-size: 0.95rem;
            text-shadow: 0 1px 2px rgba(0,0,0,0.2);
        }
        </style>
        <div class="website-container">
            <a href="https://keratonian-scent.com/" target="_blank" class="website-button">
                <span class="website-icon">🌐</span>
                <span class="website-text">Visit Keratonian Website</span>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # MAIN HEADER (PREMIUM REVAMP)
    # ============================================
    
    st.markdown(f"""
        <div class="premium-header">
            <div class="header-title-container">
                <div class="header-icon"></div>
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
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📋 Overview",
            "🏆 Best Seller",
            "⏰ Prime Time",
            "👥 Customers",
            "🗺️ Geographic",
            "🔮 Value Prediction",
            "🚨 Churn Analysis",
            "📦 Stock Recommendation",
            "📥 Export"
        ])
        
        # ============================================
        # TAB 1: OVERVIEW
        # ============================================
        
        with tab1:
            st.subheader("Sales Overview")
            
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
            st.write("### 🤖 Data Insights")
            
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
                
                # Use varied colors from palette for better differentiation
                # Ensure we have enough colors for the bars
                bar_count = len(best_qty['Unit'].head(8))
                bar_colors = (KERATONIAN_COLORS * 2)[:bar_count]
                
                best_qty['Unit'].head(8).plot(kind='bar', ax=ax, color=bar_colors)
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
            
            st.divider()
            
            # --- MARKETING STRATEGIES (New Section) ---
            st.subheader("💡 Marketing & Sales Strategies")
            
            # Extract top overall product
            if not best_qty.empty:
                top_p_name = best_qty.index[0]
                top_p_qty = best_qty.iloc[0]['Unit']
                top_p_rev = best_qty.iloc[0]['Revenue']
                
                strat_col1, strat_col2 = st.columns(2)
                
                with strat_col1:
                    st.info(f"**Main Focus: {top_p_name}**")
                    st.markdown(f"""
                    Produk ini adalah kontributor utama dengan **{top_p_qty:,} unit** terjual.
                    
                    **Action Plan**
                    *   **Scale Up:** Tingkatkan anggaran iklan Meta/Google untuk produk ini.
                    *   **Bundling:** Buat paket hemat dengan produk 'Slow Moving' untuk cuci gudang.
                    *   **Retention:** Berikan voucher diskon khusus untuk pembelian berikutnya pada item ini.
                    """)
                
                with strat_col2:
                    st.success("**Revenue Optimization Strategy**")
                    st.markdown(f"""
                    Total revenue dari best seller mencapai **{format_rupiah(top_p_rev)}**.
                    
                    **Recommendations**
                    *   **Subscription Model:** Jika memungkinkan, buat opsi berlangganan bulanan.
                    *   **Upselling:** Tawarkan varian premium atau ukuran lebih besar saat checkout.
                    *   **Cross-Selling:** Analisis data menunjukkan item pendukung (seperti Tatakan) dapat meningkatkan AOV (Average Order Value).
                    """)
            else:
                st.info("Pilih filter untuk melihat strategi marketing yang relevan.")
        
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
                (df['Tahun'].isin(selected_tahun)) &
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
                    color_discrete_sequence=['#622A0F', '#997950', '#7C4700', '#5C2C06', '#7F461B']
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
                color_discrete_sequence=['#622A0F', '#997950', '#7C4700']
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
                color_discrete_sequence=['#622A0F', '#997950'],
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
            
            # --- INTERACTIVE INDONESIA MAP ---
            st.write("### 🗺️ Peta Distribusi Penjualan Indonesia")
            st.caption("Visualisasi geografis intensitas penjualan per provinsi. Ukuran bubble = revenue lebih tinggi.")
            
            # Prepare province-level data
            province_sales = geo_df.groupby('Provinsi')['Total Penjualan'].sum().reset_index()
            province_sales = province_sales.sort_values('Total Penjualan', ascending=False)
            
            # Manual coordinate mapping for major Indonesian provinces (with variations)
            province_coords = {
                # Jakarta variations
                'DKI Jakarta': {'lat': -6.2088, 'lon': 106.8456},
                'Jakarta': {'lat': -6.2088, 'lon': 106.8456},
                'DKI JAKARTA': {'lat': -6.2088, 'lon': 106.8456},
                
                # Jawa Barat variations
                'Jawa Barat': {'lat': -6.9175, 'lon': 107.6191},
                'JAWA BARAT': {'lat': -6.9175, 'lon': 107.6191},
                
                # Jawa Tengah variations
                'Jawa Tengah': {'lat': -7.1508, 'lon': 110.1403},
                'JAWA TENGAH': {'lat': -7.1508, 'lon': 110.1403},
                
                # Jawa Timur variations
                'Jawa Timur': {'lat': -7.5361, 'lon': 112.2384},
                'JAWA TIMUR': {'lat': -7.5361, 'lon': 112.2384},
                
                # Banten variations
                'Banten': {'lat': -6.4058, 'lon': 106.0640},
                'BANTEN': {'lat': -6.4058, 'lon': 106.0640},
                
                # Yogyakarta variations
                'DI Yogyakarta': {'lat': -7.7956, 'lon': 110.3695},
                'Daerah Istimewa Yogyakarta': {'lat': -7.7956, 'lon': 110.3695},
                'Yogyakarta': {'lat': -7.7956, 'lon': 110.3695},
                'YOGYAKARTA': {'lat': -7.7956, 'lon': 110.3695},
                'D.I. Yogyakarta': {'lat': -7.7956, 'lon': 110.3695},
                'D.I YOGYAKARTA': {'lat': -7.7956, 'lon': 110.3695},
                
                # Bali variations
                'Bali': {'lat': -8.3405, 'lon': 115.0920},
                'BALI': {'lat': -8.3405, 'lon': 115.0920},
                
                # Sumatera Utara variations
                'Sumatera Utara': {'lat': 3.5952, 'lon': 98.6722},
                'SUMATERA UTARA': {'lat': 3.5952, 'lon': 98.6722},
                'Sumatra Utara': {'lat': 3.5952, 'lon': 98.6722},
                
                # Sumatera Barat variations
                'Sumatera Barat': {'lat': -0.7399, 'lon': 100.8000},
                'SUMATERA BARAT': {'lat': -0.7399, 'lon': 100.8000},
                'Sumatra Barat': {'lat': -0.7399, 'lon': 100.8000},
                
                # Sumatera Selatan variations
                'Sumatera Selatan': {'lat': -3.3194, 'lon': 104.9147},
                'SUMATERA SELATAN': {'lat': -3.3194, 'lon': 104.9147},
                'Sumatra Selatan': {'lat': -3.3194, 'lon': 104.9147},
                
                # Lampung variations
                'Lampung': {'lat': -5.4500, 'lon': 105.2667},
                'LAMPUNG': {'lat': -5.4500, 'lon': 105.2667},
                
                # Riau variations
                'Riau': {'lat': 0.2933, 'lon': 101.7068},
                'RIAU': {'lat': 0.2933, 'lon': 101.7068},
                
                # Jambi variations
                'Jambi': {'lat': -1.6101, 'lon': 103.6131},
                'JAMBI': {'lat': -1.6101, 'lon': 103.6131},
                
                # Bengkulu variations
                'Bengkulu': {'lat': -3.7928, 'lon': 102.2608},
                'BENGKULU': {'lat': -3.7928, 'lon': 102.2608},
                
                # Aceh variations
                'Aceh': {'lat': 4.6951, 'lon': 96.7494},
                'ACEH': {'lat': 4.6951, 'lon': 96.7494},
                'Nanggroe Aceh Darussalam': {'lat': 4.6951, 'lon': 96.7494},
                'NANGGROE ACEH DARUSSALAM (NAD)': {'lat': 4.6951, 'lon': 96.7494},
                'NAD': {'lat': 4.6951, 'lon': 96.7494},
                
                # Kepulauan Riau variations
                'Kepulauan Riau': {'lat': 3.9457, 'lon': 108.1429},
                'Kep. Riau': {'lat': 3.9457, 'lon': 108.1429},
                'KEPULAUAN RIAU': {'lat': 3.9457, 'lon': 108.1429},
                
                # Bangka Belitung variations
                'Kepulauan Bangka Belitung': {'lat': -2.7411, 'lon': 106.4406},
                'Kep. Bangka Belitung': {'lat': -2.7411, 'lon': 106.4406},
                'Bangka Belitung': {'lat': -2.7411, 'lon': 106.4406},
                'BANGKA BELITUNG': {'lat': -2.7411, 'lon': 106.4406},
                
                # Kalimantan variations
                'Kalimantan Barat': {'lat': -0.0263, 'lon': 109.3425},
                'KALIMANTAN BARAT': {'lat': -0.0263, 'lon': 109.3425},
                'Kalimantan Tengah': {'lat': -1.6815, 'lon': 113.3824},
                'KALIMANTAN TENGAH': {'lat': -1.6815, 'lon': 113.3824},
                'Kalimantan Selatan': {'lat': -3.0926, 'lon': 115.2838},
                'KALIMANTAN SELATAN': {'lat': -3.0926, 'lon': 115.2838},
                'Kalimantan Timur': {'lat': 0.5387, 'lon': 116.4194},
                'KALIMANTAN TIMUR': {'lat': 0.5387, 'lon': 116.4194},
                'Kalimantan Utara': {'lat': 3.0731, 'lon': 116.0413},
                'KALIMANTAN UTARA': {'lat': 3.0731, 'lon': 116.0413},
                
                # Sulawesi variations
                'Sulawesi Utara': {'lat': 0.6246, 'lon': 123.9750},
                'SULAWESI UTARA': {'lat': 0.6246, 'lon': 123.9750},
                'Sulawesi Tengah': {'lat': -1.4300, 'lon': 121.4456},
                'SULAWESI TENGAH': {'lat': -1.4300, 'lon': 121.4456},
                'Sulawesi Selatan': {'lat': -3.6687, 'lon': 119.9740},
                'SULAWESI SELATAN': {'lat': -3.6687, 'lon': 119.9740},
                'Sulawesi Tenggara': {'lat': -4.1448, 'lon': 122.1747},
                'SULAWESI TENGGARA': {'lat': -4.1448, 'lon': 122.1747},
                'Sulawesi Barat': {'lat': -2.8441, 'lon': 119.2320},
                'SULAWESI BARAT': {'lat': -2.8441, 'lon': 119.2320},
                
                # Gorontalo variations
                'Gorontalo': {'lat': 0.6999, 'lon': 122.4467},
                'GORONTALO': {'lat': 0.6999, 'lon': 122.4467},
                
                # Maluku variations
                'Maluku': {'lat': -3.2385, 'lon': 130.1453},
                'MALUKU': {'lat': -3.2385, 'lon': 130.1453},
                'Maluku Utara': {'lat': 1.5709, 'lon': 127.8087},
                'MALUKU UTARA': {'lat': 1.5709, 'lon': 127.8087},
                
                # Papua variations
                'Papua': {'lat': -4.2699, 'lon': 138.0804},
                'PAPUA': {'lat': -4.2699, 'lon': 138.0804},
                'Papua Barat': {'lat': -1.3361, 'lon': 133.1747},
                'PAPUA BARAT': {'lat': -1.3361, 'lon': 133.1747},
                
                # Nusa Tenggara variations
                'Nusa Tenggara Barat': {'lat': -8.6529, 'lon': 117.3616},
                'NUSA TENGGARA BARAT': {'lat': -8.6529, 'lon': 117.3616},
                'NUSA TENGGARA BARAT (NTB)': {'lat': -8.6529, 'lon': 117.3616},
                'NTB': {'lat': -8.6529, 'lon': 117.3616},
                'Nusa Tenggara Timur': {'lat': -8.6574, 'lon': 121.0794},
                'NUSA TENGGARA TIMUR': {'lat': -8.6574, 'lon': 121.0794},
                'NUSA TENGGARA TIMUR (NTT)': {'lat': -8.6574, 'lon': 121.0794},
                'NTT': {'lat': -8.6574, 'lon': 121.0794}
            }
            
            # Add coordinates to province_sales
            province_sales['lat'] = province_sales['Provinsi'].map(lambda x: province_coords.get(x, {}).get('lat'))
            province_sales['lon'] = province_sales['Provinsi'].map(lambda x: province_coords.get(x, {}).get('lon'))
            
            # Filter out provinces without coordinates
            province_sales_mapped = province_sales.dropna(subset=['lat', 'lon'])
            unmapped_provinces = province_sales[province_sales['lat'].isna()]['Provinsi'].tolist()
            
            if len(province_sales_mapped) > 0:
                # Create scatter geo map
                fig_map = px.scatter_geo(
                    province_sales_mapped,
                    lat='lat',
                    lon='lon',
                    size='Total Penjualan',
                    hover_name='Provinsi',
                    hover_data={'Total Penjualan': ':,.0f', 'lat': False, 'lon': False},
                    color='Total Penjualan',
                    color_continuous_scale=KERATONIAN_GRADIENT,
                    size_max=40,
                    labels={'Total Penjualan': 'Revenue (Rp)'}
                )
                
                # Focus on Indonesia region
                fig_map.update_geos(
                    visible=True,
                    resolution=50,
                    showcountries=True,
                    countrycolor="lightgray",
                    showcoastlines=True,
                    coastlinecolor="gray",
                    showland=True,
                    landcolor="rgb(243, 243, 243)",
                    center=dict(lat=-2.5, lon=118),
                    projection_type="natural earth",
                    lataxis_range=[-11, 6],
                    lonaxis_range=[95, 141]
                )
                
                fig_map.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_colorbar=dict(
                        title="Revenue",
                        tickformat='.2s',
                        len=0.7
                    )
                )
                
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Show mapping status
                if len(unmapped_provinces) > 0:
                    with st.expander(f"ℹ️ Info Pemetaan ({len(province_sales_mapped)} dari {len(province_sales)} provinsi terpetakan)"):
                        st.caption(f"**Provinsi yang belum terpetakan:** {', '.join(unmapped_provinces)}")
                        st.caption("Provinsi ini tetap ditampilkan di bar chart di bawah.")
            else:
                st.warning(f"📍 Tidak ada provinsi yang dapat dipetakan dari {len(province_sales)} provinsi di data Anda.")
                st.info(f"**Nama provinsi di data Anda:** {', '.join(province_sales['Provinsi'].tolist()[:10])}" + 
                       (f" ... dan {len(province_sales) - 10} lainnya" if len(province_sales) > 10 else ""))
                st.caption("Silakan hubungi developer untuk menambahkan mapping provinsi yang belum terdaftar.")
            
            st.divider()
            
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
            
            st.divider()
            
            # --- AI GEOGRAPHIC INSIGHTS ---
            st.write("### 🤖 Geographic Insights ")
            
            # Calculate insights
            total_provinces = geo_df['Provinsi'].nunique()
            total_districts = geo_df['Daerah'].nunique()
            top_province = by_province.iloc[0]['Provinsi'] if len(by_province) > 0 else "N/A"
            top_province_revenue = by_province.iloc[0]['Total Penjualan'] if len(by_province) > 0 else 0
            top_province_pct = (top_province_revenue / geo_df['Total Penjualan'].sum() * 100) if geo_df['Total Penjualan'].sum() > 0 else 0
            
            top_district = by_district.iloc[0]['Daerah'] if len(by_district) > 0 else "N/A"
            top_district_revenue = by_district.iloc[0]['Total Penjualan'] if len(by_district) > 0 else 0
            
            # Top 3 provinces for concentration analysis
            top_3_provinces = by_province.head(3)['Total Penjualan'].sum() if len(by_province) >= 3 else 0
            top_3_pct = (top_3_provinces / geo_df['Total Penjualan'].sum() * 100) if geo_df['Total Penjualan'].sum() > 0 else 0
            
            # Java vs Non-Java analysis
            java_provinces = ['DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'Jawa Timur', 'Banten', 'DI Yogyakarta', 'Daerah Istimewa Yogyakarta']
            java_revenue = geo_df[geo_df['Provinsi'].isin(java_provinces)]['Total Penjualan'].sum()
            java_pct = (java_revenue / geo_df['Total Penjualan'].sum() * 100) if geo_df['Total Penjualan'].sum() > 0 else 0
            
            ins_geo_col1, ins_geo_col2 = st.columns(2)
            
            with ins_geo_col1:
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #622A0F;">
                    <div class="insight-title"><span class="insight-icon">🏆</span> Dominasi Regional</div>
                    <div class="insight-text">
                        Provinsi <b>{top_province}</b> mendominasi dengan kontribusi <b>{top_province_pct:.1f}%</b> dari total revenue. 
                        Top 3 provinsi menguasai <b>{top_3_pct:.1f}%</b> pasar, menunjukkan konsentrasi geografis yang tinggi.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #7C4700;">
                    <div class="insight-title"><span class="insight-icon">🎯</span> Fokus Kabupaten/Kota</div>
                    <div class="insight-text">
                        <b>{top_district}</b> adalah kabupaten/kota terkuat dengan revenue <b>{format_rupiah(top_district_revenue)}</b>. 
                        Pertimbangkan membuka distributor lokal atau gudang regional di area ini.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with ins_geo_col2:
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #997950;">
                    <div class="insight-title"><span class="insight-icon">🗺️</span> Jawa vs Luar Jawa</div>
                    <div class="insight-text">
                        Pulau Jawa berkontribusi <b>{java_pct:.1f}%</b> dari total penjualan. 
                        {"Dominasi Jawa sangat kuat, pertimbangkan ekspansi agresif ke luar Jawa untuk diversifikasi pasar." if java_pct > 70 else "Distribusi geografis cukup seimbang, pertahankan strategi multi-regional."}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #5C2C06;">
                    <div class="insight-title"><span class="insight-icon">📊</span> Jangkauan Geografis</div>
                    <div class="insight-text">
                        Produk Anda telah menjangkau <b>{total_provinces} provinsi</b> dan <b>{total_districts} kabupaten/kota</b>. 
                        Jangkauan yang luas ini menunjukkan penetrasi pasar nasional yang solid.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ============================================
        # TAB 6: VALUE PREDICTION
        # ============================================
        
        with tab6:
            st.subheader("🔮 Transaction Value Prediction")
            st.write("Prediksi apakah transaksi baru akan menghasilkan nilai tinggi (**High Value**) atau rendah (**Low Value**) berdasarkan AI.")
            
            with st.form("prediction_form"):
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    qty_input = st.number_input("Quantity (Qty)", min_value=1, value=1)
                    price_input = st.number_input("Harga Jual", min_value=1, value=150000)
                    discount_input = st.number_input("Total Diskon", min_value=0, value=0)
                
                with col_p2:
                    provinsi_opt = sorted(df['Provinsi'].astype(str).unique())
                    provinsi_input = st.selectbox("Provinsi Kirim", provinsi_opt)
                    
                    channel_opt = sorted(df['Channel'].astype(str).unique())
                    channel_input = st.selectbox("Sales Channel", channel_opt)
                    
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
                        val_pred = model.predict(input_data)[0]
                        val_prob = model.predict_proba(input_data)[0][val_pred]
                        
                        if val_pred == 1:
                            st.success(f"### Result: 💎 High Value Transaction")
                            st.write(f"Confidence: {val_prob:.2%}")
                        else:
                            st.info(f"### Result: 📦 Low Value Transaction")
                            st.write(f"Confidence: {val_prob:.2%}")
                            
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
                    color_discrete_map={'Historical': '#622A0F', 'Forecast (Est)': '#997950'}
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

            st.divider()

            # --- STRATEGIC Q1 2026 PREDICTION (New Requested Feature) ---
            st.subheader("🎯 Strategic Forecast: Q1 2026 Analysis")
            st.write("Prediksi performa Kuartal 1 (Jan-Mar) tahun 2026 berdasarkan pola musiman 3 tahun terakhir.")

            # Filter data khusus Q1 untuk semua tahun yang tersedia di base data (df)
            q1_all_years = df[df['Tanggal Order'].dt.quarter == 1].copy()
            q1_summary = q1_all_years.groupby(q1_all_years['Tanggal Order'].dt.year)['Total Penjualan'].sum().reset_index()
            q1_summary.columns = ['Tahun', 'Revenue']

            if not q1_summary.empty and len(q1_summary) >= 2:
                # --- MODELING BASED PREDICTION (Linear Regression) ---
                # Prepare X (years as 1, 2, 3...) and Y (revenue)
                X_reg = np.array(range(len(q1_summary))).reshape(-1, 1)
                y_reg = q1_summary['Revenue'].values
                
                # Train simple regression model
                reg_model = LinearRegression()
                reg_model.fit(X_reg, y_reg)
                
                # Predict for next year (index len(X))
                next_index = np.array([[len(q1_summary)]])
                proj_q1_2026 = reg_model.predict(next_index)[0]
                
                # Calculate growth representation for delta
                last_q1_rev = q1_summary['Revenue'].iloc[-1]
                growth_delta = (proj_q1_2026 - last_q1_rev) / last_q1_rev if last_q1_rev > 0 else 0

                col_strat1, col_strat2 = st.columns([1.5, 1])

                with col_strat1:
                    # Bar Chart Historical Q1
                    fig_q1_hist = px.bar(
                        q1_summary,
                        x='Tahun',
                        y='Revenue',
                        title='Historical Q1 Performance (Jan-Mar)',
                        labels={'Revenue': 'Total Revenue', 'Tahun': 'Tahun'},
                        color='Revenue',
                        color_continuous_scale=KERATONIAN_GRADIENT,
                        text_auto='.2s'
                    )
                    fig_q1_hist.update_layout(xaxis=dict(type='category'))
                    st.plotly_chart(fig_q1_hist, use_container_width=True)

                with col_strat2:
                    st.write("#### AI Regression Projection")
                    st.metric(
                        "Projected Q1 2026 (AI Model)",
                        format_rupiah(proj_q1_2026),
                        delta=f"{(growth_delta*100):.1f}% vs 2025"
                    )
                    
                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color: #7C4700; height: auto; padding: 15px;">
                        <div class="insight-title"><span class="insight-icon">🤖</span> AI Modeling Insight</div>
                        <div class="insight-text" style="font-size: 0.85rem;">
                           Berikut prediksi penjualan Q1 di 2026 berdasarkan histori data penjualan pada Q1 di 2023,2024 dan 2025 untuk memprediksi tren yang lebih stabil.
                            <br><br>
                            <b>Hasil Analisis:</b>
                            <ul>
                                <li>Model AI memprediksi pendapatan Q1 2026 di angka <b>{format_rupiah(proj_q1_2026)}</b>.</li>
                                <li>Angka ini lebih realistis karena memperhitungkan perlambatan laju pertumbuhan di tahun 2025.</li>
                                <li><b>Strategi:</b> Fokus pada optimasi operasional untuk menjaga tren pertumbuhan tetap di angka {(growth_delta*100):.1f}%.</li>
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Data historis Q1 tidak cukup untuk melakukan analisis perbandingan tahunan.")


        # ============================================
        # TAB 7: CHURN ANALYSIS
        # ============================================
        
        with tab7:
            st.subheader("🚨 Customer Churn Analysis & Risk Detection")
            st.write("Analisis pelanggan yang berisiko berhenti berlangganan atau tidak melakukan order ulang.")
            
            # --- CHURN CALCULATION FOR CURRENT FILTERED DATA ---
            max_d = filtered_df['Tanggal Order'].max()
            churn_analysis = filtered_df.groupby('Cust').agg({
                'Tanggal Order': 'max',
                'Total Penjualan': 'sum',
                'Pesanan': 'count',
                'Qty': 'sum',
                'Provinsi': 'first',
                'Channel': 'first'
            }).rename(columns={'Pesanan': 'Frequency', 'Tanggal Order': 'Last_Order'})
            
            churn_analysis['Days_Since_Last'] = (max_d - churn_analysis['Last_Order']).dt.days
            churn_analysis['Status'] = np.where(churn_analysis['Days_Since_Last'] > 180, 'Churned', 'Active')
            
            c_metric1, c_metric2, c_metric3 = st.columns(3)
            with c_metric1:
                total_cust = len(churn_analysis)
                st.metric("Total Customers", f"{total_cust:,}")
            with c_metric2:
                active_cust = len(churn_analysis[churn_analysis['Status'] == 'Active'])
                st.metric("Active Customers", f"{active_cust:,}", delta=f"{(active_cust/total_cust*100) if total_cust > 0 else 0:.1f}%")
            with c_metric3:
                churned_cust = len(churn_analysis[churn_analysis['Status'] == 'Churned'])
                st.metric("Churned Customers (>180d)", f"{churned_cust:,}", delta=f"-{(churned_cust/total_cust*100) if total_cust > 0 else 0:.1f}%", delta_color="inverse")
            
            st.divider()
            
            col_ch1, col_ch2 = st.columns([1, 1])
            
            with col_ch1:
                st.write("### Churn Distribution")
                status_counts = churn_analysis['Status'].value_counts()
                fig_churn = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    hole=0.4,
                    color=status_counts.index,
                    color_discrete_map={'Active': '#7C4700', 'Churned': '#622A0F'}
                )
                st.plotly_chart(fig_churn, use_container_width=True)
                
            with col_ch2:
                st.write("### 🤖 Data Insights: Churn Analysis")
                
                # Logic for Churn Insights
                avg_days = churn_analysis['Days_Since_Last'].mean()
                top_churn_prov = churn_analysis[churn_analysis['Status'] == 'Churned']['Provinsi'].mode().iloc[0] if not churn_analysis[churn_analysis['Status'] == 'Churned'].empty else "N/A"
                
                st.markdown(f"""
                <div class="insight-card" style="border-left-color: #622A0F; height: auto;">
                    <div class="insight-title"><span class="insight-icon">📉</span> Analisis Loyalitas</div>
                    <div class="insight-text">
                        Rata-rata pelanggan sudah tidak melakukan transaksi selama <b>{avg_days:.1f} hari</b>. 
                        Provinsi dengan tingkat churn tertinggi adalah <b>{top_churn_prov}</b>.
                        <br><br>
                        <b>Rekomendasi Strategis</b>
                        <ul>
                            <li><b>Re-engagement:</b> Kirim pesan singkat "We Miss You" kepada pelanggan yang terdaftar di daftar Churned.</li>
                            <li><b>Loyalty Program:</b> Berikan poin atau reward khusus untuk pelanggan Active agar tidak berpindah ke kompetitor.</li>
                            <li><b>Survey:</b> Lakukan survey kepuasan pelanggan di area {top_churn_prov} untuk mencari tahu alasan berhenti order.</li>
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            st.write("### 👥 Customers at Risk / Churned List")
            at_risk = churn_analysis[churn_analysis['Status'] == 'Churned'].sort_values('Days_Since_Last', ascending=False)
            
            if not at_risk.empty:
                st.warning(f"Ditemukan {len(at_risk)} pelanggan yang tidak aktif lebih dari 6 bulan.")
                at_risk_display = at_risk[['Last_Order', 'Days_Since_Last', 'Frequency', 'Total Penjualan', 'Provinsi']].copy()
                at_risk_display['Last_Order'] = pd.to_datetime(at_risk_display['Last_Order']).dt.date
                at_risk_display['Total Penjualan'] = at_risk_display['Total Penjualan'].apply(format_rupiah)
                st.dataframe(at_risk_display.head(50), use_container_width=True)
            else:
                st.success("Hebat! Semua pelanggan aktif dalam 6 bulan terakhir.")

        # ============================================
        # TAB 8: STOCK RECOMMENDATION
        # ============================================
        
        with tab8:
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
                if "High" in val: color = '#622A0F' # Cinnamon (Dark)
                elif "Medium" in val: color = '#7C4700' # Brown
                else: color = '#997950' # Tortilla (Light)
                return f'background-color: {color}; color: white; font-weight: bold'

            def color_rec(val):
                if "URGENT" in val: color = '#622A0F'
                elif "Siaga" in val: color = '#7C4700'
                else: color = '#997950'
                return f'color: {color}; font-weight: bold'

            st.dataframe(
                stock_data.style.applymap(color_market, subset=['Kondisi Pasar'])
                .applymap(color_rec, subset=['REKOMENDASI GUDANG']),
                use_container_width=True,
                hide_index=True
            )

        # ============================================
        # TAB 9: EXPORT
        # ============================================
        
        with tab9:
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
