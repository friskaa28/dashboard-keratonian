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

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Keratonian Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling dengan GRADIENT BACKGROUND
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        min-height: 100vh;
    }
    
    .main {
        padding-top: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        margin: 20px;
    }
    
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }
    .welcome-container {
        text-align: center;
        padding: 50px 20px;
    }
    .setup-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
    }

    /* Styling untuk Quarter buttons */
    .quarter-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quarter-button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .quarter-button:active {
        transform: scale(0.98);
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# HELPER FUNCTIONS
# ============================================

def format_rupiah(value):
    """Format angka ke format Rupiah (Rp) - versi Indonesia"""
    if pd.isna(value):
        return "Rp 0"
    
    value = float(value)
    if value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} Miliar"
    elif value >= 1_000_000:
        return f"Rp {value/1_000_000:.1f} Juta"
    elif value >= 1_000:
        return f"Rp {value/1_000:.0f} Ribu"
    else:
        return f"Rp {value:,.0f}"

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
    
    # Extract time components
    df['Tahun'] = df['Tanggal Order'].dt.year
    df['Bulan'] = df['Tanggal Order'].dt.month
    df['Bulan_Nama'] = df['Tanggal Order'].dt.strftime('%B')
    df['Kuartal'] = df['Tanggal Order'].dt.quarter
    df['Hari_Minggu'] = df['Tanggal Order'].dt.day_name()
    
    return df

# Load data
try:
    df = load_data('Keratonian_anomalies_marked_CLEAN.csv')
except FileNotFoundError:
    st.error("❌ File 'Keratonian_anomalies_marked_CLEAN.csv' tidak ditemukan!")
    st.info("Pastikan file CSV ada di folder yang sama dengan script ini")
    st.stop()

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'dashboard_ready' not in st.session_state:
    st.session_state.dashboard_ready = False
    st.session_state.selected_tahun = sorted(df['Tahun'].unique())[0]
    st.session_state.selected_quarter = None

# ============================================
# WELCOME/SETUP PAGE
# ============================================

if not st.session_state.dashboard_ready:
    st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
    
    # Header dengan Logo
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("# 📊 KERATONIAN SALES DASHBOARD")
        st.markdown("### Interactive Analytics & Reporting System")
    
    st.divider()
    
    # Setup form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Tahun Selection
        st.markdown("#### 📅 Pilih Tahun:")
        tahun_list = sorted(df['Tahun'].unique())
        tahun_cols = st.columns(len(tahun_list))
        
        selected_tahun = st.session_state.selected_tahun
        for idx, (col, tahun) in enumerate(zip(tahun_cols, tahun_list)):
            with col:
                if st.button(f"📆 {tahun}", key=f"tahun_{tahun}", use_container_width=True):
                    st.session_state.selected_tahun = tahun
                    selected_tahun = tahun
        
        st.divider()
        
        # Kuartal Selection - CENTERED with better spacing
        st.markdown("#### 📊 Pilih Kuartal:")
        
        quarters = [
            ("Q1", 1),
            ("Q2", 2),
            ("Q3", 3),
            ("Q4", 4),
            ("FULL YEAR", None)
        ]
        
        # Buat 3 baris untuk quarter buttons
        row1_cols = st.columns([1, 1, 1, 1, 1])
        
        selected_quarter = st.session_state.selected_quarter
        
        for idx, (col, (label, value)) in enumerate(zip(row1_cols, quarters)):
            with col:
                # Highlight selected quarter dengan styling
                if selected_quarter == value:
                    button_style = "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-weight: bold;"
                else:
                    button_style = ""
                
                if st.button(label, key=f"quarter_{label}", use_container_width=True):
                    st.session_state.selected_quarter = value
                    selected_quarter = value
                    st.rerun()
        
        st.divider()
        
        # Preview - dengan card styling
        st.markdown("#### ✅ Preview:")
        
        # Calculate preview data
        preview_df = df[df['Tahun'] == selected_tahun]
        if selected_quarter is not None:
            preview_df = preview_df[preview_df['Kuartal'] == selected_quarter]
        
        # Display metrics dengan styling yang lebih baik
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("📅 Tahun", selected_tahun)
        
        with metric_col2:
            if selected_quarter is None:
                quarter_text = "Full Year"
            else:
                quarter_text = f"Q{selected_quarter}"
            st.metric("📊 Kuartal", quarter_text)
        
        with metric_col3:
            st.metric("📦 Data", f"{len(preview_df):,} transaksi")
        
        st.divider()
        
        # Open Dashboard Button - dengan styling gradient
        col_open1, col_open2, col_open3 = st.columns([1, 1, 1])
        with col_open2:
            if st.button("📊 BUKA DASHBOARD", use_container_width=True, type="primary"):
                st.session_state.dashboard_ready = True
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# MAIN DASHBOARD (After Setup)
# ============================================

else:
    # Get selected values from session state
    selected_tahun = st.session_state.selected_tahun
    selected_quarter = st.session_state.selected_quarter
    
    # Prepare data based on selection
    dashboard_df = df[df['Tahun'] == selected_tahun].copy()
    
    if selected_quarter is not None:
        dashboard_df = dashboard_df[dashboard_df['Kuartal'] == selected_quarter]
        quarter_label = f"Q{selected_quarter} {selected_tahun}"
    else:
        quarter_label = f"Full Year {selected_tahun}"
    
    # Back button in sidebar
    st.sidebar.title("🎛️ DASHBOARD")
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
    
    # Customer Type Filter
    cust_types = [ct for ct in dashboard_df['Kategori Channel'].unique() if ct != 'Sample']
    selected_cust_types = st.sidebar.multiselect(
        "👥 Tipe Pelanggan:",
        cust_types,
        default=cust_types
    )
    
    # Reset Button
    if st.sidebar.button("🔄 Reset Filter", use_container_width=True):
        st.rerun()
    
    # Apply Filters - PERBAIKAN: Handle empty filter
    if len(selected_channels) == 0:
        selected_channels = channels
    if len(selected_cust_types) == 0:
        selected_cust_types = cust_types
    
    filtered_df = dashboard_df[
        (dashboard_df['Channel'].isin(selected_channels)) &
        (dashboard_df['Kategori Channel'].isin(selected_cust_types))
    ].copy()
    
    st.sidebar.divider()
    st.sidebar.metric("📊 Data Ditampilkan", f"{len(filtered_df):,} transaksi")
    
    # ============================================
    # MAIN HEADER
    # ============================================
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 📊 KERATONIAN SALES DASHBOARD")
        st.markdown(f"### {quarter_label}")
    
    st.divider()
    
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
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Overview",
            "🏆 Best Seller",
            "⏰ Prime Time",
            "👥 Customers",
            "🗺️ Geographic",
            "📥 Export"
        ])
        
        # ============================================
        # TAB 1: OVERVIEW
        # ============================================
        
        with tab1:
            st.subheader("📊 Dashboard Overview")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("**Top 5 Produk by Quantity**")
                top_products = filtered_df.groupby('Produk')['Qty'].sum().nlargest(5)
                fig, ax = plt.subplots(figsize=(10, 5))
                top_products.plot(kind='barh', ax=ax, color='steelblue')
                ax.set_title('Top 5 Produk', fontweight='bold', fontsize=11)
                ax.set_xlabel('Quantity')
                st.pyplot(fig, use_container_width=True)
            
            with col_right:
                st.write("**Revenue by Channel**")
                channel_sales = filtered_df.groupby('Channel')['Total Penjualan'].sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(channel_sales.values, labels=channel_sales.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Sales Distribution', fontweight='bold', fontsize=11)
                st.pyplot(fig, use_container_width=True)
        
        # ============================================
        # TAB 2: BEST SELLER
        # ============================================
        
        with tab2:
            st.subheader("🏆 Best Seller Analysis")
            
            best_qty = filtered_df.groupby('Produk').agg({
                'Qty': 'sum',
                'Total Penjualan': 'sum',
                'Pesanan': 'count'
            }).sort_values('Qty', ascending=False)
            best_qty.columns = ['Unit', 'Revenue', 'Transaksi']
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### By Quantity")
                fig, ax = plt.subplots(figsize=(10, 6))
                best_qty['Unit'].head(8).plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title('Top 8 Produk', fontweight='bold')
                ax.set_ylabel('Unit Terjual')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig, use_container_width=True)
            
            with col_right:
                st.write("### By Revenue (Hover untuk lihat nilai pasti)")
                best_revenue = filtered_df.groupby('Produk')['Total Penjualan'].sum().sort_values(ascending=False).head(8)
                
                # Buat Plotly chart dengan custom hover
                fig = go.Figure(data=[
                    go.Bar(
                        x=best_revenue.values,
                        y=best_revenue.index,
                        orientation='h',
                        marker=dict(color='coral'),
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
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### Monthly Revenue Trend (Hover untuk lihat nilai)")
                monthly = filtered_df.groupby('Bulan_Nama')['Total Penjualan'].sum()
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly = monthly.reindex([m for m in month_order if m in monthly.index])
                
                fig = go.Figure(data=[
                    go.Scatter(
                        x=[m[:3] for m in monthly.index],
                        y=monthly.values,
                        mode='lines+markers',
                        line=dict(color='#1f77b4', width=2.5),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Revenue: %{text}<extra></extra>',
                        text=[format_rupiah(v) for v in monthly.values]
                    )
                ])
                fig.update_layout(
                    title='Revenue Trend',
                    xaxis_title='Bulan',
                    yaxis_title='Revenue (Rp)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.write("### By Day of Week (Hover untuk lihat nilai)")
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekly = filtered_df.groupby('Hari_Minggu')['Total Penjualan'].sum().reindex(day_order)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=[d[:3] for d in day_order],
                        y=weekly.values,
                        marker=dict(color='lightgreen'),
                        hovertemplate='<b>%{x}</b><br>Revenue: %{text}<extra></extra>',
                        text=[format_rupiah(v) for v in weekly.values]
                    )
                ])
                fig.update_layout(
                    title='Sales by Day of Week',
                    xaxis_title='Hari',
                    yaxis_title='Revenue (Rp)',
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ============================================
        # TAB 4: CUSTOMER ANALYSIS
        # ============================================
        
        with tab4:
            st.subheader("👥 Customer Segmentation")
            
            cust_analysis = filtered_df.groupby('Cust').agg({
                'Pesanan': 'count',
                'Total Penjualan': 'sum'
            }).sort_values('Total Penjualan', ascending=False)
            cust_analysis.columns = ['Frequency', 'Monetary']
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### Top 10 by Frequency")
                top_freq = cust_analysis['Frequency'].nlargest(10)
                fig, ax = plt.subplots(figsize=(10, 6))
                top_freq.plot(kind='barh', ax=ax, color='skyblue')
                ax.set_title('Most Active Customers', fontweight='bold')
                ax.set_xlabel('Purchase Count')
                st.pyplot(fig, use_container_width=True)
            
            with col_right:
                st.write("### Top 10 by Spending (Hover untuk lihat nilai pasti)")
                top_monetary = cust_analysis['Monetary'].nlargest(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=top_monetary.values,
                        y=top_monetary.index,
                        orientation='h',
                        marker=dict(color='coral'),
                        hovertemplate='<b>%{y}</b><br>Total Spending: %{text}<extra></extra>',
                        text=[format_rupiah(v) for v in top_monetary.values]
                    )
                ])
                fig.update_layout(
                    title='Highest Spenders',
                    xaxis_title='Revenue (Rp)',
                    yaxis_title='Customer',
                    height=400,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.write("### Customer Type Distribution")
            cust_type = filtered_df.groupby('Kategori Channel')['Total Penjualan'].sum()
            fig, ax = plt.subplots(figsize=(10, 5))
            cust_type.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'])
            ax.set_title('Revenue by Customer Type (Rp)', fontweight='bold')
            ax.set_ylabel('Revenue (Rp)')
            plt.xticks(rotation=0)
            st.pyplot(fig, use_container_width=True)
            
            # Format monetary column for display
            cust_analysis_display = cust_analysis.copy()
            cust_analysis_display['Monetary'] = cust_analysis_display['Monetary'].apply(format_rupiah)
            st.dataframe(cust_analysis_display.head(15), use_container_width=True)
        
        # ============================================
        # TAB 5: GEOGRAPHIC
        # ============================================
        
        with tab5:
            st.subheader("🗺️ Geographic Distribution")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.write("### Top 10 Provinces (Hover untuk lihat nilai pasti)")
                by_province = filtered_df.groupby('Provinsi')['Total Penjualan'].sum().sort_values(ascending=False).head(10)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=by_province.values,
                        y=by_province.index,
                        orientation='h',
                        marker=dict(color='steelblue'),
                        hovertemplate='<b>%{y}</b><br>Revenue: %{text}<extra></extra>',
                        text=[format_rupiah(v) for v in by_province.values]
                    )
                ])
                fig.update_layout(
                    title='Revenue by Province',
                    xaxis_title='Revenue (Rp)',
                    yaxis_title='Provinsi',
                    height=400,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_right:
                st.write("### Channel Performance")
                by_channel = filtered_df.groupby('Channel')['Total Penjualan'].sum()
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.pie(by_channel.values, labels=by_channel.index, autopct='%1.1f%%', startangle=90)
                ax.set_title('Revenue by Channel', fontweight='bold')
                st.pyplot(fig, use_container_width=True)
            
            # PERBAIKAN: Cek apakah kolom Kabupaten ada
            if 'Kabupaten' in filtered_df.columns and len(filtered_df) > 0:
                st.write("### Top 10 Districts (Hover untuk lihat nilai pasti)")
                try:
                    by_district = filtered_df.groupby('Kabupaten')['Total Penjualan'].sum().sort_values(ascending=False).head(10)
                    
                    if len(by_district) > 0:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=by_district.values,
                                y=by_district.index,
                                orientation='h',
                                marker=dict(color='steelblue'),
                                hovertemplate='<b>%{y}</b><br>Revenue: %{text}<extra></extra>',
                                text=[format_rupiah(v) for v in by_district.values]
                            )
                        ])
                        fig.update_layout(
                            title='Revenue by District',
                            xaxis_title='Revenue (Rp)',
                            yaxis_title='Kabupaten',
                            height=400,
                            hovermode='closest'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("ℹ️ Tidak ada data District untuk periode ini")
                except Exception as e:
                    st.warning(f"⚠️ Tidak bisa menampilkan data District: {str(e)}")
            else:
                st.info("ℹ️ Kolom Kabupaten tidak tersedia atau data kosong")
        
        # ============================================
        # TAB 6: EXPORT
        # ============================================
        
        with tab6:
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
                    'Best_Seller': filtered_df.groupby('Produk').agg({'Qty': 'sum', 'Total Penjualan': 'sum'}).sort_values('Qty', ascending=False),
                    'Top_Customers': filtered_df.groupby('Cust').agg({'Pesanan': 'count', 'Total Penjualan': 'sum'}).sort_values('Total Penjualan', ascending=False),
                    'By_Province': filtered_df.groupby('Provinsi').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}).sort_values('Total Penjualan', ascending=False),
                    'By_Channel': filtered_df.groupby('Channel').agg({'Total Penjualan': 'sum', 'Qty': 'sum'}),
                }
                
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    for sheet_name, df_sheet in export_sheets.items():
                        df_sheet.to_excel(writer, sheet_name=sheet_name[:31])
                
                st.download_button(
                    label="⬇️ Download Excel",
                    data=output.getvalue(),
                    file_name=f"Dashboard_Keratonian_{selected_tahun}_{quarter_label.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                # CSV Export
                st.write("**📄 Export ke CSV**")
                csv_data = filtered_df[['Tanggal Order', 'Channel', 'Cust', 'Produk', 'Qty', 'Total Penjualan', 'Provinsi']].to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"Dashboard_Keratonian_{selected_tahun}_{quarter_label.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.divider()
            st.write("### Preview Data (100 rows)")
            
            # Format revenue for preview - PERBAIKAN: Handle missing columns
            preview_cols = ['Tanggal Order', 'Channel', 'Cust', 'Produk', 'Qty', 'Total Penjualan', 'Provinsi']
            # Cek kolom mana yang available
            available_cols = [col for col in preview_cols if col in filtered_df.columns]
            
            if len(available_cols) > 0:
                preview_df = filtered_df[available_cols].head(100).copy()
                # Format revenue if available
                if 'Total Penjualan' in preview_df.columns:
                    preview_df['Total Penjualan'] = preview_df['Total Penjualan'].apply(format_rupiah)
                st.dataframe(preview_df, use_container_width=True)
            else:
                st.warning("⚠️ Tidak ada kolom untuk ditampilkan")
    
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
