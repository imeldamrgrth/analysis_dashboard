import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# Konfigurasi halaman
st.set_page_config(page_title="E-Commerce Data Analysis", layout="wide")

# Sidebar dengan logo dan filter tanggal
with st.sidebar:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        image = Image.open(logo_path)
        st.image(image, use_container_width=True)
    
    st.write("### Pilih Rentang Tanggal")
    date_range = st.date_input("Rentang Tanggal", value=(pd.to_datetime("2017-01-01"), pd.to_datetime("2018-01-01")), 
                               min_value=pd.to_datetime("2016-01-01"), max_value=pd.to_datetime("2018-12-31"))

# Load dataset
@st.cache_data
def load_data():
    geolocation = pd.read_csv("geolocation_cleaned.csv")
    orders = pd.read_csv("orders_cleaned.csv", parse_dates=["order_purchase_timestamp"])
    order_items = pd.read_csv("order_items_cleaned.csv")
    order_payments = pd.read_csv("order_payments_cleaned.csv")
    order_reviews = pd.read_csv("order_reviews_cleaned.csv", parse_dates=["review_creation_date", "review_answer_timestamp"])
    products = pd.read_csv("products_cleaned.csv")
    sellers = pd.read_csv("sellers_cleaned.csv")
    
    return geolocation, orders, order_items, order_payments, order_reviews, products, sellers

# Load semua dataset
geolocation, orders, order_items, order_payments, order_reviews, products, sellers = load_data()

# Konversi ke datetime64[ns]
start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Filter orders berdasarkan tanggal
filtered_orders = orders[(orders['order_purchase_timestamp'] >= start_date) & 
                         (orders['order_purchase_timestamp'] <= end_date)]
filtered_order_payments = order_payments[order_payments['order_id'].isin(filtered_orders['order_id'])]
payment_counts = filtered_order_payments['payment_type'].value_counts()
filtered_order_items = order_items[order_items['order_id'].isin(filtered_orders['order_id'])]
filtered_order_reviews = order_reviews[order_reviews['order_id'].isin(filtered_orders['order_id'])]

# Hitung Total Orders dan Total Revenue
total_orders = len(filtered_orders)
total_revenue = filtered_order_items['price'].sum()

# Header Dashboard
st.title("📊 E-Commerce Data Analysis Dashboard")
st.write("## Daily Orders")
col1, col2 = st.columns(2)
col1.metric("Total Orders", total_orders)
col2.metric("Total Revenue", f"AUD {total_revenue:,.2f}")

# Menghitung total pendapatan dan jumlah pesanan per penjual
seller_performance = order_items.groupby('seller_id').agg(
    total_sales=('price', 'sum'),
    total_orders=('order_id', 'nunique')
)

st.write("## 🛍️ Performansi Penjual")
st.bar_chart(seller_performance['total_sales'].sort_values(ascending=False).head(10))  # Menampilkan 10 penjual dengan total penjualan tertinggi

# Menggabungkan order_items_cleaned dengan products_cleaned berdasarkan 'product_id'
merged_order_items = pd.merge(order_items, products, on='product_id', how='left')

# Sekarang kolom 'product_category_name' sudah ada di dalam merged_order_items
category_sales = merged_order_items.groupby('product_category_name').agg(
    total_sales=('price', 'sum'),
    total_units_sold=('order_item_id', 'count')
)

st.write("## 📦 Kategori Produk Paling Laris")
st.bar_chart(category_sales['total_units_sold'].sort_values(ascending=False).head(10))  # Menampilkan 10 kategori produk dengan penjualan tertinggi

# Menghitung distribusi metode pembayaran berdasarkan jumlah dan total pembayaran
payment_distribution = order_payments.groupby('payment_type').agg(
    total_payment_value=('payment_value', 'sum'),
    total_transactions=('payment_value', 'count')
)

st.write("## 💳 Distribusi Metode Pembayaran")
st.bar_chart(payment_distribution['total_transactions'].sort_values(ascending=False))  # Menampilkan metode pembayaran berdasarkan jumlah transaksi


# 1. Pengaruh Keterlambatan Pengiriman terhadap Customer Churn
# Pastikan kolom tanggal dalam format datetime
filtered_orders['order_delivered_customer_date'] = pd.to_datetime(filtered_orders['order_delivered_customer_date'])
filtered_orders['order_estimated_delivery_date'] = pd.to_datetime(filtered_orders['order_estimated_delivery_date'])

# Menambahkan kolom 'is_late' untuk keterlambatan pengiriman
filtered_orders['is_late'] = filtered_orders['order_delivered_customer_date'] > filtered_orders['order_estimated_delivery_date']

# Menampilkan churn rate berdasarkan keterlambatan pengiriman
churn_rate = filtered_orders.groupby('is_late')['order_id'].nunique()
st.write("## 📉 Pengaruh Keterlambatan Pengiriman terhadap Customer Churn")
st.bar_chart(churn_rate)


# 2. Menggabungkan data order dengan review untuk analisis kepuasan
merged_reviews = pd.merge(orders, order_reviews, on='order_id', how='left')

# Membuat kolom 'is_late' untuk mengetahui apakah pengiriman terlambat
merged_reviews['is_late'] = merged_reviews['order_delivered_customer_date'] > merged_reviews['order_estimated_delivery_date']

# Visualisasi box plot skor review berdasarkan keterlambatan pengiriman
st.write("## 📊 Pengaruh Keterlambatan Pengiriman terhadap Skor Ulasan")
fig, ax = plt.subplots()
sns.boxplot(x='is_late', y='review_score', data=merged_reviews, palette='Set2')
ax.set_title("Keterlambatan Pengiriman vs Skor Ulasan")
st.pyplot(fig)



# 4. Tren Penjualan Berdasarkan Waktu
# Mengonversi kolom order_purchase_timestamp ke datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Menghitung volume penjualan per bulan
monthly_sales = orders.groupby(orders['order_purchase_timestamp'].dt.to_period('M')).size()

st.write("## 📅 Prediksi Volume Penjualan di Masa Depan")
st.line_chart(monthly_sales)



# Menggabungkan data order_items dengan review_score
merged_data = pd.merge(order_items, order_reviews, on='order_id', how='left')

# Visualisasi hubungan harga produk dan skor ulasan
st.write("## 💸 Harga Produk vs Skor Ulasan")
fig, ax = plt.subplots()
sns.scatterplot(data=merged_data, x='price', y='review_score', ax=ax)
ax.set_title("Harga Produk vs Skor Ulasan")
st.pyplot(fig)


