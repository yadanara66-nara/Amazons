import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import plotly.express as px
import spacy
from transformers import pipeline
# import pickle

st.set_page_config(page_title="Amazon AI Dashboard", layout="wide")

st.title("Amazon Product Intelligence Dashboard")
st.write("AI-driven insights from Amazon product data")

df = pd.read_csv("data/cleaned_amazon.csv")

st.subheader("Product Dataset")
st.dataframe(df.head())

df = pd.read_csv("data/cleaned_amazon.csv")
print(df.columns)



df['sentiment'] = df['review_content'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity
)

def get_sentiment(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df['sentiment_category'] = df['sentiment'].apply(get_sentiment)

sentiment_counts = df['sentiment_category'].value_counts()

st.bar_chart(sentiment_counts)

# model = pickle.load(open("models/product_model.pkl", "rb"))

# KPI
total_products = df['product_id'].nunique()
avg_rating = df['rating'].mean()
total_reviews = df['rating_count'].sum()

col1, col2, col3 = st.columns(3)

col1.metric("Total Products", total_products)
col2.metric("Average Rating", round(avg_rating,2))
col3.metric("Total Reviews", total_reviews)


# product insights

# Step 1: Extract the first word of product name
df['short_product_name'] = df['product_name'].str.split().str[0]

# Step 2: Title
st.subheader("Top Products by Review Count")

# Step 3: Group data
top_products = df.groupby('short_product_name')['rating_count'] \
                 .sum() \
                 .sort_values(ascending=False) \
                 .head(10) \
                 .reset_index()

# Step 4: Create interactive chart
fig = px.bar(
    top_products,
    x='short_product_name',
    y='rating_count',
    title="Top Products by Review Count",
    hover_data=['rating_count']
)

# Step 5: Show chart
st.plotly_chart(fig)

#  Extract main category
df['main_category'] = df['category'].str.split('|').str[0]

# Title
st.subheader("Product Category Distribution")

# Count categories
category_counts = df['main_category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']

# Create interactive chart
fig = px.bar(
    category_counts,
    x='Category',
    y='Count',
    title="Product Category Distribution",
    hover_data=['Count']
)

# Show chart
st.plotly_chart(fig)

# Title
st.subheader("Rating Distribution")

# Step 1: Count how many reviews for each rating
rating_counts = df['rating'].value_counts().sort_index().reset_index()
rating_counts.columns = ['Rating', 'Count']

# Step 2: Create interactive bar chart
fig = px.bar(
    rating_counts,
    x='Rating',
    y='Count',
    title="How Many People Gave Each Rating",
    hover_data=['Count']
)

# Step 3: Show chart
st.plotly_chart(fig)

st.subheader("Sales: Normal Price vs Discount")

# Step 1: Create a new column to classify price type
df['price_type'] = df['discount_percentage'].apply(
    lambda x: 'Discounted' if x > 0 else 'Normal Price'
)

# Step 2: Count sales in each group
sales_counts = df['price_type'].value_counts().reset_index()
sales_counts.columns = ['Price Type', 'Sales Count']

# Step 3: Create interactive bar chart
fig = px.bar(
    sales_counts,
    x='Price Type',
    y='Sales Count',
    color='Price Type',
    title="Sales Comparison: Normal Price vs Discount",
    hover_data=['Sales Count']
)

# Step 4: Show chart
st.plotly_chart(fig)









# pip install textblob
# streamlit run app.py
# python app.py