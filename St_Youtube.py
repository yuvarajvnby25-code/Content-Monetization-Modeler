import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px


# Load dataset

df = pd.read_csv("/Users/yuvaraj/GUVI CLASSESS/Content Monetization Modeler/youtube_engineered.csv")  


# Encode categorical columns

le_category = LabelEncoder()
le_device = LabelEncoder()
le_country = LabelEncoder()

df['category'] = le_category.fit_transform(df['category'])
df['device'] = le_device.fit_transform(df['device'])
df['country'] = le_country.fit_transform(df['country'])


# Features and target

features = ['views', 'likes', 'comments', 'watch_time_minutes',
            'video_length_minutes', 'subscribers', 'category', 'device', 'country']
X = df[features]
y = df['ad_revenue_usd']

# Train Linear Regression
model = LinearRegression()
model.fit(X, y)


# Prepare df_plot for charts

df_plot = df.copy()
df_plot['category_name'] = le_category.inverse_transform(df_plot['category'])
df_plot['device_name'] = le_device.inverse_transform(df_plot['device'])
df_plot['country_name'] = le_country.inverse_transform(df_plot['country'])


# Streamlit Layout

st.title("Ad Revenue Predictor 💰")
st.write("Enter video details and explore revenue insights:")


# INPUTS :

with st.container():
    st.header("Input Video Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        views = st.number_input("Views", min_value=0, value=1000)
    with col2:
        likes = st.number_input("Likes", min_value=0, value=100)
    with col3:
        comments = st.number_input("Comments", min_value=0, value=10)

    col1, col2, col3 = st.columns(3)
    with col1:
        watch_time_minutes = st.number_input("Watch Time Minutes", min_value=0, value=500)
    with col2:
        video_length_minutes = st.number_input("Video Length Minutes", min_value=1, value=10)
    with col3:
        subscribers = st.number_input("Subscribers", min_value=0, value=10000)

    col1, col2, col3 = st.columns(3)
    with col1:
        category_input = st.selectbox("Category", df['category'].unique())
    with col2:
        device_input = st.selectbox("Device", df['device'].unique())
    with col3:
        country_input = st.selectbox("Country", df['country'].unique())

    # Predict button below
    if st.button("Predict Revenue"):
        input_df = pd.DataFrame({
            'views':[views],'likes':[likes],'comments':[comments],
            'watch_time_minutes':[watch_time_minutes],'video_length_minutes':[video_length_minutes],
            'subscribers':[subscribers],'category':[category_input],
            'device':[device_input],'country':[country_input]
        })
        revenue = model.predict(input_df)[0]
        st.success(f"Estimated Ad Revenue: ${revenue:.2f}")

# OUTPUTS: 

st.header("📊 Revenue Insights Dashboard")
chart_col1, chart_col2 = st.columns(2)

# --- Chart 1: Avg Revenue by Category (Pie) ---
with chart_col1:
    revenue_by_category = df_plot.groupby('category_name')['ad_revenue_usd'].mean().sort_values(ascending=False)
    st.subheader("Avg Revenue by Category")
    fig1 = px.pie(values=revenue_by_category.values, names=revenue_by_category.index,
                  title="Revenue Distribution by Category", hole=0.4)
    st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Views vs Revenue (Scatter) ---
with chart_col2:
    st.subheader("Views vs Revenue")
    fig2 = px.scatter(df_plot, x='views', y='ad_revenue_usd',
                      color='ad_revenue_usd', size='watch_time_minutes',
                      hover_data=['likes','comments'], labels={'views':'Views','ad_revenue_usd':'Revenue'},
                      color_continuous_scale='Viridis')
    st.plotly_chart(fig2, use_container_width=True)

# --- Chart 3: Avg Revenue by Device (Bar) ---
with chart_col1:
    revenue_by_device = df_plot.groupby('device_name')['ad_revenue_usd'].mean().sort_values()
    st.subheader("Avg Revenue by Device")
    fig3 = px.bar(revenue_by_device, x=revenue_by_device.index, y=revenue_by_device.values,
                  labels={'x':'Device','y':'Revenue'}, color=revenue_by_device.values,
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig3, use_container_width=True)

# --- Chart 4: Avg Revenue by Country (Pie) ---
with chart_col2:
    revenue_by_country = df_plot.groupby('country_name')['ad_revenue_usd'].mean().sort_values(ascending=False)
    st.subheader("Avg Revenue by Country")
    fig4 = px.pie(values=revenue_by_country.values, names=revenue_by_country.index,
                  title="Revenue Distribution by Country", hole=0.4)
    st.plotly_chart(fig4, use_container_width=True)

# --- Feature Importance ---
st.subheader("Top 5 Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False).head(5)
fig5 = px.bar(feature_importance, x='Feature', y='Coefficient', color='Coefficient', color_continuous_scale='Viridis')
st.plotly_chart(fig5, use_container_width=True)
