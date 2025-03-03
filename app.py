import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import requests

# Set up the app title and description
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title('📈 Stock Trend Prediction App')
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        color: #2E86C1;
    }
    </style>
    """, unsafe_allow_html=True)
st.markdown('<p class="big-font">Predict stock trends using historical data, technical indicators, and sentiment analysis.</p>', unsafe_allow_html=True)

# Sidebar for user inputs
with st.sidebar:
    st.header("⚙️ User Inputs")
    user_input = st.text_input('🔍 Enter Stock Ticker:', 'AAPL')
    start_date = st.date_input("📅 Select Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("📅 Select End Date", value=pd.to_datetime("2025-01-01"))
    
    # Optional comparison feature
    compare_stocks = st.checkbox("🔀 Compare with another stock")
    if compare_stocks:
        compare_stock_ticker = st.text_input('🔍 Enter Second Stock Ticker:', 'MSFT')

# Fetching data
st.subheader("📥 Fetching Data...")
with st.spinner("Loading data..."):
    df = yf.download(user_input, start=start_date, end=end_date)
st.success("Data loaded successfully!")

# Display data description
st.subheader('📊 Data Summary')
st.write(df.describe())

# Closing Price vs Time chart
st.subheader('📈 Closing Price vs Time')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price', color='#3498DB')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Closing Price with 100MA chart
st.subheader('📉 Closing Price vs Time with 100-Day Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price', color='#3498DB')
plt.plot(ma100, label='100-Day MA', color='#E67E22')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# Closing Price with 100MA & 200MA chart
st.subheader('📉 Closing Price vs Time with 100-Day & 200-Day Moving Averages')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price', color='#3498DB')
plt.plot(ma100, label='100-Day MA', color='#E67E22')
plt.plot(ma200, label='200-Day MA', color='#2ECC71')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
st.pyplot(fig)

# MACD Indicator
st.subheader('📊 MACD Indicator')
df['ema_12'] = df.Close.ewm(span=12, adjust=False).mean()
df['ema_26'] = df.Close.ewm(span=26, adjust=False).mean()
df['macd'] = df['ema_12'] - df['ema_26']
df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

fig_macd = plt.figure(figsize=(12, 6))
plt.plot(df['macd'], label='MACD', color='#3498DB')
plt.plot(df['signal'], label='Signal', color='#E67E22')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("Time")
plt.ylabel("MACD")
plt.legend()
plt.grid(True)
st.pyplot(fig_macd)

# RSI Indicator
st.subheader('📊 RSI Indicator')
delta = df['Close'].diff(1)
gain, loss = delta.copy(), delta.copy()
gain[gain < 0] = 0
loss[loss > 0] = 0
avg_gain = gain.rolling(window=14).mean()
avg_loss = abs(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100.0 - (100.0 / (1.0 + rs))

fig_rsi = plt.figure(figsize=(12, 6))
plt.plot(df['rsi'], label='RSI', color='#E67E22')
plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
plt.xlabel("Time")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)
st.pyplot(fig_rsi)

# Sentiment Analysis
st.subheader('💬 Sentiment Analysis')
sia = SentimentIntensityAnalyzer()

# Example news article or social media post
news_text = f"Recent news about {user_input} indicates a positive outlook due to strong quarterly earnings."
sentiment_scores = sia.polarity_scores(news_text)

st.write(f"**Sentiment Scores for {user_input}:**")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Positive Sentiment", f"{sentiment_scores['pos']:.2f}")
col2.metric("Negative Sentiment", f"{sentiment_scores['neg']:.2f}")
col3.metric("Neutral Sentiment", f"{sentiment_scores['neu']:.2f}")
col4.metric("Compound Sentiment", f"{sentiment_scores['compound']:.2f}")

# Optional two-stock comparison
if compare_stocks:
    st.subheader(f'📈 Closing Price Comparison: {user_input} vs {compare_stock_ticker}')
    compare_df = yf.download(compare_stock_ticker, start=start_date, end=end_date)
    fig_compare = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label=user_input, color='#3498DB')
    plt.plot(compare_df.Close, label=compare_stock_ticker, color='#E74C3C')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_compare)

# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# Display training and testing data shapes
st.write(f"Training Data Shape: {data_training.shape}")
st.write(f"Testing Data Shape: {data_testing.shape}")

# Scaling data for model input
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load pre-trained model
model = load_model('keras_model.h5')

# Prepare test data for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
st.subheader("🔮 Making Predictions...")
with st.spinner("Predicting stock prices..."):
    y_predicted = model.predict(x_test)

# Rescale predictions back to original scale
scaler_factor = 1 / scaler.scale_[0]
y_predicted *= scaler_factor
y_test *= scaler_factor

# Predictions vs Original plot
st.subheader('🔮 Predictions vs Original Prices')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price', linewidth=2)
plt.plot(y_predicted, 'r', label='Predicted Price', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Stock Price Prediction")
plt.legend(loc="upper left")

# Adjust x-axis to show actual dates
test_dates = df.index[int(len(df)*0.7):]  # Extract test data dates
plt.xticks(
    ticks=np.linspace(0, len(test_dates) - 1, num=10).astype(int),  # Select evenly spaced ticks
    labels=test_dates[np.linspace(0, len(test_dates) - 1, num=10).astype(int)].strftime('%Y'),  # Format as year
    rotation=45  # Rotate labels for better readability
)

plt.grid(True)
st.pyplot(fig2)

# Add a section for downloading actual and predicted prices as CSV
st.subheader("📥 Download Actual and Predicted Prices")

# Create a DataFrame for actual and predicted prices
results_df = pd.DataFrame({
    "Date": df.index[int(len(df)*0.7):],  # Use test data dates
    "Actual Price": y_test,
    "Predicted Price": y_predicted.flatten()
})

# Function to convert DataFrame to CSV format
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Convert the DataFrame to CSV
csv_data = convert_df_to_csv(results_df)

# Add a download button
st.download_button(
    label="📥 Download Results as CSV",
    data=csv_data,
    file_name="actual_vs_predicted_prices.csv",
    mime="text/csv"
)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_predicted)

# Calculate R²
r2 = r2_score(y_test, y_predicted)

# Display metrics in Streamlit
st.subheader("📊 Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
col2.metric("Mean Absolute Percentage Error (MAPE)", f"{mape * 100:.2f}%")
col3.metric("R-Squared (R²)", f"{r2:.2f}")

def fetch_news(ticker, api_key):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    return []