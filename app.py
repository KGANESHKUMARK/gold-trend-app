# gold_api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/gold-price")
def get_gold_price():
    today = datetime.date.today()
    start = today - datetime.timedelta(days=90)
    df = yf.download("GC=F", start=start, end=today)
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df['Date'] = df['Date'].astype(str)
    return df.to_dict(orient="records")

@app.get("/gold-price/india")
def get_gold_price_india():
    url = "https://www.goodreturns.in/gold-rates/"
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, 'html.parser')
    rate_tag = soup.find("div", class_="gold_silver_table")
    rows = rate_tag.find_all("tr")
    gold_prices = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            city = cols[0].text.strip()
            price = cols[1].text.strip().replace(',', '').replace('₹', '')
            try:
                gold_prices.append({"city": city, "price": float(price)})
            except:
                pass
    return gold_prices

@app.get("/war-news")
def get_latest_war_news():
    url = "https://www.aljazeera.com/tag/war/"
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, 'html.parser')
    articles = soup.find_all('a', class_="u-clickable-card__link", limit=5)
    news = []
    for article in articles:
        title_tag = article.find('span')
        if title_tag:
            title = title_tag.get_text()
            link = "https://www.aljazeera.com" + article['href']
            news.append({'title': title, 'link': link})
    return news

@app.get("/gold-price/predict")
def predict_gold_price():
    today = datetime.date.today()
    start = today - datetime.timedelta(days=90)
    df = yf.download("GC=F", start=start, end=today)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df['days'] = (df['Date'] - df['Date'].min()).dt.days

    X = df[['days']].values
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array([[df['days'].max() + 1]])
    predicted_price = model.predict(future_days)[0]

    return {
        "predicted_price_tomorrow": round(predicted_price, 2),
        "min_price_last_30_days": round(df['Close'][-30:].min(), 2),
        "max_price_last_30_days": round(df['Close'][-30:].max(), 2),
        "last_updated": str(today)
    }
