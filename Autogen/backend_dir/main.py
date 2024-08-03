from fastapi import FastAPI
import yfinance as yf

app = FastAPI()

@app.get("/cdr_daily_spread")
async def calculate_daily_spread():
    cdr = yf.Ticker("CDR.WA")
    today_data = cdr.history(period="1d")
    spread = ((today_data["High"] - today_data["Low"]) / today_data["Low"]) * 100
    return spread