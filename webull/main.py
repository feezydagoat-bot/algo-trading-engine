import os 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
load_dotenv()
from webull import webull, paper_webull
from supabase import create_client

app = FastAPI(title="Webull Trading Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WEBULL_ACCESS_TOKEN  = os.getenv("WEBULL_ACCESS_TOKEN")
WEBULL_REFRESH_TOKEN = os.getenv("WEBULL_REFRESH_TOKEN")
WEBULL_TOKEN_EXPIRY  = os.getenv("WEBULL_TOKEN_EXPIRY")
WEBULL_UUID          = os.getenv("WEBULL_UUID")
WEBULL_TRADE_TOKEN   = os.getenv("WEBULL_TRADE_TOKEN")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TAKE_PROFIT_PCT = 0.10
STOP_LOSS_PCT   = 0.05

def get_wb(paper=True):
    wb = paper_webull() if paper else webull()
    if WEBULL_ACCESS_TOKEN:
        wb._access_token  = WEBULL_ACCESS_TOKEN
        wb._refresh_token = WEBULL_REFRESH_TOKEN
        wb._token_expiry  = WEBULL_TOKEN_EXPIRY
        wb._uuid          = WEBULL_UUID
        if WEBULL_TRADE_TOKEN:
            wb._trade_token = WEBULL_TRADE_TOKEN
        try:
            wb.refresh_login()
        except Exception:
            pass
    return wb

def get_bars(symbols, days=120):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    frames = []
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("empty")
            # yfinance >=0.2.x returns MultiIndex columns ("Close","AAPL") for single tickers
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "timestamp", "Close": "close"})
            df["symbol"] = sym
            frames.append(df)
        except Exception:
            try:
                wb = get_wb()
                data = wb.get_bars(stock=sym, interval="d1", count=days)
                if data:
                    df = pd.DataFrame(data)[["timestamp", "close"]]
                    df["symbol"] = sym
                    frames.append(df)
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["symbol", "timestamp", "close"])
    result = pd.concat(frames, ignore_index=True)
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    return result.dropna(subset=["close"])

def get_current_price(symbol):
    try:
        wb = get_wb()
        q = wb.get_quote(stock=symbol)
        price = float(q.get("close") or q.get("pPrice") or 0)
        if price > 0:
