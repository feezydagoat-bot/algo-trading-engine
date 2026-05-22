import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from supabase import create_client
from datetime import datetime, timedelta

app = FastAPI(title="Algo Trading Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class RunRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    params: Optional[Dict[str, Any]] = {}
    dry_run: bool = True

def get_bars(symbols, days=60):
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.now() - timedelta(days=days)
    )
    return data_client.get_stock_bars(req).df.reset_index()

def run_macd(data, params):
    signals = []
    fast, slow, sig = params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
    for symbol in data["symbol"].unique():
        df = data[data["symbol"] == symbol].sort_values("timestamp")
        if len(df) < slow + sig:
            continue
        macd = df["close"].ewm(span=fast).mean() - df["close"].ewm(span=slow).mean()
        sl = macd.ewm(span=sig).mean()
        hist = macd - sl
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            signals.append({"symbol": symbol, "side": "buy", "confidence": 0.8})
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            signals.append({"symbol": symbol, "side": "sell", "confidence": 0.8})
    return signals

def run_momentum(data, params):
    signals = []
    lb, thr = params.get("lookback", 20), params.get("threshold", 0.02)
    for symbol in data["symbol"].unique():
        df = data[data["symbol"] == symbol].sort_values("timestamp")
        if len(df) < lb:
            continue
        ret = df["close"].pct_change(lb).iloc[-1]
        if ret > thr:
            signals.append({"symbol": symbol, "side": "buy", "confidence": min(ret / thr, 1.0)})
        elif ret < -thr:
            signals.append({"symbol": symbol, "side": "sell", "confidence": min(abs(ret) / thr, 1.0)})
    return signals

STRATEGIES = {"macd": run_macd, "momentum": run_momentum}

@app.post("/run-strategy")
async def run_strategy(req: RunRequest):
    if req.strategy_name not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy_name}")
    try:
        data = get_bars(req.symbols)
        signals = STRATEGIES[req.strategy_name](data, req.params)
        account = trading_client.get_account()
        pv = float(account.portfolio_value)
        results = []
        for s in signals:
            price = float(data[data["symbol"] == s["symbol"]]["close"].iloc[-1])
            qty = round((pv * 0.05 * s["confidence"]) / price, 2)
            if not req.dry_run and qty > 0:
                order = trading_client.submit_order(MarketOrderRequest(
                    symbol=s["symbol"], qty=qty,
                    side=OrderSide.BUY if s["side"] == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                ))
                supabase.table("orders").insert({
                    "symbol": s["symbol"], "qty": qty, "side": s["side"],
                    "status": "submitted", "alpaca_order_id": str(order.id)
                }).execute()
            results.append({**s, "qty": qty, "price": price})
        return {"signals": results, "dry_run": req.dry_run, "portfolio_value": pv}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/strategies")
async def get_strategies():
    return supabase.table("strategies").select("*").execute().data

@app.get("/performance")
async def get_performance():
    return supabase.table("performance").select("*").order("date", desc=True).limit(30).execute().data

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
