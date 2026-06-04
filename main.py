import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta
load_dotenv()
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from supabase import create_client

app = FastAPI(title="Algo Trading Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
data_client = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

TAKE_PROFIT_PCT = 0.10
STOP_LOSS_PCT = 0.05

class RunRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    params: Optional[Dict[str, Any]] = {}
    dry_run: bool = True

def get_bars(symbols, days=120):
    start_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start_str,
        feed="iex"
    )
    bars = data_client.get_stock_bars(req).df
    if bars.empty:
        return pd.DataFrame(columns=["symbol", "timestamp", "close"])
    bars = bars.reset_index()
    bars.columns = [c.lower() for c in bars.columns]
    return bars

def get_positions():
    try:
        positions = trading_client.get_all_positions()
        return {p.symbol: p for p in positions}
    except:
        return {}

def run_momentum(data, params):
    signals = []
    lb, thr = params.get("lookback", 20), params.get("threshold", 0.02)
    for symbol in data["symbol"].unique():
        df = data[data["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        if len(df) < lb + 1:
            continue
        ret = (df["close"].iloc[-1] - df["close"].iloc[-lb]) / df["close"].iloc[-lb]
        if ret > thr:
            signals.append({"symbol": symbol, "side": "buy", "confidence": min(ret / thr, 1.0)})
        elif ret < -thr:
            signals.append({"symbol": symbol, "side": "sell", "confidence": min(abs(ret) / thr, 1.0)})
    return signals

def run_macd(data, params):
    signals = []
    fast, slow, sig = params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
    for symbol in data["symbol"].unique():
        df = data[data["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        if len(df) < slow + sig + 2:
            continue
        ema_f = df["close"].ewm(span=fast, adjust=False).mean()
        ema_s = df["close"].ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        sl = macd.ewm(span=sig, adjust=False).mean()
        hist = macd - sl
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            signals.append({"symbol": symbol, "side": "buy", "confidence": 0.8})
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            signals.append({"symbol": symbol, "side": "sell", "confidence": 0.8})
    return signals

STRATEGIES = {"macd": run_macd, "momentum": run_momentum}

@app.post("/run-strategy")
async def run_strategy(req: RunRequest):
    clock = trading_client.get_clock()
    if not clock.is_open:
        return {
            "skipped": True,
            "reason": "market_closed",
            "next_open": clock.next_open.isoformat(),
            "message": "Strategy aborted — NYSE is not currently open."
        }
    if req.strategy_name not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy_name}")
    try:
        data = get_bars(req.symbols)
        if data.empty:
            return {"signals": [], "dry_run": req.dry_run, "portfolio_value": 0, "actions": []}
        signals = STRATEGIES[req.strategy_name](data, req.params)
        positions = get_positions()
        account = trading_client.get_account()
        pv = float(account.portfolio_value or account.cash or 0)
        results = []
        actions_taken = []
        for symbol, pos in positions.items():
            if symbol not in req.symbols:
                continue
            current_price = float(data[data["symbol"] == symbol]["close"].iloc[-1]) if symbol in data["symbol"].values else None
            if current_price is None:
                continue
            entry_price = float(pos.avg_entry_price)
            pnl_pct = (current_price - entry_price) / entry_price
            qty = float(pos.qty)
            should_close = False
            close_reason = ""
            if pnl_pct >= TAKE_PROFIT_PCT:
                should_close = True
                close_reason = f"take profit +{pnl_pct*100:.1f}%"
            elif pnl_pct <= -STOP_LOSS_PCT:
                should_close = True
                close_reason = f"stop loss {pnl_pct*100:.1f}%"
            else:
                sell_signal = any(s["symbol"] == symbol and s["side"] == "sell" for s in signals)
                if sell_signal:
                    should_close = True
                    close_reason = "strategy sell signal"
            if should_close:
                action = {"symbol": symbol, "action": "close", "reason": close_reason,
                          "qty": qty, "entry": entry_price, "current": current_price,
                          "pnl_pct": round(pnl_pct * 100, 2)}
                if not req.dry_run:
                    trading_client.close_position(symbol)
                    supabase.table("orders").insert({
                        "symbol": symbol, "qty": qty, "side": "sell",
                        "status": "closed", "filled_price": current_price,
                        "strategy": req.strategy_name, "mode": "paper",
                        "created_at": datetime.now().isoformat()
                    }).execute()
                actions_taken.append(action)
                results.append({"symbol": symbol, "side": "sell", "confidence": 1.0,
                                 "qty": qty, "price": current_price, "reason": close_reason})
        for s in signals:
            if s["side"] != "buy":
                continue
            symbol = s["symbol"]
            if symbol in positions:
                continue
            sym_data = data[data["symbol"] == symbol]
            if sym_data.empty:
                continue
            price = float(sym_data["close"].iloc[-1])
            qty = round((max(pv, 10000) * 0.05 * s["confidence"]) / price, 2)
            if qty <= 0:
                continue
            action = {"symbol": symbol, "action": "buy", "qty": qty, "price": price,
                      "confidence": s["confidence"]}
            if not req.dry_run:
                order = trading_client.submit_order(MarketOrderRequest(
                    symbol=symbol, qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                ))
                supabase.table("orders").insert({
                    "symbol": symbol, "qty": qty, "side": "buy",
                    "status": "submitted", "alpaca_order_id": str(order.id),
                    "filled_price": price, "strategy": req.strategy_name,
                    "mode": "paper", "created_at": datetime.now().isoformat()
                }).execute()
            actions_taken.append(action)
            results.append({**s, "qty": qty, "price": price, "reason": "new position opened"})
        return {"signals": results, "actions": actions_taken,
                "positions_held": list(positions.keys()),
                "dry_run": req.dry_run, "portfolio_value": pv}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/positions")
async def get_open_positions():
    try:
        positions = trading_client.get_all_positions()
        return [{"symbol": p.symbol, "qty": p.qty, "entry": p.avg_entry_price,
                 "current": p.current_price, "pnl": p.unrealized_pl,
                 "pnl_pct": p.unrealized_plpc} for p in positions]
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/orders")
async def get_orders():
    try:
        return supabase.table("orders").select("*").order("created_at", desc=True).limit(100).execute().data
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
