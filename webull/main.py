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
            return price
    except Exception:
        pass
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get("lastPrice") or ticker.fast_info.get("previousClose")
        if price:
            return float(price)
    except Exception:
        pass
    return None

def get_positions(paper=True):
    try:
        wb = get_wb(paper=paper)
        positions = wb.get_positions()
        result = {}
        for p in (positions or []):
            sym = p.get("ticker", {}).get("symbol") or p.get("symbol")
            if sym:
                result[sym] = p
        return result
    except Exception:
        return {}

def get_portfolio_value(paper=True):
    try:
        wb = get_wb(paper=paper)
        acct = wb.get_account()
        net = acct.get("netLiquidation") or acct.get("totalMarketValue") or 0
        if isinstance(net, dict):
            net = net.get("value", 0)
        return float(net or 100000)
    except Exception:
        return 100000

def is_market_open():
    now = datetime.utcnow() - timedelta(hours=4)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t

def next_open_str():
    now = datetime.utcnow() - timedelta(hours=4)
    days = 1
    if now.weekday() == 4: days = 3
    elif now.weekday() == 5: days = 2
    nxt = (now + timedelta(days=days)).replace(hour=9, minute=30, second=0, microsecond=0)
    return nxt.isoformat()

def run_momentum(data, params):
    signals = []
    lb, thr = params.get("lookback", 20), params.get("threshold", 0.02)
    for sym in data["symbol"].unique():
        df = data[data["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if len(df) < lb + 1:
            continue
        ret = (df["close"].iloc[-1] - df["close"].iloc[-lb]) / df["close"].iloc[-lb]
        if ret > thr:
            signals.append({"symbol": sym, "side": "buy",  "confidence": min(ret / thr, 1.0)})
        elif ret < -thr:
            signals.append({"symbol": sym, "side": "sell", "confidence": min(abs(ret) / thr, 1.0)})
    return signals

def run_macd(data, params):
    signals = []
    fast, slow, sig = params.get("fast", 12), params.get("slow", 26), params.get("signal", 9)
    for sym in data["symbol"].unique():
        df = data[data["symbol"] == sym].sort_values("timestamp").reset_index(drop=True)
        if len(df) < slow + sig + 2:
            continue
        ema_f = df["close"].ewm(span=fast, adjust=False).mean()
        ema_s = df["close"].ewm(span=slow, adjust=False).mean()
        macd  = ema_f - ema_s
        sl    = macd.ewm(span=sig, adjust=False).mean()
        hist  = macd - sl
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            signals.append({"symbol": sym, "side": "buy",  "confidence": 0.8})
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            signals.append({"symbol": sym, "side": "sell", "confidence": 0.8})
    return signals

STRATEGIES = {"momentum": run_momentum, "macd": run_macd}

class RunRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    params: Optional[Dict[str, Any]] = {}
    dry_run: bool = True
    paper: bool = True

@app.post("/run-strategy")
async def run_strategy(req: RunRequest):
    if not is_market_open():
        return {"skipped": True, "reason": "market_closed",
                "next_open": next_open_str(),
                "message": "Strategy aborted — NYSE is not currently open."}
    if req.strategy_name not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy_name}")
    try:
        data      = get_bars(req.symbols)
        if data.empty:
            return {"signals": [], "dry_run": req.dry_run, "portfolio_value": 0, "actions": []}
        signals   = STRATEGIES[req.strategy_name](data, req.params)
        positions = get_positions(paper=req.paper)
        pv        = get_portfolio_value(paper=req.paper)
        results, actions_taken = [], []
        wb_client = get_wb(paper=req.paper)
        for sym, pos in positions.items():
            if sym not in req.symbols:
                continue
            current_price = get_current_price(sym)
            if current_price is None:
                sd = data[data["symbol"] == sym]
                current_price = float(sd["close"].iloc[-1]) if not sd.empty else None
            if current_price is None:
                continue
            cost  = float(pos.get("costPrice") or pos.get("avgCost") or 0)
            qty   = float(pos.get("position") or pos.get("qty") or 0)
            if cost <= 0 or qty <= 0:
                continue
            pnl_pct = (current_price - cost) / cost
            should_close, close_reason = False, ""
            if pnl_pct >= TAKE_PROFIT_PCT:
                should_close = True; close_reason = f"take profit +{pnl_pct*100:.1f}%"
            elif pnl_pct <= -STOP_LOSS_PCT:
                should_close = True; close_reason = f"stop loss {pnl_pct*100:.1f}%"
            elif any(s["symbol"] == sym and s["side"] == "sell" for s in signals):
                should_close = True; close_reason = "strategy sell signal"
            if should_close:
                if not req.dry_run:
                    wb_client.place_order(stock=sym, action="SELL", orderType="MKT", enforce="DAY", quant=qty)
                    supabase.table("orders").insert({"symbol": sym, "qty": qty, "side": "sell",
                        "status": "closed", "filled_price": current_price,
                        "strategy": req.strategy_name, "mode": "paper" if req.paper else "live",
                        "created_at": datetime.now().isoformat()}).execute()
                actions_taken.append({"symbol": sym, "action": "close", "reason": close_reason, "qty": qty})
                results.append({"symbol": sym, "side": "sell", "confidence": 1.0,
                                 "qty": qty, "price": current_price, "reason": close_reason})
        for s in signals:
            if s["side"] != "buy": continue
            sym = s["symbol"]
            if sym in positions: continue
            current_price = get_current_price(sym)
            if current_price is None:
                sd = data[data["symbol"] == sym]
                current_price = float(sd["close"].iloc[-1]) if not sd.empty else None
            if current_price is None: continue
            qty = round((max(pv, 10000) * 0.05 * s["confidence"]) / current_price, 2)
            if qty <= 0: continue
            if not req.dry_run:
                wb_client.place_order(stock=sym, action="BUY", orderType="MKT", enforce="DAY", quant=qty)
                supabase.table("orders").insert({"symbol": sym, "qty": qty, "side": "buy",
                    "status": "submitted", "filled_price": current_price,
                    "strategy": req.strategy_name, "mode": "paper" if req.paper else "live",
                    "created_at": datetime.now().isoformat()}).execute()
            actions_taken.append({"symbol": sym, "action": "buy", "qty": qty, "price": current_price})
            results.append({**s, "qty": qty, "price": current_price, "reason": "new position opened"})
        return {"signals": results, "actions": actions_taken,
                "positions_held": list(positions.keys()),
                "dry_run": req.dry_run, "portfolio_value": pv}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/positions")
async def get_open_positions(paper: bool = True):
    try:
        positions = get_positions(paper=paper)
        result = []
        for sym, p in positions.items():
            cost  = float(p.get("costPrice") or p.get("avgCost") or 0)
            qty   = float(p.get("position") or p.get("qty") or 0)
            price = get_current_price(sym) or cost
            pnl   = (price - cost) * qty
            result.append({"symbol": sym, "qty": qty, "entry": cost, "current": price,
                           "pnl": round(pnl, 2), "pnl_pct": round((price-cost)/cost*100, 2) if cost > 0 else 0})
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/orders")
async def get_orders():
    try:
        return supabase.table("orders").select("*").order("created_at", desc=True).limit(100).execute().data
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "broker": "webull", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
