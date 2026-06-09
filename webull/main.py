import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
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

ET = ZoneInfo("America/New_York")

DEFAULT_SYMBOLS  = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ"]
DEFAULT_STRATEGY = "momentum"

class RunRequest(BaseModel):
    strategy_name: str = DEFAULT_STRATEGY
    symbols: List[str] = DEFAULT_SYMBOLS
    params: Optional[Dict[str, Any]] = {}
    dry_run: bool = False


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


def is_market_open():
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close


def get_next_open():
    now = datetime.now(ET)
    candidate = now + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.replace(hour=9, minute=30, second=0, microsecond=0).isoformat()


def get_bars(symbols, days=120):
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    frames = []
    for sym in symbols:
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError("empty")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()[["Date", "Close"]].rename(
                columns={"Date": "timestamp", "Close": "close"}
            )
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
        price = ticker.fast_info.last_price
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return None


def get_positions():
    try:
        wb = get_wb()
        positions = wb.get_positions()
        result = {}
        for p in positions:
            ticker = p.get("ticker", {})
            symbol = ticker.get("symbol") or p.get("symbol")
            if symbol:
                result[symbol] = p
        return result
    except Exception:
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
            signals.append({"symbol": symbol, "side": "buy",  "confidence": min(ret / thr, 1.0)})
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
        macd  = ema_f - ema_s
        sl    = macd.ewm(span=sig, adjust=False).mean()
        hist  = macd - sl
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            signals.append({"symbol": symbol, "side": "buy",  "confidence": 0.8})
        elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            signals.append({"symbol": symbol, "side": "sell", "confidence": 0.8})
    return signals


STRATEGIES = {"macd": run_macd, "momentum": run_momentum}


async def _execute_strategy(req: RunRequest):
    if not is_market_open():
        return {
            "skipped": True,
            "reason": "market_closed",
            "next_open": get_next_open(),
            "message": "Strategy aborted — NYSE is not currently open."
        }
    if req.strategy_name not in STRATEGIES:
        raise HTTPException(400, f"Unknown strategy: {req.strategy_name}")
    try:
        wb   = get_wb()
        data = get_bars(req.symbols)
        if data.empty:
            return {"signals": [], "dry_run": req.dry_run, "portfolio_value": 0, "actions": []}

        signals   = STRATEGIES[req.strategy_name](data, req.params)
        positions = get_positions()

        try:
            account = wb.get_account()
            pv = float(
                account.get("netLiquidation")
                or account.get("totalMarketValue")
                or account.get("cashBalance")
                or 10000
            )
        except Exception:
            pv = 10000

        results       = []
        actions_taken = []

        for symbol, pos in positions.items():
            if symbol not in req.symbols:
                continue
            try:
                ticker      = pos.get("ticker", {})
                entry_price = float(pos.get("costPrice") or ticker.get("close") or 0)
                qty         = float(pos.get("position") or pos.get("qty") or 0)
            except Exception:
                continue
            current_price = get_current_price(symbol)
            if current_price is None or entry_price == 0 or qty == 0:
                continue

            pnl_pct      = (current_price - entry_price) / entry_price
            should_close = False
            close_reason = ""

            if pnl_pct >= TAKE_PROFIT_PCT:
                should_close = True
                close_reason = f"take profit +{pnl_pct*100:.1f}%"
            elif pnl_pct <= -STOP_LOSS_PCT:
                should_close = True
                close_reason = f"stop loss {pnl_pct*100:.1f}%"
            else:
                if any(s["symbol"] == symbol and s["side"] == "sell" for s in signals):
                    should_close = True
                    close_reason = "strategy sell signal"

            if should_close:
                action = {
                    "symbol": symbol, "action": "close", "reason": close_reason,
                    "qty": qty, "entry": entry_price, "current": current_price,
                    "pnl_pct": round(pnl_pct * 100, 2)
                }
                if not req.dry_run:
                    wb.place_order(stock=symbol, price=current_price,
                                   action="SELL", orderType="MKT",
                                   enforce="DAY", quant=qty)
                    supabase.table("orders").insert({
                        "symbol": symbol, "qty": qty, "side": "sell",
                        "status": "closed", "filled_price": current_price,
                        "strategy": req.strategy_name, "mode": "paper",
                        "created_at": datetime.now().isoformat()
                    }).execute()
                actions_taken.append(action)
                results.append({
                    "symbol": symbol, "side": "sell", "confidence": 1.0,
                    "qty": qty, "price": current_price, "reason": close_reason
                })

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
            qty   = round((max(pv, 10000) * 0.05 * s["confidence"]) / price, 2)
            if qty <= 0:
                continue
            action = {
                "symbol": symbol, "action": "buy",
                "qty": qty, "price": price, "confidence": s["confidence"]
            }
            if not req.dry_run:
                wb.place_order(stock=symbol, price=price,
                               action="BUY", orderType="MKT",
                               enforce="DAY", quant=qty)
                supabase.table("orders").insert({
                    "symbol": symbol, "qty": qty, "side": "buy",
                    "status": "submitted", "filled_price": price,
                    "strategy": req.strategy_name, "mode": "paper",
                    "created_at": datetime.now().isoformat()
                }).execute()
            actions_taken.append(action)
            results.append({**s, "qty": qty, "price": price, "reason": "new position opened"})

        return {
            "signals": results,
            "actions": actions_taken,
            "positions_held": list(positions.keys()),
            "dry_run": req.dry_run,
            "portfolio_value": pv
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/run-strategy")
async def run_strategy_post(req: RunRequest):
    return await _execute_strategy(req)


@app.get("/run-strategy")
async def run_strategy_get():
    return await _execute_strategy(RunRequest())


@app.get("/positions")
async def get_open_positions():
    try:
        wb        = get_wb()
        positions = wb.get_positions()
        result    = []
        for p in positions:
            ticker  = p.get("ticker", {})
            symbol  = ticker.get("symbol") or p.get("symbol", "")
            qty     = float(p.get("position") or p.get("qty") or 0)
            entry   = float(p.get("costPrice") or 0)
            current = get_current_price(symbol) or float(ticker.get("close") or 0)
            pnl_per = current - entry
            result.append({
                "symbol":  symbol,
                "qty":     qty,
                "entry":   entry,
                "current": current,
                "pnl":     round(pnl_per * qty, 2),
                "pnl_pct": round((pnl_per / entry) * 100, 2) if entry else 0
            })
        return result
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
    return {"status": "ok", "broker": "webull", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
