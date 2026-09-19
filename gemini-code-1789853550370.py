import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration & Theme ---
st.set_page_config(page_title="Risk & Screener Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .metric-card { background-color: #1E212B; padding: 15px; border-radius: 8px; text-align: center; }
    .status-valid { color: #00C853; font-weight: bold; }
    .status-invalid { color: #D50000; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- API Handling & Caching ---
BASE_URL = "https://fapi.binance.com"

@st.cache_data(ttl=15, show_spinner=False)
def fetch_24h_tickers():
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/ticker/24hr", timeout=5)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error (24h Ticker): {e}")
        return []

@st.cache_data(ttl=15, show_spinner=False)
def fetch_klines(symbol, interval, limit=50):
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5)
        res.raise_for_status()
        df = pd.DataFrame(res.json(), columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"])
        df = df.apply(pd.to_numeric, errors='ignore')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        return df
    except requests.exceptions.RequestException:
        return pd.DataFrame()

@st.cache_data(ttl=15, show_spinner=False)
def fetch_order_book(symbol):
    try:
        res = requests.get(f"{BASE_URL}/fapi/v1/depth", params={"symbol": symbol, "limit": 20}, timeout=5)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return {"bids": [], "asks": []}

@st.cache_data(ttl=15, show_spinner=False)
def fetch_sentiment(symbol):
    try:
        pi_res = requests.get(f"{BASE_URL}/fapi/v1/premiumIndex", params={"symbol": symbol}, timeout=5)
        oi_res = requests.get(f"{BASE_URL}/fapi/v1/openInterest", params={"symbol": symbol}, timeout=5)
        return pi_res.json(), oi_res.json()
    except requests.exceptions.RequestException:
        return {}, {}

# --- Module 1: Universe Selection ---
def filter_universe(tickers):
    exclude_keywords = ['USDC', 'BUSD', 'FDUSD', 'MUUSDT', 'SNDKUSDT', 'SKHYNIXUSDT', 'SPCXUSDT', 'CLUSDT', 'XAUUSDT', 'XAGUSDT']
    
    valid_pairs = []
    for t in tickers:
        symbol = t['symbol']
        if not symbol.endswith('USDT'):
            continue
        if any(ex in symbol for ex in exclude_keywords):
            continue
        valid_pairs.append({
            'Symbol': symbol,
            'Volume (USDT)': float(t['quoteVolume']),
            '24h Change (%)': float(t['priceChangePercent']),
            'Last Price': float(t['lastPrice'])
        })
    
    df = pd.DataFrame(valid_pairs).sort_values(by='Volume (USDT)', ascending=False).head(25).reset_index(drop=True)
    
    def vol_flag(chg):
        if 2.0 <= chg <= 15.0: return "🟢 Healthy Trend"
        elif chg > 40.0: return "🔴 Extended"
        elif chg < 1.5 and chg > -1.5: return "⚪ Flat"
        else: return "🟡 Volatile/Downtrend"
        
    df['Momentum Flag'] = df['24h Change (%)'].apply(vol_flag)
    return df

# --- Module 5: Data Validation ---
def validate_data(klines, ob, symbol):
    errors = []
    if klines.empty: return False, ["Kline data missing"]
    
    # 1. Geometry
    invalid_geometry = klines[(klines['high'] < klines[['open', 'close']].max(axis=1)) | 
                              (klines['low'] > klines[['open', 'close']].min(axis=1))]
    if not invalid_geometry.empty: errors.append("Candlestick geometry mismatch")
        
    # 2. Spread
    if ob['bids'] and ob['asks']:
        best_bid = float(ob['bids'][0][0])
        best_ask = float(ob['asks'][0][0])
        if best_bid >= best_ask: errors.append(f"Spread inverted (Bid: {best_bid} >= Ask: {best_ask})")
    else:
        errors.append("Orderbook empty")
        
    # 3. Latency
    latest_close = klines.iloc[-1]['open_time']
    time_diff = (datetime.now(timezone.utc) - latest_close).total_seconds()
    if time_diff > 900: # greater than 15 mins
        errors.append("Data latency exceeds acceptable threshold")
        
    return len(errors) == 0, errors

# --- Analytical Processing ---
def analyze_asset(symbol):
    k_4h = fetch_klines(symbol, '4h', 150)
    k_15m = fetch_klines(symbol, '15m', 150)
    k_5m = fetch_klines(symbol, '5m', 150)
    ob = fetch_order_book(symbol)
    pi, oi = fetch_sentiment(symbol)
    
    is_valid, errors = validate_data(k_5m, ob, symbol)
    
    if not is_valid:
        return {"valid": False, "errors": errors}
    
    # MAs
    k_4h['MA25'] = k_4h['close'].rolling(25).mean()
    k_4h['MA99'] = k_4h['close'].rolling(99).mean()
    
    k_15m['MA7'] = k_15m['close'].rolling(7).mean()
    k_15m['MA25'] = k_15m['close'].rolling(25).mean()
    k_15m['MA99'] = k_15m['close'].rolling(99).mean()
    k_15m['Vol_MA10'] = k_15m['volume'].rolling(10).mean()
    
    k_5m['MA7'] = k_5m['close'].rolling(7).mean()
    k_5m['MA25'] = k_5m['close'].rolling(25).mean()
    k_5m['Taker_Sell'] = k_5m['volume'] - k_5m['taker_buy_base']
    k_5m['Vol_Delta'] = k_5m['taker_buy_base'] - k_5m['Taker_Sell']
    
    # OB Analysis
    bids = pd.DataFrame(ob['bids'], columns=['price', 'qty'], dtype=float)
    asks = pd.DataFrame(ob['asks'], columns=['price', 'qty'], dtype=float)
    
    best_bid = bids['price'].iloc[0]
    best_ask = asks['price'].iloc[0]
    spread_pct = ((best_ask - best_bid) / best_bid) * 100
    
    total_bid = bids['qty'].sum()
    total_ask = asks['qty'].sum()
    bid_ask_ratio = total_bid / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0
    
    mean_bid_size = bids['qty'].mean()
    mean_ask_size = asks['qty'].mean()
    
    bid_walls = bids[bids['qty'] > (3.0 * mean_bid_size)]
    ask_walls = asks[asks['qty'] > (3.0 * mean_ask_size)]
    
    primary_bid_wall = bid_walls.iloc[0] if not bid_walls.empty else pd.Series({'price': 0, 'qty': 0})
    primary_ask_wall = ask_walls.iloc[0] if not ask_walls.empty else pd.Series({'price': 0, 'qty': 0})
    
    # Sentiment
    funding_rate = float(pi.get('lastFundingRate', 0)) * 100
    
    # Scoring
    score = 0
    last_4h = k_4h.iloc[-1]
    if last_4h['close'] > last_4h['MA25'] > last_4h['MA99']: score += 25
    
    last_15m = k_15m.iloc[-1]
    prev_15m = k_15m.iloc[-2]
    if last_15m['close'] > last_15m['MA25'] and last_15m['MA7'] > prev_15m['MA7']: score += 25
        
    if bid_ask_ratio > 0.55 and not bid_walls.empty: score += 25
    if funding_rate < 0.02 and spread_pct < 0.025: score += 25
        
    # Trade Setup Calculation
    entry_price = max(k_5m.iloc[-1]['MA7'], primary_bid_wall['price'])
    stop_loss = k_5m.iloc[-1]['MA25'] * 0.99
    risk = entry_price - stop_loss
    tp1 = entry_price + (risk * 2)
    tp2 = entry_price + (risk * 3)
    
    return {
        "valid": True,
        "symbol": symbol,
        "score": score,
        "current_price": best_bid,
        "spread_pct": spread_pct,
        "funding_rate": funding_rate,
        "bid_ask_ratio": bid_ask_ratio,
        "primary_bid_wall": primary_bid_wall.to_dict(),
        "primary_ask_wall": primary_ask_wall.to_dict(),
        "entry": entry_price,
        "sl": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "k_15m": k_15m.tail(50),
        "k_5m": k_5m.tail(50),
        "oi": oi.get('openInterest', 0)
    }

# --- UI Application ---
st.title("⚡ Quantitative Futures Screener & Risk Engine")

tickers = fetch_24h_tickers()
if not tickers:
    st.error("Failed to connect to Binance API.")
    st.stop()

universe_df = filter_universe(tickers)

st.header("Section A: Market Overview")
st.dataframe(
    universe_df.style.format({'Volume (USDT)': '{:,.0f}', '24h Change (%)': '{:+.2f}%', 'Last Price': '{:.4f}'}),
    use_container_width=True, hide_index=True
)

st.header("Section B: Deep Dive Analyzer")
selected_symbol = st.selectbox("Select Asset for Algorithmic Audit", universe_df['Symbol'].tolist())

if selected_symbol:
    with st.spinner(f"Analyzing {selected_symbol}..."):
        analysis = analyze_asset(selected_symbol)
        
    if not analysis["valid"]:
        st.error("❌ Data Validation Failed")
        for e in analysis["errors"]: st.write(f"- {e}")
    else:
        st.markdown(f"<div class='status-valid'>✅ Data Validated across Geometry, Spread, and Latency.</div>", unsafe_allow_html=True)
        st.progress(analysis["score"] / 100, text=f"Composite Setup Quality Score: {analysis['score']}/100")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='metric-card'><b>Spread %</b><br>{analysis['spread_pct']:.4f}%</div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><b>Funding Rate</b><br>{analysis['funding_rate']:.4f}%</div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><b>Demand (Bid %)</b><br>{(analysis['bid_ask_ratio']*100):.1f}%</div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='metric-card'><b>Open Interest</b><br>{float(analysis['oi']):,.0f}</div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Market Structure (15M)", "🧮 Order Book Audit", "🤖 Trade Setup & JSON"])
        
        with tab1:
            df_plot = analysis["k_15m"]
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2], vertical_spacing=0.05)
            fig.add_trace(go.Candlestick(x=df_plot['open_time'], open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['open_time'], y=df_plot['MA25'], line=dict(color='orange', width=1.5), name='MA25'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_plot['open_time'], y=df_plot['volume'], name='Volume', marker_color='gray'), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            bc1, bc2 = st.columns(2)
            with bc1:
                st.subheader("Support (Bid Walls)")
                if analysis['primary_bid_wall']['price'] > 0:
                    st.success(f"Wall Detected at **{analysis['primary_bid_wall']['price']}** (Size: {analysis['primary_bid_wall']['qty']})")
                else:
                    st.write("No extreme liquidity deviations detected.")
            with bc2:
                st.subheader("Resistance (Ask Walls)")
                if analysis['primary_ask_wall']['price'] > 0:
                    st.error(f"Wall Detected at **{analysis['primary_ask_wall']['price']}** (Size: {analysis['primary_ask_wall']['qty']})")
                else:
                    st.write("No extreme liquidity deviations detected.")

        with tab3:
            st.subheader("Suggested Risk Parameters")
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Limit Entry", f"{analysis['entry']:.5f}")
            rc2.metric("Stop Loss", f"{analysis['sl']:.5f}", f"{((analysis['sl']-analysis['entry'])/analysis['entry'])*100:.2f}%")
            rc3.metric("Target 1 (1:2)", f"{analysis['tp1']:.5f}")
            rc4.metric("Target 2 (1:3)", f"{analysis['tp2']:.5f}")
            
            st.markdown("---")
            st.subheader("One-Click AI Export")
            export_payload = {
                "Asset": analysis["symbol"],
                "Score": analysis["score"],
                "Metrics": {
                    "Spread_Percent": round(analysis["spread_pct"], 4),
                    "Funding_Percent": round(analysis["funding_rate"], 4),
                    "Bid_Ask_Ratio": round(analysis["bid_ask_ratio"], 2),
                    "Open_Interest": analysis["oi"]
                },
                "Liquidity_Walls": {
                    "Bid_Wall_Price": analysis["primary_bid_wall"]["price"],
                    "Ask_Wall_Price": analysis["primary_ask_wall"]["price"]
                },
                "Execution": {
                    "Entry": analysis["entry"],
                    "Stop_Loss": analysis["sl"],
                    "Target_1": analysis["tp1"],
                    "Target_2": analysis["tp2"]
                }
            }
            
            json_dump = json.dumps(export_payload, indent=2)
            st.code(json_dump, language="json")