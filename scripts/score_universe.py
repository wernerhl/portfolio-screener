"""
Portfolio Scoring Model — Universe Screener
============================================
Screens S&P 500 + 50 AI/infrastructure mid-caps.
Computes 4-factor score: Fundamental Quality, Technical Positioning,
Visibility/Backlog, and Correlation Penalty.
Outputs ranked watchlist with entry levels.

Usage:
    python score_universe.py

Output:
    data/scored_universe.csv   — full scored universe
    data/watchlist_top30.csv   — top 30 candidates with entry levels
    data/portfolio_report.txt  — summary report
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    os.system(f"{sys.executable} -m pip install yfinance --break-system-packages -q")
    import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Current portfolio (for correlation penalty)
CURRENT_PORTFOLIO = {
    "MSFT": 0.15,   # weight as fraction of equity
    "AVGO": 0.18,
    "GOOG": 0.16,
    "NVDA": 0.17,
    "TSM":  0.10,
    "ETN":  0.06,
    "VRT":  0.06,
    "CEG":  0.06,
    # Add BRK-B, PM, JPM if purchased
}

# The 50 mid-cap AI/infrastructure additions beyond S&P 500
MIDCAP_ADDITIONS = [
    # AI Infrastructure — Connectivity & Networking
    "CRDO",   # Credo Technology — AEC cables
    "ALAB",   # Astera Labs — PCIe retimers
    "COHR",   # Coherent — photonics
    "LITE",   # Lumentum — optical components
    "CIEN",   # Ciena — networking
    "CALX",   # Calix — cloud networking
    "FN",     # Fabrinet — optical manufacturing

    # AI Infrastructure — Semicap & Test
    "ONTO",   # Onto Innovation — process control
    "ACMR",   # ACM Research — wet processing
    "AEHR",   # Aehr Test — burn-in testing
    "PLAB",   # Photronics — photomasks
    "COHU",   # Cohu — test equipment
    "UCTT",   # Ultra Clean — parts & gas delivery
    "KLIC",   # Kulicke & Soffa — bonding equipment

    # AI Power & Grid
    "TLN",    # Talen Energy — nuclear
    "NRG",    # NRG Energy — power generation
    "OKLO",   # Oklo — SMR nuclear
    "BWXT",   # BWX Technologies — nuclear
    "POWL",   # Powell Industries — electrical equipment

    # AI Cloud / HPC
    "CRWV",   # CoreWeave — AI cloud (if public)
    "NBIS",   # Nebius — AI cloud
    "DLR",    # Digital Realty — data center REIT
    "EQIX",   # Equinix — data center REIT
    "QTS",    # QTS Realty (if still public)

    # AI Software / Applications
    "PLTR",   # Palantir
    "AI",     # C3.ai
    "SOUN",   # SoundHound
    "BBAI",   # BigBear.ai
    "PATH",   # UiPath
    "ESTC",   # Elastic
    "MDB",    # MongoDB
    "CFLT",   # Confluent
    "DDOG",   # Datadog

    # Quantum
    "IONQ",   # IonQ
    "RGTI",   # Rigetti
    "QBTS",   # D-Wave

    # Other thematic
    "MRVL",   # Marvell — custom silicon #2
    "ARM",    # ARM Holdings — royalty model
    "SMCI",   # Super Micro — AI servers
    "DELL",   # Dell — servers
    "PSTG",   # Pure Storage
    "WDC",    # Western Digital
    "WOLF",   # Wolfspeed — SiC
    "GEV",    # GE Vernova
    "VST",    # Vistra — power
    "IREN",   # IREN — HPC hosting
    "WULF",   # TeraWulf — HPC hosting
    "APLD",   # Applied Digital — HPC hosting
    "CIFR",   # Cipher Mining
]


# ── S&P 500 Tickers ───────────────────────────────────────────────────

def get_sp500_tickers():
    """Fetch current S&P 500 constituents."""
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = table[0]['Symbol'].tolist()
        # Fix tickers with dots (BRK.B -> BRK-B for Yahoo)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"  Failed to fetch S&P 500 list: {e}")
        print("  Using cached list...")
        # Fallback: top 100 by market cap
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOG","META","BRK-B","AVGO","LLY","JPM",
            "TSLA","V","UNH","XOM","MA","COST","PG","JNJ","HD","ABBV",
            "WMT","NFLX","BAC","CRM","ORCL","CVX","MRK","KO","AMD","PEP",
            "TMO","ADBE","ACN","LIN","MCD","CSCO","PM","ABT","GE","ISRG",
            "TXN","DHR","INTU","NOW","QCOM","CAT","VZ","AMGN","IBM","GS",
            "AXP","BKNG","MS","SPGI","BLK","PFE","T","NEE","LOW","UNP",
            "RTX","HON","ELV","SYK","AMAT","DE","LRCX","SCHW","PLD","TJX",
            "KLAC","ADP","VRTX","MMC","REGN","C","ADI","BSX","PANW","CB",
            "FI","BMY","MDLZ","SBUX","SO","MO","CL","ICE","CME","WM",
            "GD","MCK","APD","FCX","USB","TT","ORLY","AZO","HCA","ANET",
            "TSM","MU","CEG","ETN","VRT","GEV",
        ]


# ── Data Fetching ─────────────────────────────────────────────────────

def fetch_price_data(tickers, period="1y"):
    """Batch download price history."""
    print(f"  Downloading price data for {len(tickers)} tickers...")
    # Split into chunks to avoid timeout
    chunk_size = 100
    all_data = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = yf.download(chunk, period=period, progress=False, auto_adjust=True, threads=True)
            if data is not None and 'Close' in data.columns:
                closes = data['Close']
                if isinstance(closes, pd.Series):
                    all_data[chunk[0]] = closes
                else:
                    for t in closes.columns:
                        if not closes[t].isna().all():
                            all_data[t] = closes[t]
            time.sleep(1)
        except Exception as e:
            print(f"    Chunk {i//chunk_size + 1} error: {e}")
    
    print(f"  Got price data for {len(all_data)} tickers")
    return pd.DataFrame(all_data)


def fetch_fundamentals(tickers):
    """Fetch fundamental data for each ticker."""
    print(f"  Fetching fundamentals for {len(tickers)} tickers...")
    results = {}
    errors = 0
    
    for i, ticker in enumerate(tickers):
        if i % 50 == 0 and i > 0:
            print(f"    ... {i}/{len(tickers)} done ({errors} errors)")
            time.sleep(2)  # Rate limit
        
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or 'marketCap' not in info:
                errors += 1
                continue
            
            results[ticker] = {
                'marketCap': info.get('marketCap', 0),
                'forwardPE': info.get('forwardPE'),
                'trailingPE': info.get('trailingPE'),
                'priceToBook': info.get('priceToBook'),
                'revenueGrowth': info.get('revenueGrowth'),
                'grossMargins': info.get('grossMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'profitMargins': info.get('profitMargins'),
                'returnOnEquity': info.get('returnOnEquity'),
                'returnOnAssets': info.get('returnOnAssets'),
                'debtToEquity': info.get('debtToEquity'),
                'freeCashflow': info.get('freeCashflow'),
                'totalRevenue': info.get('totalRevenue'),
                'earningsGrowth': info.get('earningsGrowth'),
                'currentPrice': info.get('currentPrice') or info.get('regularMarketPrice'),
                'fiftyDayAverage': info.get('fiftyDayAverage'),
                'twoHundredDayAverage': info.get('twoHundredDayAverage'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'shortName': info.get('shortName', ticker),
                'beta': info.get('beta'),
                'dividendYield': info.get('dividendYield'),
            }
        except Exception:
            errors += 1
    
    print(f"  Got fundamentals for {len(results)} tickers ({errors} errors)")
    return results


# ── Factor Computations ───────────────────────────────────────────────

def compute_technical_score(prices_df, ticker):
    """
    Technical Positioning Score (0-25)
    - Distance from 200-DMA
    - RSI (14-day)
    - Relative strength vs SPX (6-month)
    - Distance from 52-week support
    """
    if ticker not in prices_df.columns:
        return None, {}
    
    px = prices_df[ticker].dropna()
    if len(px) < 60:
        return None, {}
    
    current = px.iloc[-1]
    
    # 200-day MA distance (score higher when near or below — buy at support)
    ma200 = px.rolling(200).mean().iloc[-1] if len(px) >= 200 else px.rolling(len(px)).mean().iloc[-1]
    ma200_dist = (current - ma200) / ma200  # negative = below MA
    # Score: best at -5% to +5% of MA (buying near support), worst at +30%+ (extended)
    if ma200_dist < -0.15:
        ma_score = 3.0  # deeply oversold — might be broken, not just cheap
    elif ma200_dist < -0.05:
        ma_score = 6.0  # oversold, near support
    elif ma200_dist < 0.05:
        ma_score = 5.5  # near MA, healthy
    elif ma200_dist < 0.15:
        ma_score = 4.0  # slightly extended
    elif ma200_dist < 0.30:
        ma_score = 2.5  # extended
    else:
        ma_score = 1.0  # very extended
    
    # RSI (14-day)
    delta = px.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    # Score: best at 30-45 (oversold), worst at 75+ (overbought)
    if rsi_val < 30:
        rsi_score = 5.5  # deeply oversold
    elif rsi_val < 45:
        rsi_score = 6.0  # oversold, ideal entry
    elif rsi_val < 55:
        rsi_score = 5.0  # neutral
    elif rsi_val < 70:
        rsi_score = 3.5  # overbought
    else:
        rsi_score = 1.5  # very overbought
    
    # Relative strength vs SPX (6-month)
    if '^GSPC' in prices_df.columns or 'SPY' in prices_df.columns:
        spx_col = '^GSPC' if '^GSPC' in prices_df.columns else 'SPY'
        spx = prices_df[spx_col].dropna()
        min_len = min(len(px), len(spx), 126)
        if min_len > 20:
            stock_ret = (px.iloc[-1] / px.iloc[-min_len] - 1)
            spx_ret = (spx.iloc[-1] / spx.iloc[-min_len] - 1)
            rel_strength = stock_ret - spx_ret
            # Score: outperforming = higher score
            rs_score = min(6.5, max(1.0, 3.5 + rel_strength * 15))
        else:
            rs_score = 3.5
            rel_strength = 0
    else:
        rs_score = 3.5
        rel_strength = 0
    
    # Distance from 52-week support
    low_52w = px.tail(252).min() if len(px) >= 252 else px.min()
    high_52w = px.tail(252).max() if len(px) >= 252 else px.max()
    range_52w = high_52w - low_52w
    if range_52w > 0:
        pct_from_low = (current - low_52w) / range_52w
        # Score: best near the low (buying support), worst near high
        support_score = max(1.0, min(6.5, 6.5 - pct_from_low * 5))
    else:
        support_score = 3.5
    
    total = ma_score + rsi_score + rs_score + support_score
    total = min(25.0, max(0.0, total))
    
    details = {
        'ma200_dist': round(ma200_dist * 100, 1),
        'rsi': round(rsi_val, 1),
        'rel_strength_6m': round(rel_strength * 100, 1),
        'pct_52w_range': round(pct_from_low * 100 if range_52w > 0 else 50, 1),
    }
    
    return round(total, 1), details


def compute_fundamental_score(fund_data):
    """
    Fundamental Quality Score (0-25)
    - FCF Yield (or inverse forward P/E as proxy)
    - Revenue growth
    - Margin quality (gross + operating)
    - ROIC proxy (ROE adjusted for leverage)
    """
    if not fund_data:
        return None, {}
    
    scores = []
    details = {}
    
    # FCF Yield / Valuation (0-7)
    fwd_pe = fund_data.get('forwardPE')
    mcap = fund_data.get('marketCap', 0)
    fcf = fund_data.get('freeCashflow', 0)
    
    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap * 100
        details['fcf_yield'] = round(fcf_yield, 1)
        if fcf_yield > 6: val_score = 7.0
        elif fcf_yield > 4: val_score = 6.0
        elif fcf_yield > 2: val_score = 4.5
        elif fcf_yield > 0: val_score = 3.0
        else: val_score = 1.0
    elif fwd_pe and fwd_pe > 0:
        details['forward_pe'] = round(fwd_pe, 1)
        if fwd_pe < 12: val_score = 6.5
        elif fwd_pe < 20: val_score = 5.5
        elif fwd_pe < 30: val_score = 4.0
        elif fwd_pe < 50: val_score = 2.5
        else: val_score = 1.0
    else:
        val_score = 3.0
    scores.append(val_score)
    
    # Revenue Growth (0-6)
    rev_growth = fund_data.get('revenueGrowth')
    if rev_growth is not None:
        details['rev_growth'] = round(rev_growth * 100, 1)
        if rev_growth > 0.40: growth_score = 6.0
        elif rev_growth > 0.20: growth_score = 5.0
        elif rev_growth > 0.10: growth_score = 4.0
        elif rev_growth > 0.0: growth_score = 3.0
        elif rev_growth > -0.10: growth_score = 2.0
        else: growth_score = 1.0
    else:
        growth_score = 3.0
    scores.append(growth_score)
    
    # Margin Quality (0-6)
    gross = fund_data.get('grossMargins')
    operating = fund_data.get('operatingMargins')
    if gross is not None and operating is not None:
        details['gross_margin'] = round(gross * 100, 1)
        details['op_margin'] = round(operating * 100, 1)
        # Combined margin score
        margin_avg = (gross + max(0, operating)) / 2
        if margin_avg > 0.40: margin_score = 6.0
        elif margin_avg > 0.25: margin_score = 5.0
        elif margin_avg > 0.15: margin_score = 3.5
        elif margin_avg > 0.05: margin_score = 2.0
        else: margin_score = 1.0
    else:
        margin_score = 3.0
    scores.append(margin_score)
    
    # Return on Equity / Capital Efficiency (0-6)
    roe = fund_data.get('returnOnEquity')
    dte = fund_data.get('debtToEquity')
    if roe is not None:
        details['roe'] = round(roe * 100, 1)
        # Penalize high leverage
        leverage_penalty = 0
        if dte and dte > 200:
            leverage_penalty = 1.5
        elif dte and dte > 100:
            leverage_penalty = 0.5
        
        if roe > 0.30: roe_score = 6.0 - leverage_penalty
        elif roe > 0.20: roe_score = 5.0 - leverage_penalty
        elif roe > 0.10: roe_score = 3.5 - leverage_penalty
        elif roe > 0: roe_score = 2.0
        else: roe_score = 1.0
        roe_score = max(1.0, roe_score)
    else:
        roe_score = 3.0
    scores.append(roe_score)
    
    total = sum(scores)
    total = min(25.0, max(0.0, total))
    
    return round(total, 1), details


def compute_visibility_score(fund_data, ticker):
    """
    Visibility / Backlog Score (0-25)
    
    This is the hardest factor to automate. Backlog data isn't in standard feeds.
    We use proxies:
    - Revenue predictability (low quarterly variance)
    - Recurring revenue indicators (SaaS, subscriptions, service contracts)
    - Sector-based visibility premium (utilities, defense, infrastructure > cyclicals)
    
    Manual overrides for known backlog data should be added to VISIBILITY_OVERRIDES.
    """
    VISIBILITY_OVERRIDES = {
        # Ticker: (score, reason)
        "GEV":  (23, "$163B backlog, 25yr service contracts"),
        "AVGO": (22, "$73B contracted backlog through 2028"),
        "ANET": (20, "Multi-year AI networking pipeline"),
        "VRT":  (20, "$15B backlog, 109% YoY growth"),
        "CEG":  (21, "Multi-decade nuclear PPAs"),
        "LMT":  (22, "$166B defense backlog"),
        "RTX":  (21, "$196B defense/aero backlog"),
        "GD":   (21, "$91B defense backlog"),
        "NOC":  (21, "$85B defense backlog"),
        "ORCL": (19, "RPO $130B+"),
        "GOOG": (20, "$460B cloud backlog"),
        "MSFT": (18, "$627B commercial RPO"),
        "AMZN": (17, "AWS backlog growing"),
        "NVDA": (20, "$1T order book through 2027"),
        "TSM":  (21, "Multi-year contracted wafer starts"),
        "ETN":  (19, "Record electrical backlog"),
        "POWL": (18, "Backlog at record levels"),
        "TLN":  (19, "AWS nuclear PPA"),
    }
    
    if ticker in VISIBILITY_OVERRIDES:
        score, reason = VISIBILITY_OVERRIDES[ticker]
        return score, {'override': reason}
    
    if not fund_data:
        return 12.5, {}
    
    # Sector-based visibility premium
    sector = fund_data.get('sector', '')
    industry = fund_data.get('industry', '')
    
    sector_premiums = {
        'Utilities': 18,
        'Consumer Defensive': 16,
        'Healthcare': 15,
        'Financial Services': 14,
        'Industrials': 14,
        'Technology': 13,
        'Communication Services': 13,
        'Basic Materials': 11,
        'Consumer Cyclical': 11,
        'Energy': 12,
        'Real Estate': 15,
    }
    
    base = sector_premiums.get(sector, 12.5)
    
    # Adjust for margin stability (high margins = more pricing power = more visibility)
    gross = fund_data.get('grossMargins')
    if gross and gross > 0.60:
        base += 2
    elif gross and gross > 0.40:
        base += 1
    
    # Adjust for recurring revenue indicators
    if any(kw in industry.lower() for kw in ['software', 'saas', 'subscription', 'service']):
        base += 2
    
    return min(25, round(base, 1)), {'sector_base': sector}


def compute_correlation_penalty(prices_df, ticker, portfolio_tickers, portfolio_weights):
    """
    Correlation Penalty (0 to -10)
    Penalizes tickers highly correlated with existing portfolio.
    """
    if ticker not in prices_df.columns:
        return 0, {}
    
    px = prices_df[ticker].dropna()
    if len(px) < 60:
        return 0, {}
    
    returns = prices_df.pct_change().dropna()
    if ticker not in returns.columns:
        return 0, {}
    
    max_corr = 0
    max_corr_ticker = None
    weighted_corr = 0
    
    for pticker, weight in portfolio_tickers.items():
        pticker_yf = pticker.replace('.', '-')
        if pticker_yf in returns.columns:
            corr_series = returns[[ticker, pticker_yf]].dropna()
            if len(corr_series) > 30:
                corr = corr_series.corr().iloc[0, 1]
                if not np.isnan(corr):
                    weighted_corr += abs(corr) * weight
                    if abs(corr) > max_corr:
                        max_corr = abs(corr)
                        max_corr_ticker = pticker
    
    # Penalty: 0 for uncorrelated, up to -10 for highly correlated
    if max_corr > 0.85:
        penalty = -10.0
    elif max_corr > 0.75:
        penalty = -7.0
    elif max_corr > 0.65:
        penalty = -4.0
    elif max_corr > 0.50:
        penalty = -2.0
    else:
        penalty = 0.0
    
    # Additional penalty for high weighted correlation
    if weighted_corr > 0.5:
        penalty -= 2.0
    
    penalty = max(-10.0, penalty)
    
    details = {
        'max_corr': round(max_corr, 3),
        'max_corr_with': max_corr_ticker,
        'weighted_corr': round(weighted_corr, 3),
    }
    
    return round(penalty, 1), details


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"PORTFOLIO SCORING MODEL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # 1. Build universe
    print("\n[1/5] Building universe...")
    sp500 = get_sp500_tickers()
    universe = list(set(sp500 + MIDCAP_ADDITIONS))
    # Add SPX for relative strength calculation
    universe_with_bench = list(set(universe + ['^GSPC']))
    print(f"  Universe: {len(universe)} tickers ({len(sp500)} S&P 500 + {len(MIDCAP_ADDITIONS)} mid-caps)")
    
    # 2. Fetch price data
    print("\n[2/5] Fetching price data...")
    prices = fetch_price_data(universe_with_bench, period="1y")
    
    # 3. Fetch fundamentals
    print("\n[3/5] Fetching fundamentals...")
    fundamentals = fetch_fundamentals(universe)
    
    # 4. Score everything
    print("\n[4/5] Scoring universe...")
    rows = []
    scored = 0
    
    for ticker in universe:
        fund = fundamentals.get(ticker, {})
        if not fund:
            continue
        
        # Technical score
        tech_score, tech_details = compute_technical_score(prices, ticker)
        if tech_score is None:
            continue
        
        # Fundamental score
        fund_score, fund_details = compute_fundamental_score(fund)
        if fund_score is None:
            continue
        
        # Visibility score
        vis_score, vis_details = compute_visibility_score(fund, ticker)
        
        # Correlation penalty
        corr_penalty, corr_details = compute_correlation_penalty(
            prices, ticker, CURRENT_PORTFOLIO, CURRENT_PORTFOLIO
        )
        
        # Composite score
        composite = tech_score + fund_score + vis_score + corr_penalty
        composite = max(0, min(75, composite))  # 0-75 range (25+25+25-10)
        
        # Category classification
        sector = fund.get('sector', '')
        rev_growth = fund.get('revenueGrowth')
        mcap = fund.get('marketCap', 0)
        
        if rev_growth and rev_growth > 0.30:
            category = "Growth"
        elif vis_score >= 20:
            category = "Compounder"
        elif sector in ['Energy', 'Basic Materials']:
            category = "Cyclical"
        elif mcap and mcap < 10e9:
            category = "Speculative"
        else:
            category = "Core"
        
        # Entry level suggestion (20-DMA or recent support)
        px_series = prices[ticker].dropna() if ticker in prices.columns else pd.Series()
        if len(px_series) > 20:
            ma20 = px_series.rolling(20).mean().iloc[-1]
            ma50 = px_series.rolling(50).mean().iloc[-1] if len(px_series) > 50 else ma20
            current_px = px_series.iloc[-1]
            # Suggest entry at the higher of: 5% below current, or 50-DMA
            entry_level = max(ma50, current_px * 0.95)
            entry_level = round(entry_level, 2)
        else:
            current_px = fund.get('currentPrice', 0)
            entry_level = round(current_px * 0.95, 2) if current_px else 0
        
        # Already in portfolio?
        in_portfolio = ticker in CURRENT_PORTFOLIO or ticker.replace('-', '.') in CURRENT_PORTFOLIO
        
        rows.append({
            'ticker': ticker,
            'name': fund.get('shortName', ticker)[:30],
            'sector': sector,
            'industry': fund.get('industry', '')[:30],
            'category': category,
            'mcap_B': round(mcap / 1e9, 1) if mcap else 0,
            'current_price': round(current_px, 2) if isinstance(current_px, (int, float)) else 0,
            'entry_level': entry_level,
            # Scores
            'fundamental': fund_score,
            'technical': tech_score,
            'visibility': vis_score,
            'corr_penalty': corr_penalty,
            'composite': round(composite, 1),
            # Key metrics
            'fwd_pe': fund_details.get('forward_pe', fund_details.get('fcf_yield', '')),
            'rev_growth_pct': fund_details.get('rev_growth', ''),
            'gross_margin_pct': fund_details.get('gross_margin', ''),
            'roe_pct': fund_details.get('roe', ''),
            'rsi': tech_details.get('rsi', ''),
            'ma200_dist_pct': tech_details.get('ma200_dist', ''),
            'rel_str_6m_pct': tech_details.get('rel_strength_6m', ''),
            'max_corr': corr_details.get('max_corr', ''),
            'max_corr_with': corr_details.get('max_corr_with', ''),
            'in_portfolio': in_portfolio,
        })
        scored += 1
    
    print(f"  Scored: {scored} tickers")
    
    # 5. Build output
    print("\n[5/5] Building output...")
    df = pd.DataFrame(rows)
    df = df.sort_values('composite', ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed rank
    df.index.name = 'rank'
    
    # Save full universe
    df.to_csv(OUTPUT_DIR / "scored_universe.csv")
    print(f"  Full universe: {OUTPUT_DIR / 'scored_universe.csv'}")
    
    # Top 30 watchlist (excluding current portfolio)
    watchlist = df[~df['in_portfolio']].head(30)
    watchlist.to_csv(OUTPUT_DIR / "watchlist_top30.csv")
    print(f"  Top 30 watchlist: {OUTPUT_DIR / 'watchlist_top30.csv'}")
    
    # Summary report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"PORTFOLIO SCORING MODEL — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("=" * 70)
    report_lines.append(f"\nUniverse: {len(df)} scored tickers")
    report_lines.append(f"Score range: {df['composite'].min():.1f} — {df['composite'].max():.1f}")
    report_lines.append(f"Median: {df['composite'].median():.1f}")
    
    report_lines.append("\n" + "─" * 70)
    report_lines.append("TOP 30 CANDIDATES (excluding current portfolio)")
    report_lines.append("─" * 70)
    report_lines.append(f"{'Rank':<5} {'Ticker':<7} {'Name':<25} {'Score':>6} {'Fund':>5} {'Tech':>5} {'Vis':>5} {'Corr':>5} {'Entry':>9} {'Cat':<12}")
    report_lines.append("─" * 70)
    
    for idx, row in watchlist.iterrows():
        report_lines.append(
            f"{idx:<5} {row['ticker']:<7} {row['name']:<25} {row['composite']:>6.1f} "
            f"{row['fundamental']:>5.1f} {row['technical']:>5.1f} {row['visibility']:>5.1f} "
            f"{row['corr_penalty']:>5.1f} {row['entry_level']:>9.2f} {row['category']:<12}"
        )
    
    report_lines.append("\n" + "─" * 70)
    report_lines.append("CURRENT PORTFOLIO SCORES")
    report_lines.append("─" * 70)
    portfolio_df = df[df['in_portfolio']]
    for idx, row in portfolio_df.iterrows():
        report_lines.append(
            f"  {row['ticker']:<7} {row['name']:<25} Score: {row['composite']:>5.1f}  "
            f"(F:{row['fundamental']:.0f} T:{row['technical']:.0f} V:{row['visibility']:.0f})"
        )
    
    report_lines.append("\n" + "─" * 70)
    report_lines.append("SECTOR DISTRIBUTION — TOP 30")
    report_lines.append("─" * 70)
    sector_counts = watchlist['sector'].value_counts()
    for sector, count in sector_counts.items():
        report_lines.append(f"  {sector:<30} {count}")
    
    report_lines.append("\n" + "─" * 70)
    report_lines.append("CATEGORY DISTRIBUTION — TOP 30")
    report_lines.append("─" * 70)
    cat_counts = watchlist['category'].value_counts()
    for cat, count in cat_counts.items():
        report_lines.append(f"  {cat:<20} {count}")
    
    report_text = "\n".join(report_lines)
    with open(OUTPUT_DIR / "portfolio_report.txt", "w") as f:
        f.write(report_text)
    
    print(f"  Report: {OUTPUT_DIR / 'portfolio_report.txt'}")
    
    # Print summary to console
    print("\n" + report_text)
    
    print("\n" + "=" * 70)
    print("DONE. Next steps:")
    print("  1. Review top 30 — apply discretionary judgment layer")
    print("  2. Write entry tickets for selected names")
    print("  3. Set limit orders at entry levels")
    print("  4. Re-run monthly to update scores")
    print("=" * 70)


if __name__ == "__main__":
    main()
