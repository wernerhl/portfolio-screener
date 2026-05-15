"""
JSON output wrapper for the scoring model.
Reads config.json, runs scoring, outputs data/scores.json for the frontend.
"""
import json, sys, os
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Patch the scoring script to use config
ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config.json"
OUTPUT = ROOT / "data" / "scores.json"

def load_config():
    with open(CONFIG) as f:
        return json.load(f)

def main():
    config = load_config()
    
    # Monkey-patch the portfolio and midcap list into score_universe
    import score_universe as su
    
    # Set portfolio weights (normalize)
    portfolio = config.get("portfolio", {})
    total_equity = sum(
        h.get("shares", 0) * h.get("cost", 0) 
        for h in portfolio.values()
    )
    if total_equity > 0:
        su.CURRENT_PORTFOLIO = {
            t: (h["shares"] * h["cost"]) / total_equity
            for t, h in portfolio.items()
        }
    
    # Set midcap additions
    su.MIDCAP_ADDITIONS = config.get("midcap_additions", [])
    
    # Set visibility overrides
    for ticker, (score, reason) in config.get("visibility_overrides", {}).items():
        # Will be picked up in compute_visibility_score
        pass  # Already in the function via VISIBILITY_OVERRIDES dict
    
    # Run the main scoring pipeline
    su.main()
    
    # Now convert CSV output to JSON for the frontend
    import pandas as pd
    from datetime import datetime
    
    csv_path = ROOT / "data" / "scored_universe.csv"
    if not csv_path.exists():
        print("ERROR: scored_universe.csv not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # Build portfolio summary
    portfolio_summary = []
    for ticker, holdings in portfolio.items():
        row = df[df['ticker'] == ticker]
        if len(row) > 0:
            r = row.iloc[0]
            current_price = r.get('current_price', 0)
            cost = holdings.get('cost', 0)
            shares = holdings.get('shares', 0)
            gain_pct = ((current_price / cost) - 1) * 100 if cost > 0 else 0
            portfolio_summary.append({
                'ticker': ticker,
                'name': r.get('name', ticker),
                'shares': shares,
                'cost': cost,
                'current_price': round(current_price, 2),
                'market_value': round(current_price * shares, 2),
                'gain_pct': round(gain_pct, 1),
                'category': holdings.get('category', 'Core'),
                'composite': round(r.get('composite', 0), 1),
                'fundamental': round(r.get('fundamental', 0), 1),
                'technical': round(r.get('technical', 0), 1),
                'visibility': round(r.get('visibility', 0), 1),
            })
    
    total_equity_value = sum(p['market_value'] for p in portfolio_summary)
    cash = config.get("cash", 0)
    total_value = total_equity_value + cash
    
    # Build watchlist (top 30 non-portfolio)
    portfolio_tickers = set(portfolio.keys())
    watchlist_df = df[~df['ticker'].isin(portfolio_tickers)].head(30)
    
    watchlist = []
    for _, r in watchlist_df.iterrows():
        watchlist.append({
            'rank': int(r.get('rank', 0)) if 'rank' in r else int(_ + 1),
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': r.get('sector', ''),
            'category': r.get('category', ''),
            'mcap_B': round(r.get('mcap_B', 0), 1),
            'current_price': round(r.get('current_price', 0), 2),
            'entry_level': round(r.get('entry_level', 0), 2),
            'composite': round(r.get('composite', 0), 1),
            'fundamental': round(r.get('fundamental', 0), 1),
            'technical': round(r.get('technical', 0), 1),
            'visibility': round(r.get('visibility', 0), 1),
            'corr_penalty': round(r.get('corr_penalty', 0), 1),
            'fwd_pe': r.get('fwd_pe', ''),
            'rev_growth_pct': r.get('rev_growth_pct', ''),
            'gross_margin_pct': r.get('gross_margin_pct', ''),
            'rsi': r.get('rsi', ''),
            'max_corr': r.get('max_corr', ''),
            'max_corr_with': r.get('max_corr_with', ''),
        })
    
    # Full universe stats
    universe_stats = {
        'total': len(df),
        'median_score': round(df['composite'].median(), 1),
        'mean_score': round(df['composite'].mean(), 1),
        'max_score': round(df['composite'].max(), 1),
        'min_score': round(df['composite'].min(), 1),
        'sectors': df['sector'].value_counts().to_dict(),
        'categories': df['category'].value_counts().to_dict(),
    }
    
    # Assemble output
    output = {
        'updated': datetime.now().isoformat(),
        'portfolio': {
            'holdings': portfolio_summary,
            'total_equity': round(total_equity_value, 2),
            'cash': cash,
            'total_value': round(total_value, 2),
            'equity_pct': round(total_equity_value / total_value * 100, 1) if total_value > 0 else 0,
            'cash_pct': round(cash / total_value * 100, 1) if total_value > 0 else 0,
        },
        'watchlist': watchlist,
        'universe': universe_stats,
    }
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    with open(OUTPUT, 'w') as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    
    print(f"\nJSON output: {OUTPUT}")
    print(f"Portfolio: {len(portfolio_summary)} positions, ${total_equity_value:,.0f} equity + ${cash:,.0f} cash = ${total_value:,.0f}")
    print(f"Watchlist: {len(watchlist)} candidates")


if __name__ == "__main__":
    main()
