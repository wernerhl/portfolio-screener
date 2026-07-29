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

    # Broken-base cap + visibility governance knobs (plan 2.3 / 2.5)
    bb = config.get("broken_base", {})
    su.BROKEN_BASE_GAP_PCT  = float(bb.get("gap_pct",  su.BROKEN_BASE_GAP_PCT))
    su.BROKEN_BASE_RSI      = float(bb.get("rsi",      su.BROKEN_BASE_RSI))
    su.BROKEN_BASE_TECH_CAP = float(bb.get("tech_cap", su.BROKEN_BASE_TECH_CAP))
    su.VISIBILITY_FALLBACK_CAP = float(config.get("visibility_fallback_cap",
                                                  su.VISIBILITY_FALLBACK_CAP))
    
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
    
    # Build watchlist — INCLUDE held names (you can always add to a position).
    # Held names stay in the ranking, flagged `held:true` so the frontend can
    # tag them as add-to-position candidates rather than hiding them. Top 40 so
    # the held names don't crowd out fresh ideas.
    portfolio_tickers = set(portfolio.keys())
    watchlist_df = df.head(40)

    import math

    def _num(x, nd=3):
        """Coerce to a rounded float, or None for NaN/empty/None — typed
        numeric fields are a number or null, never '' or NaN."""
        try:
            if x is None or (isinstance(x, str) and not x.strip()):
                return None
            v = float(x)
            return None if (math.isnan(v) or math.isinf(v)) else round(v, nd)
        except (TypeError, ValueError):
            return None

    watchlist = []
    for _, r in watchlist_df.iterrows():
        watchlist.append({
            'rank': int(r.get('rank', 0)) if 'rank' in r else int(_ + 1),
            'ticker': r['ticker'],
            'held': bool(r.get('in_portfolio', False)) or (r['ticker'] in portfolio_tickers),
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
            'broken_base': bool(r.get('broken_base', False)),
            # None = correlation unavailable (visible impairment) — never 0-filled
            'corr_penalty': _num(r.get('corr_penalty'), 1),
            'corr_status': r.get('corr_status', 'ok') if isinstance(r.get('corr_status'), str) else 'ok',
            'portfolio_corr': _num(r.get('portfolio_corr')),
            'max_corr': _num(r.get('max_corr')),
            'max_corr_with': (r.get('max_corr_with') if isinstance(r.get('max_corr_with'), str) and r.get('max_corr_with') else None),
            # Typed fields: fwd_pe is a P/E ratio or null; fcf_yield_pct is a
            # percent or null. Never fall back across semantic types.
            'fwd_pe': _num(r.get('fwd_pe'), 1),
            'fcf_yield_pct': _num(r.get('fcf_yield_pct'), 1),
            'data_flags': r.get('data_flags', '') if isinstance(r.get('data_flags'), str) else '',
            'rev_growth_pct': r.get('rev_growth_pct', ''),
            'gross_margin_pct': r.get('gross_margin_pct', ''),
            'rsi': r.get('rsi', ''),
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
    
    # ── CI tripwires ─────────────────────────────────────────────────────
    # Fail the build LOUDLY rather than publish a silently impaired board.
    # July 2026 lesson: the correlation column died to all-zeros for weeks and
    # nothing noticed, because nothing asserted anything.
    def validate_outputs(df, watchlist):
        # 1) Correlation column must be ALIVE. Against a 4-megacap tech book,
        #    something in a 500+ name universe must correlate > 0.3 (market
        #    beta alone guarantees it). All null/zero → computation dead.
        mc = pd.to_numeric(df.get('max_corr'), errors='coerce')
        pc = pd.to_numeric(df.get('portfolio_corr'), errors='coerce')
        n_alive = int(((mc.abs() > 0.3) | (pc.abs() > 0.3)).sum())
        if n_alive == 0:
            raise SystemExit(
                "CI FAIL: correlation column is dead — no name in the universe "
                "shows |corr| > 0.3 vs the portfolio. The computation is broken "
                "(this exact failure shipped silently in July 2026); refusing to publish.")
        # 2) Typed ranges (defense in depth — score_universe nulls at source)
        for e in watchlist:
            if e['fwd_pe'] is not None and not (3 <= e['fwd_pe'] <= 150):
                raise SystemExit(f"CI FAIL: fwd_pe out of range for {e['ticker']}: {e['fwd_pe']}")
            if e['fcf_yield_pct'] is not None and not (-5 <= e['fcf_yield_pct'] <= 25):
                raise SystemExit(f"CI FAIL: fcf_yield_pct out of range for {e['ticker']}: {e['fcf_yield_pct']}")
        # 3) Structural floors
        if len(watchlist) < 30:
            raise SystemExit(f"CI FAIL: watchlist collapsed to {len(watchlist)} rows")
        if len(df) < 400:
            raise SystemExit(f"CI FAIL: scored universe collapsed to {len(df)} rows "
                             "(535 expected — Wikipedia-fallback-style regression?)")
        print(f"  validate_outputs: OK  (corr alive on {n_alive} names, "
              f"{len(watchlist)} watchlist rows, {len(df)} scored)")

    validate_outputs(df, watchlist)

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

    # Sanitize NaN/Infinity → null. Python's json.dump emits literal `NaN`
    # (valid for Python, INVALID for the browser's JSON.parse — it throws
    # "Unexpected token N" and the page renders "No data found"). This bit
    # when the universe grew to 535 + held names were included: names with
    # no correlation match carry max_corr_with = NaN. allow_nan=False makes
    # any future NaN a loud CI failure instead of a silently-broken page.
    import math
    def _clean(o):
        if isinstance(o, dict):  return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):  return [_clean(v) for v in o]
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
        if isinstance(o, (np.floating,)):
            x = float(o); return None if (math.isnan(x) or math.isinf(x)) else x
        return o

    with open(OUTPUT, 'w') as f:
        json.dump(_clean(output), f, indent=2, cls=NpEncoder, allow_nan=False)
    
    print(f"\nJSON output: {OUTPUT}")
    print(f"Portfolio: {len(portfolio_summary)} positions, ${total_equity_value:,.0f} equity + ${cash:,.0f} cash = ${total_value:,.0f}")
    print(f"Watchlist: {len(watchlist)} candidates")

    # Daily as-published vintage (plan 2.7a): immutable copies under
    # data/vintages/<date>/ — the tournament's published-vintage convention.
    # This is what makes the pre-registered forward validation (Spearman IC on
    # frozen vintages, first evaluation Feb 2027) possible at all.
    import shutil
    vint = ROOT / "data" / "vintages" / datetime.now().strftime("%Y-%m-%d")
    vint.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT, vint / "scores.json")
    for fn in ("scored_universe.csv", "watchlist_top30.csv"):
        src = ROOT / "data" / fn
        if src.exists():
            shutil.copy2(src, vint / fn)
    print(f"Vintage archived: {vint.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
