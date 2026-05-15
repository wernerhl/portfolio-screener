# Portfolio Screener

Private repo. 4-factor scoring model for equity universe screening. Runs monthly, deploys to GitHub Pages.

## Setup

```bash
# 1. Create private repo on GitHub
gh repo create portfolio-screener --private

# 2. Push this code
git init && git add . && git commit -m "init"
git remote add origin git@github.com:wernerhl/portfolio-screener.git
git push -u origin main

# 3. Enable Pages: Settings → Pages → Source: GitHub Actions

# 4. Run locally first
pip install yfinance pandas numpy
cd scripts && python score_universe.py && python build_json.py
# Open index.html in browser
```

## Update Portfolio

Edit `config.json`:
- `portfolio`: current holdings (ticker, shares, cost basis, category)
- `cash`: current cash balance
- `visibility_overrides`: manual backlog/RPO data from earnings calls
- `midcap_additions`: non-S&P 500 tickers to include in universe

## Scoring Model

| Factor | Weight | Source | What It Measures |
|--------|--------|--------|-----------------|
| Fundamental | 0-25 | Yahoo Finance | FCF yield, revenue growth, margins, ROIC |
| Technical | 0-25 | Yahoo Finance | 200-DMA distance, RSI, relative strength, support |
| Visibility | 0-25 | Manual + sector | Contracted backlog, RPO, revenue predictability |
| Correlation | 0 to -10 | Price data | Penalty for overlap with existing portfolio |

**Composite = Fundamental + Technical + Visibility + Correlation Penalty** (0-75 range)

## Monthly Workflow

1. Update `config.json` with current holdings and cash
2. Run `python scripts/score_universe.py && python scripts/build_json.py`
3. Review top 30 watchlist — apply discretionary judgment
4. Write entry tickets for selected names (see Portfolio OS)
5. Set limit orders at entry levels
6. Push to trigger GitHub Pages deploy
