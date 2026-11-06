"""EODHD gap-fill (stub – only built if the coverage audit says it adds value).

Why it might be needed: EODHD keeps the price history of delisted US tickers (its own
example: Twitter's series still runs to its last trading day in October 2022). The
EOD Historical All World plan is US$19.99 a month with 30+ years of history; its
pricing page (checked Nov 6, 2026) offers students 50% off for 12 months, after which
the full price applies automatically – cancel after the one month you need. Tickers
delisted before 2018 have end-of-day prices only, later ones fundamentals as well.

Planned use, if the audit's verdict is "recommended":
  1. Read the API key from a Colab secret EODHD_API_KEY (never from code).
  2. Pull the list of delisted US tickers (GET /api/exchange-symbol-list/US?delisted=1),
     then end-of-day history only for tickers the SimFin cache and Stooq miss (one
     call per ticker, cached on Drive; paid plans allow 100,000 calls a day).
  3. Merge with the same canonical-ticker and date rules as the other sources.

Before paying: confirm the licence lets you keep using downloaded data after the
month ends and publish derived statistics. Raw prices never go into the public repo.
"""
from __future__ import annotations


def fetch_missing(*_args, **_kwargs):
    raise NotImplementedError(__doc__)
