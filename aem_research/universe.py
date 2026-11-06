"""Start universe filters.

Pre-registered in config/preregistration.yaml (universe): US common stock on NYSE,
NASDAQ or NYSE American that files 10-K and 10-Q reports with XBRL financials (which
leaves out ETFs, closed-end funds, ADRs and other foreign private issuers); no
pre-merger SPAC shells (SIC 6770); price >= $1; market cap between $100M and $2B
(AEM-2 v2.6); median 20-day dollar volume >= $100k – all measured with information
available before the start date.

The filters are applied in `features.assemble` (column `in_universe`), with market cap
from the latest cover-page share count filed before the start date. `READY` is the
code-level gate the notebook checks before the locked outcome phase.
"""
from __future__ import annotations

READY = True
