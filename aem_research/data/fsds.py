"""SEC Financial Statement Data Sets: the XBRL facts behind the fundamentals.

Each quarterly zip holds sub.txt (one row per filing: CIK, form, period, filing date,
SIC, filer status) and num.txt (one row per reported value: tag, period end `ddate`,
number of quarters `qtrs`, unit, value). Comparative columns are reported again in
every later filing, so the prior-year values needed for growth rates come from the
same filing that reports the current ones.

The extract keeps 10-K and 10-Q filings and only the tags listed in TAG_GROUPS,
non-dimensional and consolidated, in USD or shares. Share counts reported per class
(a dimension) are kept separately so classes can be summed.

The groups after "public_float" serve Paper A (operating income, depreciation and
amortization, pre-tax income, interest: the EBITDA behind a take-private's entry and
exit multiples); the AEM features ignore them. SCHEMA fingerprints the tag list, so a
cached quarter extracted with a different list is rebuilt instead of silently lacking
tags.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .sec_datasets import FSDS_URLS, parse_dates, process, quarter_items, read_member, to_int

PERIODIC_FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A", "10-KT", "10-KT/A", "10-QT", "10-QT/A")

# Group -> tags in priority order (the first tag a company reports for a period wins).
TAG_GROUPS: dict[str, tuple[str, ...]] = {
    "ocf": ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets",
              "PaymentsForCapitalImprovements", "PaymentsToAcquireOtherPropertyPlantAndEquipment"),
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "SalesRevenueGoodsNet", "SalesRevenueServicesNet", "RevenuesNetOfInterestExpense",
                "InterestAndDividendIncomeOperating"),
    "cash": ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "CashAndDueFromBanks", "Cash"),
    "cash_sti": ("CashCashEquivalentsAndShortTermInvestments",),
    "sti": ("ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "AvailableForSaleSecuritiesCurrent", "HeldToMaturitySecuritiesCurrent"),
    "debt_total": ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
                   "DebtInstrumentCarryingAmount"),
    "debt_noncurrent": ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
                        "LongTermNotesPayable", "ConvertibleNotesPayable"),
    "debt_current": ("LongTermDebtCurrent", "DebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
                     "ConvertibleNotesPayableCurrent", "NotesPayableCurrent"),
    "short_borrowings": ("ShortTermBorrowings", "LinesOfCreditCurrent", "CommercialPaper"),
    "shares_wad": ("WeightedAverageNumberOfDilutedSharesOutstanding",
                   "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
                   "WeightedAverageNumberOfSharesOutstandingBasic"),
    "shares_dei": ("EntityCommonStockSharesOutstanding",),
    "public_float": ("EntityPublicFloat",),
    # Paper A: EBITDA = operating income + depreciation and amortization; interest for coverage
    "operating_income": ("OperatingIncomeLoss",),
    "dna": ("DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization", "Depreciation"),
    "amortization": ("AmortizationOfIntangibleAssets",),
    "pretax_income": ("IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
    "interest_expense": ("InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"),
}
TAGS = {t: (g, i) for g, tags in TAG_GROUPS.items() for i, t in enumerate(tags)}
UNITS = {"USD", "shares"}
CLASS_SUMMED = {"EntityCommonStockSharesOutstanding"}
SCHEMA = hashlib.sha1(json.dumps({"tags": TAG_GROUPS, "forms": PERIODIC_FORMS, "units": sorted(UNITS)},
                                 sort_keys=True).encode()).hexdigest()[:12]


def _ym(dates: pd.Series) -> pd.Series:
    """Month index (year * 12 + month - 1); FSDS dates are already rounded to month-ends."""
    d = pd.to_datetime(dates)
    return (d.dt.year * 12 + d.dt.month - 1).astype("Int64")


def extract_quarter(zf: zipfile.ZipFile) -> dict[str, pd.DataFrame]:
    sub = read_member(zf, "sub", ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed",
                                  "afs", "fye", "instance", "prevrpt", "countryba"])
    sub = sub[sub["form"].isin(PERIODIC_FORMS)].copy()
    keep = set(sub["adsh"])

    def only_wanted(chunk: pd.DataFrame) -> pd.DataFrame:
        ok = chunk["tag"].isin(TAGS.keys()) & chunk["uom"].isin(UNITS) & chunk["adsh"].isin(keep)
        if "coreg" in chunk.columns:
            ok &= chunk["coreg"].isna()
        return chunk[ok]

    num = read_member(zf, "num", ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg",
                                  "value"], chunk_filter=only_wanted)
    num["dim"] = num["segments"].notna()
    num = num[~num["dim"] | num["tag"].isin(CLASS_SUMMED)]
    num["value"] = pd.to_numeric(num["value"], errors="coerce")
    num = num.dropna(subset=["value"])

    sub["cik"] = to_int(sub["cik"])
    sub["sic"] = to_int(sub["sic"])
    sub["filed"] = parse_dates(sub["filed"])
    sub["period"] = parse_dates(sub["period"])
    sub["prevrpt"] = sub["prevrpt"].astype(str).str.strip().eq("1")
    subs = sub[["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "afs", "fye",
                "instance", "prevrpt", "countryba"]].reset_index(drop=True)

    f = num.merge(subs[["adsh", "cik", "filed"]], on="adsh", how="inner")
    f["ym"] = _ym(parse_dates(f["ddate"]))
    f = f.dropna(subset=["cik", "ym", "filed"]).reset_index(drop=True)
    facts = pd.DataFrame({
        "cik": f["cik"].astype("int64"),
        "tag": f["tag"].astype("category"),
        "ym": f["ym"].astype("int32"),
        "qtrs": pd.to_numeric(f["qtrs"], errors="coerce").fillna(-1).astype("int16"),
        "value": f["value"].astype("float64"),
        "dim": f["dim"].astype(bool),
        "filed": f["filed"],
        "adsh": f["adsh"].astype(str),
    })
    return {"subs": subs, "facts": facts.reset_index(drop=True)}


def load(sec_dir: str | Path, start_year: int, end_year: int, *, client=None, offline: bool = False,
         today: date | None = None, log=print, local_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(subs, facts, stats) for every complete quarter, cached per quarter on Drive."""
    items = quarter_items(FSDS_URLS, start_year, end_year, today)
    tables, stats = process(items, Path(sec_dir) / "fsds", "fsds", extract_quarter, client=client,
                            offline=offline, log=log, local_dir=local_dir, schema=SCHEMA)
    subs = tables.get("subs", pd.DataFrame(columns=["adsh", "cik", "form", "period", "fp", "filed", "sic"]))
    facts = tables.get("facts", pd.DataFrame(columns=["cik", "tag", "ym", "qtrs", "value", "dim", "filed", "adsh"]))
    if len(facts):
        facts["tag"] = facts["tag"].astype(str).astype("category")
        facts = facts.drop_duplicates(["adsh", "tag", "ym", "qtrs", "dim", "value"]).reset_index(drop=True)
    if len(subs):
        subs = subs.drop_duplicates("adsh", keep="last").reset_index(drop=True)
    return subs, facts, stats
