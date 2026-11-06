"""Synthetic SEC data set zips in the published layouts (tab-delimited, header row).

Layouts follow the SEC readmes: Financial Statement Data Sets (sub.txt, num.txt with
lower-case headers, dates yyyymmdd), Insider Transactions Data Sets and Form 13F Data
Sets (upper-case headers, dates DD-MON-YYYY, .tsv members).
"""
from __future__ import annotations

import io
import zipfile

SUB_COLS = ["adsh", "cik", "name", "sic", "countryba", "stprba", "cityba", "zipba", "bas1", "bas2", "baph",
            "countryma", "stprma", "cityma", "zipma", "mas1", "mas2", "countryinc", "stprinc", "ein", "former",
            "changed", "afs", "wksi", "fye", "form", "period", "fy", "fp", "filed", "accepted", "prevrpt",
            "detail", "instance", "nciks", "aciks"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value", "footnote"]


def tsv(cols, rows) -> bytes:
    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols))
    return ("\n".join(lines) + "\n").encode("utf-8")


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def fsds_zip(subs: list[dict], nums: list[dict]) -> bytes:
    return zip_bytes({"sub.txt": tsv(SUB_COLS, subs), "num.txt": tsv(NUM_COLS, nums),
                      "readme.htm": b"<html></html>"})


def sub_row(adsh, cik, form, period, fp, filed, name="ACME CORP", sic=3571, fy=None, afs="2-ACC"):
    return {"adsh": adsh, "cik": cik, "name": name, "sic": sic, "countryba": "US", "afs": afs, "wksi": 0,
            "fye": "1231", "form": form, "period": period, "fy": fy or period[:4], "fp": fp, "filed": filed,
            "accepted": f"{filed[:4]}-{filed[4:6]}-{filed[6:]} 16:05:00.0", "prevrpt": 0, "detail": 1,
            "instance": f"acme-{period}.xml", "nciks": 1}


def num_row(adsh, tag, ddate, qtrs, value, uom="USD", segments=None, version="us-gaap/2012", coreg=None):
    return {"adsh": adsh, "tag": tag, "version": version, "ddate": ddate, "qtrs": qtrs, "uom": uom,
            "segments": segments, "coreg": coreg, "value": value}


INSIDER_SUB = ["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DATE_OF_ORIG_SUB", "NO_SECURITIES_OWNED",
               "NOT_SUBJECT_SEC16", "FORM3_HOLDINGS_REPORTED", "FORM4_TRANS_REPORTED", "DOCUMENT_TYPE",
               "ISSUERCIK", "ISSUERNAME", "ISSUERTRADINGSYMBOL", "REMARKS", "AFF10B5ONE"]
INSIDER_OWNER = ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE",
                 "RPTOWNER_TXT", "RPTOWNER_STREET1", "RPTOWNER_STREET2", "RPTOWNER_CITY", "RPTOWNER_STATE",
                 "RPTOWNER_ZIPCODE", "RPTOWNER_STATE_DESC", "FILE_NUMBER"]
INSIDER_TRANS = ["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "SECURITY_TITLE", "SECURITY_TITLE_FN", "TRANS_DATE",
                 "TRANS_DATE_FN", "DEEMED_EXECUTION_DATE", "DEEMED_EXECUTION_DATE_FN", "TRANS_FORM_TYPE",
                 "TRANS_CODE", "EQUITY_SWAP_INVOLVED", "EQUITY_SWAP_TRANS_CD_FN", "TRANS_TIMELINESS",
                 "TRANS_TIMELINESS_FN", "TRANS_SHARES", "TRANS_SHARES_FN", "TRANS_PRICEPERSHARE",
                 "TRANS_PRICEPERSHARE_FN", "TRANS_ACQUIRED_DISP_CD", "TRANS_ACQUIRED_DISP_CD_FN",
                 "SHRS_OWND_FOLWNG_TRANS", "SHRS_OWND_FOLWNG_TRANS_FN", "VALU_OWND_FOLWNG_TRANS",
                 "VALU_OWND_FOLWNG_TRANS_FN", "DIRECT_INDIRECT_OWNERSHIP", "DIRECT_INDIRECT_OWNERSHIP_FN",
                 "NATURE_OF_OWNERSHIP", "NATURE_OF_OWNERSHIP_FN"]


def sec_date(iso: str) -> str:
    """2013-05-15 -> 15-MAY-2013 (the insider and 13F data sets' layout)."""
    import datetime as _dt
    return _dt.date.fromisoformat(iso).strftime("%d-%b-%Y").upper()


def insider_zip(subs: list[dict], owners: list[dict], trans: list[dict], *, with_plan_flag: bool = True) -> bytes:
    sub_cols = INSIDER_SUB if with_plan_flag else [c for c in INSIDER_SUB if c != "AFF10B5ONE"]
    return zip_bytes({"SUBMISSION.tsv": tsv(sub_cols, subs), "REPORTINGOWNER.tsv": tsv(INSIDER_OWNER, owners),
                      "NONDERIV_TRANS.tsv": tsv(INSIDER_TRANS, trans),
                      "insider_transactions_metadata.json": b"{}"})


F13_SUB = ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"]
F13_INFO = ["ACCESSION_NUMBER", "INFOTABLE_SK", "NAMEOFISSUER", "TITLEOFCLASS", "CUSIP", "FIGI", "VALUE",
            "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "INVESTMENTDISCRETION", "OTHERMANAGER", "VOTING_AUTH_SOLE",
            "VOTING_AUTH_SHARED", "VOTING_AUTH_NONE"]
F13_COVER = ["ACCESSION_NUMBER", "REPORTCALENDARORQUARTER", "ISAMENDMENT", "AMENDMENTNO", "AMENDMENTTYPE"]


def f13_zip(subs: list[dict], infos: list[dict]) -> bytes:
    cover = [{"ACCESSION_NUMBER": s["ACCESSION_NUMBER"], "REPORTCALENDARORQUARTER": s["PERIODOFREPORT"],
              "ISAMENDMENT": "N" if s["SUBMISSIONTYPE"] == "13F-HR" else "Y"} for s in subs]
    return zip_bytes({"SUBMISSION.tsv": tsv(F13_SUB, subs), "INFOTABLE.tsv": tsv(F13_INFO, infos),
                      "COVERPAGE.tsv": tsv(F13_COVER, cover), "FORM13F_metadata.json": b"{}"})


class FakeClient:
    """Serves bytes for known URLs, 404 (None) for the rest; records every request."""

    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.requests: list[str] = []

    def get(self, url: str):
        self.requests.append(url)
        return self.files.get(url)
