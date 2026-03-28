from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Alliance:
    name: str  # "UDF" / "LDF" / "OTHER" / "UNKNOWN"


def party_to_alliance(party: str) -> Alliance:
    """
    Heuristic mapping for explanatory text only.

    Alliances in Kerala are dynamic, so this does NOT claim historical accuracy.
    It just tags likely alliance families based on party labels from your CSVs.
    """
    p = (party or "").strip().upper()
    if not p or p in {"NOTA", "UNKNOWN"}:
        return Alliance("OTHER")

    # Direct alliance name check (e.g. from aggregated predictor)
    if p in {"LDF", "UDF", "NDA"}:
        return Alliance(p)

    # LDF family (CPI/CPIM variants, JD(S), and Kerala Congress (Mani) factions)
    if "CPI(M)" in p or "CPI[M]" in p or "CPI [M]" in p:
        return Alliance("LDF")
    if p in {"CPI(M)", "CPI[M]", "CPI [M]", "CPI[M]"}:
        return Alliance("LDF")
    if p == "CPI":
        return Alliance("LDF")
    if "JD(S)" in p or p == "JDS" or "JDS" in p or p == "LJD":
        return Alliance("LDF")
    if p in {"NCP", "NCP-SP", "RJD", "C(S)", "CONG(S)", "NSC"}:
        # NCP-SP and RJD are LDF components; NSC is LDF-aligned
        return Alliance("LDF")
    if p in {"KC(B)", "KCB"}:
        # Kerala Congress (B) - LDF component
        return Alliance("LDF")
    if p in {"INL"}:
        # Indian National League - LDF ally in 2026
        return Alliance("LDF")
    if p in {"ISJD"}:
        # Indian Secular Justice Democratic - LDF ally
        return Alliance("LDF")
    if p == "RSP(L)" or p == "RSP-L":
        # RSP (Leftist) - LDF component
        return Alliance("LDF")
    if p == "KEC(M)" or "(M)" in p:
        # e.g. KC(M), KEC(M) -> Mani faction family (LDF in 2026)
        return Alliance("LDF")
    if "RSP" in p:
        # RSP - UDF component historically, but varies; use UDF as default
        return Alliance("UDF")

    # UDF family
    if "INC" in p:
        return Alliance("UDF")
    if "IUML" in p or "MLKSC" in p:
        return Alliance("UDF")
    if p in {"JD(U)", "JDU", "JD (U)"}:
        # JD(U) supports UDF
        return Alliance("UDF")
    if p in {"SJ(D)", "SJD", "SJ [D]", "SJ[D]"}:
        # Socialist Janata Democratic - UDF ally previously
        return Alliance("UDF")
    if p == "KEC" or "KERALA CONGRESS" in p:
        # Covers both (Joseph) and (Jacob) factions.
        return Alliance("UDF")
    if "KC(J)" in p or "KC (J)" in p or "KC(J)" in party:
        return Alliance("UDF")
    if "SDPI" in p:
        return Alliance("UDF")
    if p in {"RMPI", "RMPOI"}:
        # Revolutionary Marxist Party of India - UDF ally
        return Alliance("UDF")
    if p in {"CMP"}:
        # Congress (M) Party - UDF ally
        return Alliance("UDF")

    # NDA family
    if p in {"BJP", "BDJS", "TTP"}:
        # TTP (Twenty20/Trotskyist) contests under NDA in 2026
        return Alliance("NDA")

    # AAP and other third-party entrants
    if p in {"AAP"}:
        return Alliance("OTHER")

    return Alliance("OTHER")


def format_alliance_tag(party: str) -> str:
    a = party_to_alliance(party)
    if a.name == "UNKNOWN":
        return "Alliance unclear (RSP/JSS/regionally shifting)"
    return a.name

