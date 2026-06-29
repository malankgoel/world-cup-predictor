"""Replace EA-imputed player talent with calibrated Transfermarkt market value.

build_squads.py rates 70% of players from EA SPORTS FC and fills the rest with
a flat floor (talent_se == 8.0). That floor is biased: EA does not license the
Gulf, African, and Liga MX leagues, so whole squads (Qatar, Jordan, Uzbekistan,
Iran, ...) collapse toward 62 regardless of real quality. This step swaps that
floor for a market-value estimate, calibrated onto the EA talent scale.

Calibration: fit  talent ~ b0 + b1*log10(value_eur) + b2*age  on the players
who have BOTH a real EA rating and a Transfermarkt value (the overlap). Then,
for each EA-imputed player who has a market value, predict talent from that fit
and set talent_se to the calibration residual std (typically tighter than the
flat 8.0). Imputed players still missing a market value keep the old floor.

Run order:
    scripts/build_squads.py            # writes squads.csv (EA + floor)
    scripts/download_transfermarkt.py  # writes data/raw/transfermarkt.json
    scripts/apply_transfermarkt.py     # this script; rewrites squads.csv

The original EA-only file is backed up to data/input/squads_ea.csv on first run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SQUADS = ROOT / "data" / "input" / "squads.csv"
BACKUP = ROOT / "data" / "input" / "squads_ea.csv"
TM = ROOT / "data" / "raw" / "transfermarkt.json"
REPORT = ROOT / "data" / "input" / "squads_transfermarkt_report.txt"

IMPUTED_SE = 8.0          # marker set by build_squads.py for floored players
TALENT_FLOOR, TALENT_CEILING = 50.0, 92.0
MIN_OVERLAP = 30          # below this the calibration is not trustworthy


def load_market_values() -> pd.DataFrame:
    records = json.loads(TM.read_text())
    frame = pd.DataFrame(records)
    frame["value_eur_asof"] = pd.to_numeric(frame["value_eur_asof"], errors="coerce")
    frame["log_value"] = np.log10(frame["value_eur_asof"].where(frame["value_eur_asof"] > 0))
    return frame[["team", "player", "value_eur_asof", "log_value"]]


def fit_calibration(overlap: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """Least-squares talent ~ [log_value, age, 1]; return coefs, residual std, R^2."""
    design = np.column_stack(
        [overlap["log_value"], overlap["age"], np.ones(len(overlap))]
    )
    target = overlap["talent"].to_numpy(float)
    coefs, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefs
    residuals = target - predicted
    resid_std = float(residuals.std(ddof=design.shape[1]))
    ss_res = float((residuals**2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return coefs, resid_std, r2


def predict_talent(coefs: np.ndarray, log_value: float, age: float) -> float:
    value = coefs[0] * log_value + coefs[1] * age + coefs[2]
    return float(np.clip(value, TALENT_FLOOR, TALENT_CEILING))


def main() -> None:
    squads = pd.read_csv(SQUADS)
    if not BACKUP.exists():
        squads.to_csv(BACKUP, index=False)

    market = load_market_values()
    squads = squads.merge(market, on=["team", "player"], how="left")
    squads["talent"] = pd.to_numeric(squads["talent"], errors="coerce")
    squads["talent_se"] = pd.to_numeric(squads["talent_se"], errors="coerce")
    squads["age"] = pd.to_numeric(squads["age"], errors="coerce")
    # Players missing an age get their squad's median so calibration can use them.
    squads["age"] = squads["age"].fillna(
        squads.groupby("team")["age"].transform("median")
    )

    imputed = squads["talent_se"] == IMPUTED_SE
    has_value = squads["log_value"].notna() & squads["age"].notna()
    overlap = squads[~imputed & has_value & squads["talent"].notna()]
    if len(overlap) < MIN_OVERLAP:
        raise SystemExit(
            f"Only {len(overlap)} EA-rated players have a market value; "
            f"need >= {MIN_OVERLAP} to calibrate. Did download_transfermarkt.py run?"
        )

    coefs, resid_std, r2 = fit_calibration(overlap)
    new_se = float(round(max(4.0, resid_std), 1))

    replace = imputed & has_value
    squads.loc[replace, "talent"] = [
        predict_talent(coefs, lv, ag)
        for lv, ag in zip(
            squads.loc[replace, "log_value"], squads.loc[replace, "age"], strict=False
        )
    ]
    squads.loc[replace, "talent_se"] = new_se

    still_floored = int((imputed & ~has_value).sum())
    report = [
        f"calibration: talent = {coefs[0]:.2f}*log10(value) "
        f"+ {coefs[1]:.3f}*age + {coefs[2]:.2f}",
        f"overlap (EA-rated with market value): {len(overlap)} players, "
        f"R^2 = {r2:.3f}, residual std = {resid_std:.2f}",
        f"imputed players re-estimated from market value: {int(replace.sum())} "
        f"(new talent_se = {new_se})",
        f"imputed players with no market value, kept on EA floor: {still_floored}",
    ]
    by_team = (
        squads[replace]
        .groupby("team")
        .agg(replaced=("player", "size"), mean_talent=("talent", "mean"))
        .sort_values("replaced", ascending=False)
    )
    report.append("\nmost-affected squads:")
    for team, row in by_team.head(12).iterrows():
        report.append(f"  {team}: {int(row['replaced'])} replaced, "
                      f"mean talent now {row['mean_talent']:.1f}")

    squads = squads.drop(columns=["value_eur_asof", "log_value"])
    squads.to_csv(SQUADS, index=False)
    REPORT.write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nrewrote {SQUADS.relative_to(ROOT)} (EA-only backup at {BACKUP.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
