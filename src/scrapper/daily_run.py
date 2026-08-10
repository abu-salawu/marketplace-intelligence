"""
src/scraper/daily_run.py
========================
Daily orchestrator for the marketplace price panel.

Runs each platform scraper, writes per-platform raw CSVs, then folds today's
observations into two processed artefacts:

  data/processed/price_series.csv   -- daily aggregate per (date, platform,
                                       category). One row per group per day.
  data/processed/product_panel.csv  -- product-level long panel, one row per
                                       (date, platform, product_id).

FAILURE POLICY
--------------
This script exits non-zero when it collects nothing. A scraper that silently
writes an empty file produces a series with invisible gaps -- days that look
like real observations but contain nothing. Given that this panel feeds the
Granger and fusion analyses, a loud failure is strictly better than a quiet
one. Partial success (some platforms up, others down) is permitted but
recorded in the manifest.

Usage:
  python -m src.scraper.daily_run
  python -m src.scraper.daily_run --pages 3 --platforms jumia
  python -m src.scraper.daily_run --min-rows 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("daily_run")

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
SERIES_CSV = PROC_DIR / "price_series.csv"
PANEL_CSV = PROC_DIR / "product_panel.csv"
MANIFEST_DIR = Path("data/manifests")

CATEGORIES = ["electronics", "generators", "food"]

SERIES_COLS = [
    "date", "platform", "category",
    "n_products", "n_unique_ids",
    "median_price_ngn", "mean_price_ngn", "p25_price_ngn", "p75_price_ngn",
    "mean_discount_pct", "total_reviews", "scraped_at_utc",
]

PANEL_COLS = [
    "date", "platform", "category", "product_id", "product_name",
    "price_ngn", "old_price_ngn", "discount_pct",
    "rating", "review_count", "scraped_at_utc",
]


# --------------------------------------------------------------------------
# Platform registry
#
# Each entry maps a platform name to a callable(category, pages) -> list[dict]
# returning rows in the shared SCHEMA. Imports are lazy and individually
# guarded so that one broken module does not take down the whole run.
# --------------------------------------------------------------------------

def _load_scrapers(wanted: list[str]) -> dict:
    scrapers: dict[str, callable] = {}

    if "jumia" in wanted:
        try:
            from .jumia_scraper import scrape_category as jumia_scrape
            scrapers["jumia"] = jumia_scrape
        except Exception as e:
            log.error("Could not load jumia scraper: %s", e)

    # Konga and Temu share a module. Adjust the imported names here if the
    # functions in konga_temu_scraper.py are called something else.
    if "konga" in wanted or "temu" in wanted:
        try:
            from . import konga_temu_scraper as kt

            if "konga" in wanted:
                fn = getattr(kt, "scrape_konga", None) or getattr(
                    kt, "scrape_category", None
                )
                if fn:
                    scrapers["konga"] = fn
                else:
                    log.error("konga_temu_scraper exposes no usable konga entry point")

            if "temu" in wanted:
                fn = getattr(kt, "scrape_temu", None)
                if fn:
                    scrapers["temu"] = fn
                else:
                    log.error("konga_temu_scraper exposes no usable temu entry point")
        except Exception as e:
            log.error("Could not load konga/temu scraper: %s", e)

    return scrapers


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(int(q * (len(s) - 1)), len(s) - 1)
    return round(s[idx], 2)


def aggregate(rows: list[dict], scraped_at: str) -> list[dict]:
    """Collapse product rows to one aggregate row per (date, platform, category).

    Uses the MEDIAN as the headline price. The mean is retained as a secondary
    column but should not be the analysis variable: Jumia listings include
    accessories priced at a few hundred naira alongside laptops priced in the
    hundreds of thousands, and the mean tracks that mix rather than price.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["date"], r["platform"], r["category"])
        groups.setdefault(key, []).append(r)

    out = []
    for (d, plat, cat), grp in sorted(groups.items()):
        prices = [float(r["price_ngn"]) for r in grp if r.get("price_ngn") is not None]
        if not prices:
            log.warning("Group %s/%s/%s had no usable prices", d, plat, cat)
            continue
        discounts = [
            float(r["discount_pct"]) for r in grp if r.get("discount_pct") is not None
        ]
        reviews = [int(r.get("review_count") or 0) for r in grp]
        ids = {r.get("product_id") for r in grp if r.get("product_id")}

        out.append({
            "date": d,
            "platform": plat,
            "category": cat,
            "n_products": len(grp),
            "n_unique_ids": len(ids),
            "median_price_ngn": round(median(prices), 2),
            "mean_price_ngn": round(sum(prices) / len(prices), 2),
            "p25_price_ngn": _quantile(prices, 0.25),
            "p75_price_ngn": _quantile(prices, 0.75),
            "mean_discount_pct": round(sum(discounts) / len(discounts), 2)
            if discounts else None,
            "total_reviews": sum(reviews),
            "scraped_at_utc": scraped_at,
        })
    return out


def to_panel(rows: list[dict], scraped_at: str) -> list[dict]:
    """Product-level rows, deduplicated on (date, platform, product_id)."""
    seen: dict[tuple, dict] = {}
    for r in rows:
        pid = r.get("product_id") or f"noid::{r.get('product_name','')[:80]}"
        key = (r["date"], r["platform"], pid)
        if key in seen:
            continue
        seen[key] = {
            "date": r["date"],
            "platform": r["platform"],
            "category": r["category"],
            "product_id": pid,
            "product_name": r.get("product_name"),
            "price_ngn": r.get("price_ngn"),
            "old_price_ngn": r.get("old_price_ngn"),
            "discount_pct": r.get("discount_pct"),
            "rating": r.get("rating"),
            "review_count": r.get("review_count"),
            "scraped_at_utc": scraped_at,
        }
    return list(seen.values())


# --------------------------------------------------------------------------
# Idempotent append
# --------------------------------------------------------------------------

def upsert_csv(path: Path, new_rows: list[dict], cols: list[str], keys: list[str]) -> int:
    """Append new_rows to path, replacing any existing rows with matching keys.

    Idempotent by key, so a re-run on the same day corrects that day's entry
    instead of duplicating it. Duplicated dates would silently break the
    Granger lag structure, which assumes one observation per period.
    """
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    new_keys = {tuple(str(r.get(k, "")) for k in keys) for r in new_rows}
    kept = [r for r in existing if tuple(str(r.get(k, "")) for k in keys) not in new_keys]
    replaced = len(existing) - len(kept)

    combined = kept + [{c: r.get(c) for c in cols} for r in new_rows]
    combined.sort(key=lambda r: tuple(str(r.get(k, "")) for k in keys))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(combined)

    log.info(
        "%s: +%d rows (%d replaced), %d total",
        path.name, len(new_rows), replaced, len(combined),
    )
    return len(combined)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Daily marketplace scrape orchestrator")
    ap.add_argument("--pages", type=int, default=5)
    ap.add_argument(
        "--platforms", default="jumia,konga,temu",
        help="comma-separated subset to run",
    )
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument(
        "--min-rows", type=int, default=1,
        help="exit non-zero if fewer than this many product rows are collected",
    )
    args = ap.parse_args()

    wanted_platforms = [p.strip() for p in args.platforms.split(",") if p.strip()]
    wanted_categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    scraped_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    today = str(date.today())

    scrapers = _load_scrapers(wanted_platforms)
    if not scrapers:
        log.error("No scrapers could be loaded. Check module names and imports.")
        return 1

    all_rows: list[dict] = []
    manifest = {
        "run_date": today,
        "scraped_at_utc": scraped_at,
        "pages_per_query": args.pages,
        "platforms": {},
    }

    for plat, fn in scrapers.items():
        plat_total = 0
        for cat in wanted_categories:
            try:
                rows = fn(cat, args.pages) or []
            except Exception as e:
                log.exception("%s/%s raised: %s", plat, cat, e)
                rows = []
            # Defensive: not every scraper may stamp these.
            for r in rows:
                r.setdefault("platform", plat)
                r.setdefault("category", cat)
                r.setdefault("date", today)
            log.info("%s/%s -> %d rows", plat, cat, len(rows))
            manifest["platforms"].setdefault(plat, {})[cat] = len(rows)
            plat_total += len(rows)
            all_rows.extend(rows)

            if rows:
                try:
                    from .jumia_scraper import save as save_raw
                    save_raw(rows, f"{plat}_{cat}", str(RAW_DIR))
                except Exception as e:
                    log.warning("Raw save failed for %s/%s: %s", plat, cat, e)

        if plat_total == 0:
            log.error("PLATFORM DOWN: %s returned zero rows across all categories", plat)

    manifest["total_rows"] = len(all_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / f"run_{today}.json").write_text(json.dumps(manifest, indent=2))

    if len(all_rows) < args.min_rows:
        log.error(
            "FAILING: collected %d rows, below --min-rows=%d. "
            "Nothing written to data/processed/ -- a gap is more honest than "
            "a fabricated observation.",
            len(all_rows), args.min_rows,
        )
        return 1

    upsert_csv(SERIES_CSV, aggregate(all_rows, scraped_at), SERIES_COLS,
               keys=["date", "platform", "category"])
    upsert_csv(PANEL_CSV, to_panel(all_rows, scraped_at), PANEL_COLS,
               keys=["date", "platform", "product_id"])

    live = [p for p, v in manifest["platforms"].items() if sum(v.values()) > 0]
    log.info("Done. %d rows from %d/%d platforms: %s",
             len(all_rows), len(live), len(scrapers), ", ".join(live) or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
