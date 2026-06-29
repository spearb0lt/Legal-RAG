"""
Indian Kanoon scraper — DISABLED BY DEFAULT.

WHY DISABLED:
    indiankanoon.org does not provide a public read API for bulk access. Their
    terms of service prohibit automated scraping for republication, and they
    actively rate-limit clients that scrape. Bulk scraping their judgments would
    risk (a) ToS violation, (b) IP-block of your machine, and (c) downstream
    legal exposure for your project if you redistribute the data.

WHAT TO DO INSTEAD:
    1. Use the IL-TUR HuggingFace dataset (data/ingest/fetch_il_tur.py) — it is
       a peer-reviewed, cite-able, redistribution-friendly source of Indian case
       texts.
    2. Use the Supreme Court of India e-SCR portal (https://judgments.ecourts.gov.in/scrsearch/)
       which permits manual download of individual judgments under its terms.
    3. Subscribe to Indian Kanoon's official API
       (https://api.indiankanoon.org/) if available to you — they offer a paid
       API for legitimate research use.
    4. Drop your own .txt judgments into data/raw/cases/ (one file per case,
       with the TITLE / DOC_TYPE / CITATION header pattern used in the seed files).

If you have explicit, written authorisation from Indian Kanoon to scrape, set
INDIAN_KANOON_AUTHORISED=true in your .env AND fill in the function below with
your custom logic. The default raises.
"""
from __future__ import annotations

import os
from typing import Iterable

from core.citation import Chunk


def scrape_indian_kanoon(query: str, max_docs: int = 20) -> list[Chunk]:
    if os.environ.get("INDIAN_KANOON_AUTHORISED", "").lower() != "true":
        raise PermissionError(
            "Indian Kanoon scraping is disabled. Set INDIAN_KANOON_AUTHORISED=true "
            "in your .env ONLY if you have authorisation. See module docstring."
        )
    raise NotImplementedError(
        "You have flagged INDIAN_KANOON_AUTHORISED=true but no scraping logic is "
        "implemented. Add your authorised crawl here, respecting robots.txt and "
        "rate-limit headers."
    )
