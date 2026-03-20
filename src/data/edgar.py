"""EDGAR EFTS API client — fetches EX-10.x contract exhibits from SEC filings.

Mirrors the logic from legal-partner's EdgarImportService.java but outputs
raw contract text for the fine-tuning pipeline.

SEC rate limit: max 10 requests/second. We use 200ms delay between requests.
SEC requires a descriptive User-Agent with contact email.
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.data.schema import ContractType, EDGAR_QUERIES, RawContract

logger = logging.getLogger(__name__)

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
USER_AGENT = "LegalFinetune Research Tool legal-finetune/1.0 (contact@legalpartner.app)"
REQUEST_DELAY = 0.2  # seconds between requests (SEC rate limit)
MAX_DOC_SIZE = 512_000  # 500KB max per document


def search_edgar(
    query: str,
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    max_results: int = 100,
) -> list[dict]:
    """Search EDGAR EFTS for EX-10.x filings matching the query.

    Returns a list of hit dicts with keys: accession, filename, entity, cik, date, url.
    """
    headers = {"User-Agent": USER_AGENT}
    hits = []
    offset = 0
    page_size = min(max_results, 100)

    while len(hits) < max_results:
        params = {
            "q": query,
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "from": offset,
            "size": page_size,
        }
        try:
            resp = requests.get(EFTS_SEARCH_URL, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("EDGAR search failed at offset %d: %s", offset, e)
            break

        raw_hits = data.get("hits", {}).get("hits", [])
        if not raw_hits:
            break

        for h in raw_hits:
            src = h.get("_source", {})
            file_type = src.get("file_type", "")
            if not file_type.startswith("EX-10"):
                continue

            # Extract CIK and build URL
            ciks = src.get("ciks", [])
            cik = ciks[0] if ciks else ""
            accession = src.get("file_num", "") or h.get("_id", "")
            filename = src.get("file_name", "")
            display_names = src.get("display_names", [])
            entity = display_names[0].split("(")[0].strip() if display_names else "Unknown"
            date = src.get("file_date", "")

            # Build direct URL to the exhibit
            acc_no_dashes = accession.replace("-", "")
            url = f"{ARCHIVES_BASE}/{cik}/{acc_no_dashes}/{filename}" if cik and filename else None

            hits.append({
                "accession": accession,
                "filename": filename,
                "entity": entity,
                "cik": cik,
                "date": date,
                "url": url,
                "file_type": file_type,
            })

            if len(hits) >= max_results:
                break

        offset += page_size
        time.sleep(REQUEST_DELAY)

    logger.info("EDGAR search for %r returned %d EX-10.x hits", query, len(hits))
    return hits


def download_exhibit(url: str) -> Optional[str]:
    """Download an EDGAR exhibit and extract plain text from HTML."""
    if not url:
        return None
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        content = resp.text[:MAX_DOC_SIZE]
    except Exception as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None

    time.sleep(REQUEST_DELAY)

    # Parse HTML and extract text
    soup = BeautifulSoup(content, "lxml")

    # Remove script/style tags
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    if len(text) < 500:
        logger.debug("Exhibit too short after extraction (%d chars), skipping", len(text))
        return None

    return text


def fetch_contracts(
    contract_type: ContractType,
    count: int = 50,
    output_dir: Optional[Path] = None,
) -> list[RawContract]:
    """Fetch EDGAR contracts for a given type and return as RawContract list.

    Optionally saves raw text files to output_dir.
    """
    query = EDGAR_QUERIES.get(contract_type)
    if not query:
        logger.warning("No EDGAR query defined for %s", contract_type)
        return []

    hits = search_edgar(query, max_results=count * 3)  # over-fetch to account for failures

    contracts = []
    for hit in tqdm(hits, desc=f"Downloading {contract_type.value}"):
        if len(contracts) >= count:
            break

        text = download_exhibit(hit["url"])
        if not text:
            continue

        contract = RawContract(
            source="edgar",
            source_id=hit["accession"],
            filename=hit["filename"],
            entity_name=hit["entity"],
            contract_type=contract_type,
            text=text,
            url=hit["url"],
            filing_date=hit["date"],
        )
        contracts.append(contract)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^\w\-.]", "_", hit["filename"] or hit["accession"])
            path = output_dir / f"{contract_type.name}_{safe_name}.txt"
            path.write_text(text, encoding="utf-8")

    logger.info("Fetched %d %s contracts from EDGAR", len(contracts), contract_type.value)
    return contracts
