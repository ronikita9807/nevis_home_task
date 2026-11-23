"""Load the demo corpus (data/demo_corpus.json) through the public API.

Usage:
    python scripts/seed.py [--url http://localhost:8000] [--api-key KEY]

Standard library only, so it also runs inside the API container:
    docker compose exec api python scripts/seed.py

Re-running is safe: a client that already exists (409) is skipped with its documents.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any

CORPUS = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.json"
CLIENT_FIELDS = ("first_name", "last_name", "email", "description", "social_links")


def post(url: str, body: dict[str, Any], headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(  # noqa: S310 - URL comes from the operator
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


def seed(base_url: str, api_key: str | None) -> None:
    corpus = json.loads(CORPUS.read_text())
    advisors = {advisor["key"]: advisor for advisor in corpus["advisors"]}
    for client in corpus["clients"]:
        advisor = advisors[client["advisor"]]
        headers = {"X-Advisor-Id": advisor["id"]} | ({"X-API-Key": api_key} if api_key else {})
        name = f"{client['first_name']} {client['last_name']} ({advisor['name']})"

        status, created = post(
            f"{base_url}/clients", {k: client[k] for k in CLIENT_FIELDS}, headers
        )
        if status == HTTPStatus.CONFLICT:
            print(f"skip    {name}: already exists")
            continue
        if status != HTTPStatus.CREATED:
            sys.exit(f"failed  {name}: HTTP {status} {created}")

        for document in client["documents"]:
            status, body = post(
                f"{base_url}/clients/{created['id']}/documents",
                {"title": document["title"], "content": document["content"]},
                headers,
            )
            if status != HTTPStatus.CREATED:
                sys.exit(f"failed  {document['title']}: HTTP {status} {body}")
        print(f"created {name}: {len(client['documents'])} documents")

    for advisor in advisors.values():
        print(f"advisor {advisor['name']}: X-Advisor-Id: {advisor['id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("SEED_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY") or None)
    args = parser.parse_args()
    seed(args.url.rstrip("/"), args.api_key)


if __name__ == "__main__":
    main()
