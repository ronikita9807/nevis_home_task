"""Domain thesaurus: names of the same kind of wealth-management / KYC document.

Why it exists: embedding models (even large ones) do not know domain facts such as
"a utility bill is the standard UK proof of address" — for "address proof" a utility bill
ranked 8th-17th of 17 documents (docs/embedding-experiment.md). Listing such equivalences
is deterministic, explainable to compliance and instant.

Each group is a document category. If a query contains any phrase of a group, the other
phrases are added to the keyword (full-text) channel. Single ambiguous words ("will",
"statement") are avoided on purpose. In production this list would be owned by
compliance/operations, or generated offline with an LLM and reviewed.
"""

import re

THESAURUS: tuple[frozenset[str], ...] = (
    frozenset(
        {  # proof of address
            "address proof",
            "proof of address",
            "proof of residence",
            "utility bill",
            "council tax bill",
            "electricity bill",
            "gas bill",
            "water bill",
            "bank statement",
            "tenancy agreement",
        }
    ),
    frozenset(
        {  # proof of identity
            "proof of identity",
            "identity document",
            "photo id",
            "passport",
            "driving licence",
            "driver's license",
            "national identity card",
        }
    ),
    frozenset(
        {  # source of wealth / funds
            "source of wealth",
            "source of funds",
            "inheritance",
            "probate",
            "sale of business",
            "completion statement",
        }
    ),
    frozenset(
        {  # proof of income
            "proof of income",
            "payslip",
            "p60",
            "sa302",
            "tax return",
            "self assessment",
        }
    ),
    frozenset(
        {  # pensions
            "pension",
            "sipp",
            "self-invested personal pension",
            "workplace pension",
            "annuity",
        }
    ),
    frozenset(
        {  # estate planning
            "estate planning",
            "last will",
            "will and testament",
            "trust deed",
            "lasting power of attorney",
            "inheritance tax",
        }
    ),
    frozenset(
        {  # risk profile
            "risk profile",
            "attitude to risk",
            "risk questionnaire",
            "capacity for loss",
        }
    ),
    frozenset(
        {  # tax-wrapped savings
            "isa",
            "stocks and shares isa",
            "junior isa",
            "individual savings account",
        }
    ),
    frozenset(
        {  # protection
            "life insurance",
            "life cover",
            "life policy",
            "whole of life policy",
            "protection policy",
        }
    ),
)


def expand_query(query: str) -> list[str]:
    """Phrases equivalent to the ones found in a normalised query, sorted, query excluded.

    `"address proof"` -> `["bank statement", "council tax bill", "utility bill", ...]`.
    Phrases match on word boundaries: "passport" does not trigger on "passports".
    """
    expansions: set[str] = set()
    for group in THESAURUS:
        if any(_contains_phrase(query, phrase) for phrase in group):
            expansions |= group
    return sorted(p for p in expansions if not _contains_phrase(query, p))


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None
