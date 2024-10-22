"""Built-in back-office tools used by suites/back-office/."""

from __future__ import annotations

from eval_harness.stages import tool


@tool("load")
def load(inputs, params):
    return inputs[params["key"]]


@tool("match_invoices")
def match_invoices(inputs, params, invoices, ledger):
    """Pair invoices with ledger entries by reference; return (ref, invoice amount, ledger amount)."""
    book = {row["ref"]: row["amount"] for row in ledger}
    return [[inv["ref"], inv["amount"], book.get(inv["ref"])] for inv in invoices]


@tool("flag_mismatches")
def flag_mismatches(inputs, params, match_invoices):
    tol = params.get("tolerance", 0.01)
    return sorted(ref for ref, inv, led in match_invoices if led is None or abs(inv - led) > tol)


@tool("total")
def total(inputs, params, **upstream):
    (rows,) = upstream.values()
    field = params.get("field", "amount")
    return round(sum(r[field] for r in rows), 2)


@tool("balance")
def balance(inputs, params, **upstream):
    """Difference between two totals; zero when the books balance."""
    a, b = (upstream[k] for k in params["between"])
    return round(a - b, 2)


@tool("draft_email")
def draft_email(inputs, params, flag_mismatches):
    if not flag_mismatches:
        return "All invoices for the period match the ledger. No action needed."
    refs = ", ".join(flag_mismatches)
    return (f"Hello {inputs.get('supplier', 'team')},\n\nThe following invoices do not match our ledger: {refs}. "
            "Please send corrected invoices or the remittance details.\n\nThanks")


@tool("approve_payment")
def approve_payment(inputs, params, flag_mismatches, **totals):
    limit = params.get("limit", 10_000)
    amount = next(iter(totals.values()))
    return "approve" if not flag_mismatches and amount <= limit else "hold"
