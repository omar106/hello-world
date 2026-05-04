#!/usr/bin/env python3
"""
Contract Analysis Multi-Agent Pipeline
=======================================
5 specialized agents working sequentially:

  Agent 1 → Read Documents
  Agent 2 → Catalog & List
  Agent 3 → Gather Requirements (interactive)
  Agent 4 → Deep Contractual Analysis (thinking + streaming)
  Agent 5 → Human Review (interactive Q&A + approval)

Usage:
  python contract_analysis.py contract1.txt contract2.pdf ...
  python contract_analysis.py --demo
"""

import json
import os
import sys
from pathlib import Path

import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-7"
client = anthropic.Anthropic()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _rule(label: str = "") -> None:
    if label:
        print(f"\n{'─' * 20} {label} {'─' * (40 - len(label))}")
    else:
        print("─" * 64)


# ── Agent 1: Document Reader ──────────────────────────────────────────────────

def agent1_read_documents(sources: list[str]) -> dict:
    """
    Reads contract files (or raw text) and extracts structured content.
    Returns raw documents + Claude's structured extraction.
    """
    _banner("AGENT 1 · Document Reader")

    documents: dict[str, dict] = {}
    for src in sources:
        path = Path(src)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            documents[path.name] = {"path": str(path), "content": text}
            print(f"  ✓  {path.name}  ({len(text):,} chars)")
        else:
            key = f"document_{len(documents) + 1}"
            documents[key] = {"path": None, "content": src}
            print(f"  ✓  {key}  ({len(src):,} chars, raw text)")

    if not documents:
        raise ValueError("No documents provided.")

    joined = "\n\n".join(
        f"=== {name} ===\n{d['content']}" for name, d in documents.items()
    )

    system = (
        "You are a specialist in legal document extraction. "
        "For each contract, extract:\n"
        "1. Document type and title\n"
        "2. All parties (names, roles, addresses)\n"
        "3. Effective date, term, and expiry\n"
        "4. Key sections and clause titles\n"
        "5. Obligations, rights, and deliverables\n"
        "6. Monetary amounts, payment schedules, penalties\n"
        "7. Governing law and jurisdiction\n"
        "Be precise and quote exact clause numbers where visible."
    )

    print("\nExtracting content …\n")
    _rule()
    extracted = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=8_000,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Extract all key information from these contracts:\n\n{joined}",
        }],
        thinking={"type": "adaptive"},
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            extracted += chunk

    print("\n")
    return {
        "documents": documents,
        "document_names": list(documents.keys()),
        "extracted_content": extracted,
    }


# ── Agent 2: Document Cataloger ───────────────────────────────────────────────

def agent2_catalog_documents(prev: dict) -> dict:
    """
    Builds a structured JSON catalog from Agent 1's extraction.
    Lists metadata, risks, and cross-document relationships.
    """
    _banner("AGENT 2 · Document Cataloger")
    print("Building catalog …")

    system = (
        "You are a contract cataloging specialist. "
        "Return ONLY valid JSON (no markdown fences) with this exact shape:\n"
        '{"catalog": [{'
        '"document_name": str, "contract_type": str, "parties": [str], '
        '"effective_date": str, "expiry_date": str, "total_value": str, '
        '"key_obligations": [str], "key_rights": [str], '
        '"risks_identified": [str], "status": str, "summary": str'
        "}], "
        '"total_documents": int, '
        '"cross_document_notes": str, '
        '"overall_risk_level": "low"|"medium"|"high"}'
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4_000,
        system=system,
        messages=[{
            "role": "user",
            "content": (
                f"Create a catalog from this extraction:\n\n{prev['extracted_content']}"
            ),
        }],
    )

    raw = resp.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0]

    try:
        catalog = json.loads(raw)
    except json.JSONDecodeError:
        catalog = {"raw": raw}

    # Pretty-print catalog summary
    _rule("Catalog")
    if "catalog" in catalog:
        for entry in catalog["catalog"]:
            print(f"\n  📄  {entry.get('document_name', '—')}")
            print(f"       Type    : {entry.get('contract_type', '—')}")
            print(f"       Parties : {', '.join(entry.get('parties', []))}")
            print(f"       Dates   : {entry.get('effective_date', '—')} → {entry.get('expiry_date', '—')}")
            print(f"       Value   : {entry.get('total_value', '—')}")
            risks = entry.get("risks_identified", [])
            if risks:
                print(f"       Risks   : {'; '.join(risks[:3])}")
            print(f"       Summary : {entry.get('summary', '—')}")

        print(f"\n  Total documents : {catalog.get('total_documents', len(prev['document_names']))}")
        print(f"  Overall risk    : {catalog.get('overall_risk_level', '—').upper()}")
        if catalog.get("cross_document_notes"):
            print(f"  Cross-doc notes : {catalog['cross_document_notes']}")
    else:
        print(raw)

    return {**prev, "catalog": catalog, "catalog_raw": raw}


# ── Agent 3: Requirements Gatherer ────────────────────────────────────────────

def agent3_gather_requirements(prev: dict) -> dict:
    """
    Asks the user targeted questions (generated by Claude) to understand
    what contractual analysis is required, then synthesizes a requirements doc.
    """
    _banner("AGENT 3 · Requirements Gatherer")

    # Generate smart questions from the catalog
    q_resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=(
            "You are a contract analysis consultant. "
            "Based on the provided catalog, write 4 concise, targeted questions "
            "to understand exactly what analysis the client needs. "
            "Number each question. Do not include any preamble."
        ),
        messages=[{
            "role": "user",
            "content": f"Contract catalog:\n\n{prev['catalog_raw']}",
        }],
    )
    questions_text = q_resp.content[0].text.strip()

    print("\nAnswer these questions to guide the deep analysis:\n")
    print(questions_text)
    _rule()

    # Collect answers
    qa_pairs: list[dict] = []
    lines = [
        l.strip() for l in questions_text.splitlines()
        if l.strip() and (l.strip()[0].isdigit() or l.strip().startswith("-"))
    ]

    for line in lines:
        q_text = line.lstrip("0123456789.-) ").strip()
        if not q_text:
            continue
        print(f"\nQ: {q_text}")
        answer = input("A: ").strip()
        if answer.lower() in ("done", "skip", ""):
            continue
        qa_pairs.append({"question": q_text, "answer": answer})

    print("\nAny additional requirements? (press Enter to skip)")
    extra = input("> ").strip()
    if extra:
        qa_pairs.append({"question": "Additional requirements", "answer": extra})

    # Synthesize into a requirements statement
    synth_resp = client.messages.create(
        model=MODEL,
        max_tokens=1_000,
        system=(
            "You are a requirements analyst. "
            "Synthesize the Q&A pairs into a clear, numbered list of analysis requirements. "
            "Be concise and actionable."
        ),
        messages=[{
            "role": "user",
            "content": json.dumps(qa_pairs, indent=2),
        }],
    )
    requirements_summary = synth_resp.content[0].text.strip()

    _rule("Requirements Summary")
    print(requirements_summary)

    return {**prev, "qa_pairs": qa_pairs, "requirements_summary": requirements_summary}


# ── Agent 4: Deep Contractual Analyzer ───────────────────────────────────────

def agent4_deep_analysis(prev: dict) -> dict:
    """
    Performs comprehensive contractual analysis with adaptive thinking.
    Uses prompt caching so the large document content isn't re-tokenised.
    Streams output for responsiveness.
    """
    _banner("AGENT 4 · Deep Contractual Analyzer")
    print("Running deep analysis (streaming with extended thinking) …\n")

    doc_block = "\n\n".join(
        f"=== {name} ===\n{d['content']}"
        for name, d in prev["documents"].items()
    )

    system = """You are a senior contracts attorney with 20+ years of experience.
Perform a comprehensive analysis structured as follows:

## 1. LEGAL RISKS & LIABILITIES
- Unfavourable or one-sided clauses
- Indemnification and liability exposure
- Missing protective provisions

## 2. OBLIGATIONS & COMPLIANCE
- Each party's obligations with due dates
- Regulatory / compliance requirements
- Performance metrics and KPIs

## 3. FINANCIAL TERMS
- Payment schedules and milestones
- Penalty and liquidated damages clauses
- Revenue and cost implications

## 4. TERMINATION & EXIT RIGHTS
- Grounds for termination (for cause / convenience)
- Notice periods and cure periods
- Survival clauses and post-termination obligations

## 5. INTELLECTUAL PROPERTY
- IP ownership and assignment provisions
- License grants, scope, and restrictions
- Confidentiality and non-disclosure obligations

## 6. DISPUTE RESOLUTION
- Governing law and jurisdiction
- Arbitration / mediation clauses
- Limitation of actions

## 7. KEY RECOMMENDATIONS
- Priority issues (ranked by urgency)
- Suggested contract amendments
- Negotiation leverage points

Cite specific clause numbers and quote exact text where material."""

    _rule("Live Output")
    analysis = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=16_000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},   # cache stable system prompt
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"CONTRACT DOCUMENTS:\n\n{doc_block}",
                    "cache_control": {"type": "ephemeral"},   # cache document corpus
                },
                {
                    "type": "text",
                    "text": (
                        f"\nANALYSIS REQUIREMENTS:\n{prev['requirements_summary']}"
                        "\n\nPlease perform the full deep contractual analysis now."
                    ),
                },
            ],
        }],
    ) as stream:
        for chunk in stream.text_stream:
            print(chunk, end="", flush=True)
            analysis += chunk
        final = stream.get_final_message()

    print("\n")
    usage = final.usage
    _rule("Token Usage")
    print(f"  Input tokens  : {usage.input_tokens:,}")
    print(f"  Output tokens : {usage.output_tokens:,}")
    if getattr(usage, "cache_read_input_tokens", 0):
        print(f"  Cache hits    : {usage.cache_read_input_tokens:,} tokens (saved cost)")

    return {
        **prev,
        "analysis": analysis,
        "usage": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0),
        },
    }


# ── Agent 5: Human Review ─────────────────────────────────────────────────────

def agent5_human_review(prev: dict) -> dict:
    """
    Presents the analysis for human review.
    Supports follow-up Q&A, saving the report, and formal approval.
    """
    _banner("AGENT 5 · Human Review")

    # Executive summary
    summ_resp = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=(
            "You are a senior legal advisor. "
            "Write a concise executive summary (6–8 bullet points) of the key findings, "
            "top risks, and critical recommendations from this contract analysis."
        ),
        messages=[{"role": "user", "content": prev["analysis"]}],
    )
    exec_summary = summ_resp.content[0].text.strip()

    _rule("Executive Summary")
    print(exec_summary)
    _rule()

    print("\nOptions:")
    print("  • Ask any question about the analysis")
    print("  • 'save'    → save full report to contract_report.md")
    print("  • 'approve' → formally approve the analysis")
    print("  • 'exit'    → finish without approving")
    _rule()

    # Seed conversation with full analysis context
    history = [
        {
            "role": "user",
            "content": (
                f"Here is the complete contract analysis:\n\n{prev['analysis']}\n\n"
                "I am ready to answer questions about it."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "I have the full analysis available. Ask me anything — "
                "about specific clauses, risks, obligations, or recommendations."
            ),
        },
    ]

    comments: list[str] = []
    approved = False

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "exit":
            break

        if cmd == "approve":
            approved = True
            print("\n  ✅  Analysis APPROVED.")
            break

        if cmd == "save":
            report_path = Path("contract_report.md")
            report_path.write_text(
                "# Contract Analysis Report\n\n"
                "## Executive Summary\n\n"
                + exec_summary
                + "\n\n---\n\n## Full Analysis\n\n"
                + prev["analysis"]
                + (
                    ("\n\n---\n\n## Review Comments\n\n"
                     + "\n".join(f"- {c}" for c in comments))
                    if comments else ""
                ),
                encoding="utf-8",
            )
            print(f"\n  💾  Report saved → {report_path.resolve()}")
            continue

        # Follow-up question
        comments.append(user_input)
        history.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)
        reply = ""
        with client.messages.stream(
            model=MODEL,
            max_tokens=2_000,
            system=(
                "You are a senior contracts attorney answering questions about a "
                "contract analysis. Be concise and cite specific clauses when relevant."
            ),
            messages=history,
        ) as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
                reply += chunk
        print()

        history.append({"role": "assistant", "content": reply})

    status = "APPROVED" if approved else "PENDING"
    _rule("Review Complete")
    print(f"  Status           : {status}")
    print(f"  Documents        : {len(prev['document_names'])}")
    print(f"  Questions asked  : {len(comments)}")

    return {
        **prev,
        "executive_summary": exec_summary,
        "review_approved": approved,
        "review_comments": comments,
        "status": status,
    }


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def run_pipeline(sources: list[str]) -> dict:
    """Run all 5 agents sequentially, passing output forward."""
    print("\n")
    print("██████████████████████████████████████████████████████████████")
    print("█                                                            █")
    print("█      CONTRACT ANALYSIS  ·  MULTI-AGENT PIPELINE           █")
    print("█      5 Agents · Sequential · Powered by Claude            █")
    print("█                                                            █")
    print("██████████████████████████████████████████████████████████████")
    print(f"\nSources ({len(sources)}):")
    for s in sources:
        p = Path(s)
        label = p.name if p.exists() else f"{s[:60]}…"
        print(f"  • {label}")

    out = agent1_read_documents(sources)
    out = agent2_catalog_documents(out)
    out = agent3_gather_requirements(out)
    out = agent4_deep_analysis(out)
    out = agent5_human_review(out)

    _banner("PIPELINE COMPLETE")
    print(f"  Final status : {out['status']}")
    print(f"  Agents run   : 5 / 5")
    print(f"  Documents    : {len(out['document_names'])}")
    print("═" * 64)

    return out


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--demo":
        demo_files = sorted(Path("sample_contracts").glob("*.txt"))
        if not demo_files:
            print("No sample contracts found in sample_contracts/. Add .txt files there.")
            sys.exit(1)
        sources_list = [str(f) for f in demo_files]
    else:
        sources_list = sys.argv[1:]

    run_pipeline(sources_list)
