# src/customer_interview/app.py
import os
import sys
import json
from pathlib import Path
import textwrap
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Path repair so we can import whether we run:
#   streamlit run src/customer_interview/app.py
# or with PYTHONPATH=./src

# -----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2] if len(_THIS.parents) >= 2 else _THIS.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --- Preflight: verify package folder exists ---------------------------------
PKG_DIR = SRC / "customer_interview"
if not PKG_DIR.exists():
    st.error(f"Package folder not found: {PKG_DIR}")
    st.stop()

# -----------------------------------------------------------------------------
# Load .env
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# --- Secrets -> Environment (robust: flach ODER verschachtelt) ---------------
def _hydrate_keys_from_secrets():
    def _get_nested(section, key):
        try:
            return str(st.secrets[section][key]).strip()
        except Exception:
            return ""

    def _set_if_missing(env_key, *candidates):
        if os.getenv(env_key, "").strip():
            return
        for cand in candidates:
            val = ""
            if isinstance(cand, tuple):
                val = _get_nested(cand[0], cand[1])
            else:
                try:
                    if cand in st.secrets:
                        val = str(st.secrets[cand]).strip()
                except Exception:
                    pass
            if val:
                os.environ[env_key] = val
                break

    # Primärschlüssel
    _set_if_missing("OPENAI_API_KEY", "OPENAI_API_KEY", ("openai", "openai_api_key"))
    _set_if_missing("YOU_API_KEY", "YOU_API_KEY", ("you", "api_key"))

    # optionale Flags
    for k in ("MODEL_NAME", "YOU_SEARCH_ENABLED", "ANSWER_MIN_SENTENCES",
              "ANSWER_MAX_SENTENCES", "ENABLE_MICRO_PROBE"):
        if not os.getenv(k, "").strip():
            try:
                if k in st.secrets:
                    os.environ[k] = str(st.secrets[k]).strip()
            except Exception:
                pass

# 1. Hydration so früh wie möglich
_hydrate_keys_from_secrets()
# -----------------------------------------------------------------------------

# --- SQLite upgrade shim (force pysqlite3 if stdlib sqlite is too old) -------
import sys as _sys
try:
    import sqlite3 as _sqlite3
    def _ver_tuple(v):
        try:
            return tuple(int(x) for x in str(v).split(".")[:3])
        except Exception:
            return (0, 0, 0)
    if _ver_tuple(_sqlite3.sqlite_version) < (3, 35, 0):
        import pysqlite3 as _pysqlite3
        _sys.modules["sqlite3"] = _pysqlite3
        _sys.modules["sqlite3.dbapi2"] = _pysqlite3.dbapi2
except Exception:
    try:
        import pysqlite3 as _pysqlite3
        _sys.modules["sqlite3"] = _pysqlite3
        _sys.modules["sqlite3.dbapi2"] = _pysqlite3.dbapi2
    except Exception:
        pass

# -----------------------------------------------------------------------------

# Basic env defaults
os.environ.setdefault("MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4o-mini"))
os.environ.setdefault("ANSWER_MIN_SENTENCES", "3")
os.environ.setdefault("ANSWER_MAX_SENTENCES", "6")
os.environ.setdefault("ENABLE_MICRO_PROBE", "1")

# -----------------------------------------------------------------------------
# Crew import (absolute, lowercase)
# -----------------------------------------------------------------------------
try:
    from customer_interview.crew import ValidationCrew  # absolute import, package-relative
except Exception as _e:
    import traceback
    st.error(
        "Import von ValidationCrew fehlgeschlagen.\n"
        "Bitte prüfe:\n"
        "1) Existiert die Datei src/customer_interview/crew.py?\n"
        "2) Heißt der Paketordner exakt 'customer_interview' (lowercase)?\n"
        "3) Wird die App via 'streamlit run src/customer_interview/app.py' gestartet?"
    )
    st.code("".join(traceback.format_exception_only(type(_e), _e)))
    st.stop()



# --- Web search provider (lazy import, case-robust) --------------------------
def _get_search():
    """Return a search provider instance or None, never raising outward."""
    # 1) relative
    try:
        from .integrations.search_factory import get_search_provider as _g
        return _g()
    except Exception:
        pass
    # 2) absolute (capitalized)
    try:
        from CustomerInterview.integrations.search_factory import get_search_provider as _g
        return _g()
    except Exception:
        pass
    # 3) absolute (lowercase)
    try:
        from customer_interview.integrations.search_factory import get_search_provider as _g
        return _g()
    except Exception:
        pass
    # 4) fallback
    return None



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_english_guardrail(text: str) -> str:
    guard = (
        "IMPORTANT: Please produce all outputs in English. "
        "Avoid German language; use natural, concise English.\n\n"
    )
    return guard + (text or "")

def bullet(s: str) -> str:
    return f"- {s}"

def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [str(x)]

def df_guidelines(guidelines):
    rows = []
    for g in (guidelines or []):
        seg = g.get("segment", "")
        for i, q in enumerate(g.get("questions", []), start=1):
            rows.append({"Segment": seg, "#": i, "Question": q})
    if not rows:
        return pd.DataFrame(columns=["Segment", "#", "Question"])
    return pd.DataFrame(rows)

def render_segments_with_archetypes(st_container, merged):
    if not merged:
        st_container.warning("No segments returned.")
        return
    for i, seg in enumerate(merged, start=1):
        st_container.markdown(f"### {i}. *{seg.get('name','(Unnamed segment)')}*  — _{seg.get('type','N/A')}_")
        cols = st_container.columns(2)
        with cols[0]:
            st.markdown("**Needs & concerns**")
            needs = as_list(seg.get("needs_and_concerns"))
            if needs:
                st.markdown("\n".join(bullet(n) for n in needs))
            else:
                st.caption("—")
            st.markdown("**Adoption likelihood**")
            st.write(seg.get("adoption_likelihood") or "—")
        with cols[1]:
            st.markdown("**Willingness to pay**")
            st.write(seg.get("willingness_to_pay") or "—")
            st.markdown("**Notes**")
            st.write(seg.get("notes") or "—")
        st_container.markdown("---")

def render_interviews(st_container, interviews):
    if not interviews:
        st_container.warning("No interviews.")
        return
    for item in interviews:
        seg = item.get("segment") or "(Segment)"
        lab = item.get("customer_label") or "(Label)"
        st_container.markdown(f"**{seg} — {lab}**")
        rows = item.get("transcript") or []
        if rows:
            df = pd.DataFrame(rows)
            st_container.table(df)
        else:
            st_container.caption("—")
        st_container.markdown("---")

def normalize_req_item(r: dict) -> dict:
    return {
        "ID": r.get("id") or r.get("ID") or "",
        "Title": r.get("title") or r.get("name") or "",
        "Category": r.get("category") or "",
        "Priority": r.get("priority") or "",
        "Must-have": r.get("must_have") if isinstance(r.get("must_have"), bool) else (r.get("must_have") or ""),
        "Acceptance criteria": "; ".join(as_list(r.get("acceptance_criteria"))),
        "Rationale": r.get("rationale") or "",
        "Depends on": ", ".join(as_list(r.get("depends_on"))),
        "Anti-reqs": "; ".join(as_list(r.get("anti_requirements") or r.get("anti_req"))),
    }

def render_summaries(st_container, summaries):
    if not summaries:
        st_container.warning("No summaries.")
        return
    for s in summaries:
        seg = s.get("segment") or "(Segment)"
        st_container.markdown(f"**{seg}**")
        cols = st_container.columns(2)
        with cols[0]:
            st.markdown("**Pain points**")
            st.markdown("\n".join(f"- {x}" for x in (s.get("pain_points") or [])) or "—")
            st.markdown("**Key needs**")
            st.markdown("\n".join(f"- {x}" for x in (s.get("key_needs") or [])) or "—")
        with cols[1]:
            st.markdown("**Adoption barriers & concerns**")
            barriers = s.get("adoption_barriers_and_concerns") or s.get("constraints") or []
            st.markdown("\n".join(f"- {x}" for x in barriers) or "—")

        quotes = s.get("representative_quotes") or []
        if quotes:
            st_container.markdown("**Representative quotes**")
            for q in quotes:
                st_container.markdown(f"> {q}")

        if s.get("narrative"):
            st_container.markdown("**Narrative**")
            st_container.write(s["narrative"])

        st_container.markdown("---")

def render_requirements(st_container, requirements_dict: dict):
    if not requirements_dict:
        st_container.warning("No product requirements returned.")
        return

    cross = requirements_dict.get("cross_segment_requirements") or []
    per_seg = requirements_dict.get("per_segment_requirements") or []

    st_container.markdown("### Cross-segment requirements")
    if cross:
        rows = [normalize_req_item(r) for r in cross if isinstance(r, dict)]
        st_container.table(pd.DataFrame(rows))
    else:
        st_container.caption("—")

    st_container.markdown("### Per-segment requirements")
    if per_seg:
        for blk in per_seg:
            seg = blk.get("segment") or "(Unnamed segment)"
            reqs = blk.get("requirements") or []
            with st_container.expander(f"{seg}"):
                rows = [normalize_req_item(r) for r in reqs if isinstance(r, dict)]
                if rows:
                    st.table(pd.DataFrame(rows))
                else:
                    st.caption("—")
    else:
        st_container.caption("—")

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Customer Interview Crew", page_icon="🧩", layout="wide")
st.title("🧩 Customer Interview Crew")
st.caption("Define idea → segment customers → guidelines → simulate interviews → synthesize → requirements.")

with st.sidebar:
    st.subheader("Settings")

    st.markdown("---")
    st.write("**Run options**")
    max_questions = st.slider("Max questions per segment", 5, 20, 12, 1)
    max_turns = st.slider("Max turns per interview", 4, 20, 8, 1)
    max_segments = st.slider("Max segments", 1, 5, 3, 1, help="Hard-cap to avoid overload.")

    st.markdown("---")
    debug = st.checkbox("Verbose debug (console)", value=False, help="Sets DEBUG_CREW=1 for raw model outputs.")
    os.environ["DEBUG_CREW"] = "1" if debug else "0"

    st.markdown("---")
    st.write("**Web search (You.com)**")
    use_search = st.toggle("Use web search (You.com)", value=os.getenv("YOU_SEARCH_ENABLED","false").lower()=="true")
    os.environ["YOU_SEARCH_ENABLED"] = "true" if use_search else "false"
    you_k = st.slider("Max results per query (You.com)", 3, 10, int(os.getenv("YOU_MAX_RESULTS","6")), 1)
    os.environ["YOU_MAX_RESULTS"] = str(you_k)

# Inputs
st.header("1) Describe the problem and the solution idea")

col1, col2, col3 = st.columns(3)
with col1:
    problem_summary = st.text_area("Problem summary *(optional)*", height=150)
with col2:
    value_prop = st.text_area("Solution & value proposition *(optional)*", height=150)
with col3:
    core_tech = st.text_area("Core technologies *(optional)*", height=150)

business_idea = "\n".join(
    [
        ensure_english_guardrail(""),
        f"Problem: {problem_summary.strip()}" if problem_summary else "",
        f"Solution & value proposition: {value_prop.strip()}" if value_prop else "",
        f"Core technologies: {core_tech.strip()}" if core_tech else "",
    ]
).strip()

if not problem_summary and not value_prop and not core_tech:
    business_idea = ensure_english_guardrail(textwrap.dedent("""
        Problem: Busy professionals struggle to maintain healthy eating habits due to lack of time and decision fatigue.
        Solution & value proposition: An AI-powered nutrition coach that plans meals, suggests quick options, and adapts to preferences and constraints while protecting privacy.
        Core technologies: Mobile app, LLM for dialogue & guidance, recommendation engine, calendar integration.
    """).strip())

market_context = "Target geography: EU. Target channels: mobile-first. Keep all outputs in English."
constraints = ["privacy-by-design", "low-friction onboarding", "mobile-first UX"]

run_clicked = st.button("▶️ Run pipeline")
st.markdown("---")

# Output containers
seg_box = st.container()
guide_box = st.container()
int_box = st.container()
sum_box = st.container()
req_box = st.container()

# --- Domain-aware evidence query builder (multi-industry) --------------------
GENERIC_NEGATIVES = [
    "-accounting", "-finance", "-tax", "-crm", "-timesheet", "-payroll", "-bookkeeping",
    "-construction", "-law", "-legal software", "-project management software",
    "-celebrity", "-sports", "-recipe", "-travel blog", "-coupon", "-promo"
]

DEFAULT_GENERIC = {
    "includes": ["consumer", "mobile app", "adoption", "retention", "usability", "privacy", "gdpr"],
    "trusted": ["site:pewresearch.org", "site:ec.europa.eu", "site:nature.com"],
    "neg": GENERIC_NEGATIVES,
}

_DOMAIN_HINTS = {
    "health": {
        "match": ["health", "healthcare", "wellness", "nutrition", "fitness", "medical", "patient", "mhealth"],
        "includes": ["digital health", "mhealth", "wellness", "nutrition", "habit", "behavior change", "GDPR", "HIPAA"],
        "trusted": [
            "site:who.int", "site:nih.gov", "site:cdc.gov", "site:nhs.uk", "site:ema.europa.eu",
            "site:nice.org.uk", "site:ec.europa.eu", "site:bmj.com", "site:jamanetwork.com", "site:nature.com",
            "site:pewresearch.org"
        ],
    },
    "fintech": {
        "match": ["fintech", "payment", "bank", "psd2", "open banking", "insurtech", "crypto", "defi"],
        "includes": ["payments", "KYC", "AML", "fraud detection", "PSD2", "open banking", "instant payments", "SCA", "risk"],
        "trusted": [
            "site:eba.europa.eu", "site:ecb.europa.eu", "site:bis.org", "site:fsb.org",
            "site:bankingsupervision.europa.eu", "site:swift.com", "site:ukfinance.org.uk"
        ],
    },
    "hr": {
        "match": ["hr", "people ops", "workforce", "recruiting", "payroll", "timesheet", "productivity", "collaboration"],
        "includes": ["employee onboarding", "engagement", "burnout", "productivity", "okrs", "time tracking", "privacy", "gdpr"],
        "trusted": [
            "site:gartner.com", "site:forrester.com", "site:pewresearch.org", "site:owasp.org"
        ],
    },
    "ecommerce": {
        "match": ["ecommerce", "e-commerce", "retail", "shop", "marketplace", "conversion"],
        "includes": ["checkout", "cart abandonment", "conversion rate", "return rate", "AOV", "LTV", "privacy", "gdpr"],
        "trusted": [
            "site:baymard.com", "site:pewresearch.org", "site:statista.com", "site:ec.europa.eu"
        ],
    },
    "edtech": {
        "match": ["education", "edtech", "learning", "school", "university", "mooc"],
        "includes": ["learning outcomes", "engagement", "assessment", "privacy", "gdpr", "accessibility"],
        "trusted": [
            "site:oecd.org", "site:unesco.org", "site:pewresearch.org", "site:ec.europa.eu"
        ],
    },
    "climate": {
        "match": ["climate", "energy", "sustainability", "carbon", "renewable", "esg"],
        "includes": ["emissions", "carbon accounting", "scope 1", "scope 2", "scope 3", "reporting", "csrd", "taxonomy"],
        "trusted": [
            "site:ipcc.ch", "site:iea.org", "site:ec.europa.eu", "site:esa.int"
        ],
    },
    "mobility": {
        "match": ["mobility", "transport", "logistics", "fleet", "ev", "charging", "rideshare"],
        "includes": ["route optimization", "telematics", "charging", "range", "total cost of ownership", "safety", "maintenance"],
        "trusted": [
            "site:acea.auto", "site:ec.europa.eu", "site:itu.int"
        ],
    },
    "cybersec": {
        "match": ["security", "cyber", "infosec", "siem", "xdr", "iam", "privacy", "gdpr", "enisa"],
        "includes": ["zero trust", "endpoint security", "identity", "data breach", "encryption", "gdpr", "nist", "iso 27001"],
        "trusted": [
            "site:nist.gov", "site:enisa.europa.eu", "site:owasp.org", "site:iso.org", "site:ec.europa.eu"
        ],
    },
    "devtools": {
        "match": ["developer", "devops", "platform", "cloud", "kubernetes", "observability", "api"],
        "includes": ["ci/cd", "observability", "mttr", "error rate", "latency p95", "slo", "security", "compliance"],
        "trusted": [
            "site:cloud.google.com", "site:aws.amazon.com", "site:learn.microsoft.com",
            "site:cncf.io", "site:linuxfoundation.org"
        ],
    },
    "manufacturing": {
        "match": ["manufacturing", "factory", "iot", "industry 4.0", "oee", "mes", "plc"],
        "includes": ["oee", "downtime", "predictive maintenance", "quality control", "traceability", "safety"],
        "trusted": [
            "site:isa.org", "site:nist.gov", "site:vdi.de", "site:ec.europa.eu"
        ],
    },
    "logistics": {
        "match": ["logistics", "supply chain", "warehouse", "wms", "transport", "freight"],
        "includes": ["on-time delivery", "lead time", "inventory turnover", "forecasting", "visibility", "carbon"],
        "trusted": [
            "site:sciencedirect.com", "site:oecd.org", "site:ec.europa.eu"
        ],
    },
    "legal": {
        "match": ["legal", "law", "compliance", "contract", "privacy", "gdpr"],
        "includes": ["contract review", "privacy", "gdpr", "compliance", "evidence", "retention"],
        "trusted": [
            "site:ec.europa.eu", "site:edps.europa.eu", "site:ico.org.uk"
        ],
    },
    "proptech": {
        "match": ["real estate", "property", "proptech", "rent", "mortgage", "facility"],
        "includes": ["occupancy", "energy", "maintenance", "capex", "opex", "retrofit", "sustainability"],
        "trusted": [
            "site:rics.org", "site:ec.europa.eu"
        ],
    },
    "travel": {
        "match": ["travel", "hospitality", "hotel", "airline", "booking", "tourism"],
        "includes": ["load factor", "revpar", "conversion", "ancillary revenue", "cancellation", "review"],
        "trusted": [
            "site:icao.int", "site:iata.org", "site:oecd.org", "site:ec.europa.eu"
        ],
    },
    "media": {
        "match": ["media", "streaming", "video", "music", "podcast", "creators"],
        "includes": ["arpu", "churn", "engagement", "content discovery", "ad load", "privacy", "gdpr"],
        "trusted": [
            "site:nielsen.com", "site:ofcom.org.uk", "site:ec.europa.eu"
        ],
    },
    "gaming": {
        "match": ["game", "gaming", "esports", "pc game", "mobile game", "console"],
        "includes": ["retention d1", "d7", "arppu", "iap", "session length", "matchmaking", "toxicity"],
        "trusted": [
            "site:newzoo.com", "site:sensor-tower.com", "site:gamedeveloper.com"
        ],
    },
}

def _detect_domain_hints(text: str):
    t = (text or "").lower()
    for key, cfg in _DOMAIN_HINTS.items():
        if any(k in t for k in cfg.get("match", [])):
            return {
                "includes": cfg["includes"],
                "trusted": cfg["trusted"],
                "neg": cfg.get("neg", GENERIC_NEGATIVES),
            }
    return DEFAULT_GENERIC

def _query_variants_for_segment(seg_name: str, seg_type: str, idea_text: str) -> list[str]:
    dom = _detect_domain_hints(idea_text)
    must = " OR ".join(f'"{w}"' for w in dom["includes"])
    neg = " ".join(dom["neg"])
    seg_angle = f'"{seg_name}"' if seg_name else ""

    if (seg_type or "").upper() == "B2B":
        audience = '"enterprise" OR "IT" OR "procurement"'
        jtbd = [
            "adoption barriers OR integration challenges OR data residency",
            "privacy OR compliance OR regulatory obligations",
            "vendor landscape OR build-vs-buy OR switching costs",
        ]
    else:
        audience = '"consumer" OR "B2C" OR "daily routine"'
        jtbd = [
            "adoption barriers OR privacy concerns OR behavior change",
            "user retention OR engagement drop-off OR activation",
            "alternatives comparison OR competitor landscape OR willingness to pay",
        ]

    trusted = dom["trusted"]
    trusted_sets = [
        " OR ".join(trusted[:3]),
        " OR ".join(trusted[3:6]),
        " OR ".join(trusted[6:]),
    ]

    variants = []
    for idx in range(3):
        j = jtbd[idx] if idx < len(jtbd) else jtbd[-1]
        trust = trusted_sets[idx] if idx < len(trusted_sets) else ""
        core = f'({must}) ({j}) {audience} {neg}'
        if seg_angle:
            core = f'{seg_angle} {core}'
        if trust:
            core = f'{core} ({trust})'
        variants.append(core)
    return variants

def _get_search():
    try:
        from .integrations.search_factory import get_search_provider
    except Exception:
        try:
            from customer_interview.integrations.search_factory import get_search_provider
        except Exception:
            def get_search_provider():
                return None
    return get_search_provider()

def _fetch_evidence_for_segments(segments: list[dict], k: int = 6) -> dict[str, list[dict]]:
    provider = _get_search()
    if provider is None or os.getenv("YOU_SEARCH_ENABLED","false").lower() != "true":
        return {}
    out: dict[str, list[dict]] = {}
    idea_text = ""  # keep simple
    for s in segments or []:
        seg_name = s.get("name") or "Segment"
        seg_type = s.get("type") or ""
        items = []
        for q in _query_variants_for_segment(seg_name, seg_type, idea_text)[:3]:
            hits = provider.search(query=q, k=k, freshness="year", news=False) or []
            for h in hits[:4]:
                url = h.get("url")
                title = (h.get("title") or url or "")[:180]
                if url:
                    items.append({"title": title, "url": url})
        seen = set(); dedup = []
        for r in items:
            u = r["url"]
            if u not in seen:
                seen.add(u); dedup.append(r)
        out[seg_name] = dedup[:10]
    return out

def _inject_evidence_into_text(base: str, ev: list[dict], seg_label: str, is_for: str) -> str:
    if not ev:
        return base
    lines = "\n".join(f"- {e['title']} — {e['url']}" for e in ev)
    postfix = (
        f"\n\nEVIDENCE_CONTEXT ({is_for} • {seg_label}):\n"
        "Use these references to keep questions/answers concrete. "
        "Compare claims with vendor capabilities, standards, integration constraints, pricing/licensing, and adoption risks.\n"
        f"{lines}\n"
    )
    return (base or "") + postfix

# Fallback questions if the model omitted some segments in guidelines
B2C_FALLBACK = [
    "Walk me through a typical day and how you currently handle this area.",
    "What’s the most frustrating part — why?",
    "What have you tried so far? What worked, what didn’t?",
    "How often does this happen and how do you respond?",
    "What would your dream product do — and why?",
    "What trade-offs are you making today?",
]
B2B_FALLBACK = [
    "How does the current process work? Who is involved and what tools are used?",
    "What’s the most frustrating step and its impact on time/cost/quality?",
    "What solutions or workarounds exist and what are their limits?",
    "How do you make buying decisions for new tools? Who is involved?",
    "Which compliance/regulatory obligations are relevant here?",
    "If your current solution disappeared tomorrow, what would you do?",
]

def _ensure_guidelines_cover_segments(crew, guidelines: list[dict]) -> list[dict]:
    gmap = {g.get("segment"): g.get("questions", []) for g in (guidelines or []) if isinstance(g, dict)}
    out = list(guidelines or [])
    for s in crew.segments or []:
        name = s.get("name")
        if not name:
            continue
        if name not in gmap or not gmap[name]:
            t = (s.get("type") or "").upper()
            out.append({"segment": name, "questions": (B2B_FALLBACK if t == "B2B" else B2C_FALLBACK)})
    return out

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if run_clicked:
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        st.error("No OpenAI API key provided. Add it in Streamlit secrets as [openai].openai_api_key.")
        st.stop()

    crew = ValidationCrew()

    # 2) Segments + Archetypes
    with st.spinner("Identifying segments and archetypes..."):
        segments = crew.identify_customer_segments(business_idea, market_context, constraints)
        crew.segments = (crew.segments or [])[:max_segments]
        segments = crew.segments

        _ = crew.propose_customer_archetypes()
        merged = crew.segments_with_archetypes()

    with seg_box:
        st.subheader("2) Segments (with customer archetypes)")
        render_segments_with_archetypes(st, merged)

    # Optional: evidence per segment (no-op if YOU_SEARCH_ENABLED=false)
    evidence_by_segment = _fetch_evidence_for_segments(segments, k=int(os.getenv("YOU_MAX_RESULTS","6")))

    # Evidence digest
    crew.evidence_by_segment = evidence_by_segment or {}
    _ = crew.digest_evidence()

    # --- Preview before generating guidelines ---
    with guide_box:
        st.subheader("3) Interview guidelines")
        with st.expander("Preview: sources that will inform the guidelines (per segment)"):
            if not any((evidence_by_segment or {}).values()):
                st.caption("No web sources (web search disabled or no results).")
            else:
                for _seg in (segments or []):
                    _name = _seg.get("name") or "Segment"
                    _items = (evidence_by_segment or {}).get(_name) or []
                    st.markdown(f"**{_name}**")
                    if not _items:
                        st.caption("—")
                    else:
                        for _it in _items:
                            _url = _it.get("url")
                            _title = _it.get("title") or _url
                            if _url:
                                st.markdown(f"- [{_title}]({_url})")

    # 3) Guidelines
    with st.spinner("Generating interview guidelines..."):
        enriched_goal = "Validate problem-solution fit. Keep all questions neutral and non-leading. Respond in English."
        for seg in (segments or []):
            name = seg.get("name") or "Segment"
            ev = evidence_by_segment.get(name) or []
            if ev:
                enriched_goal = _inject_evidence_into_text(enriched_goal, ev, name, is_for="Guidelines")

        with guide_box:
            with st.expander("Preview: enriched goal text (exact content injected into the model)"):
                st.code(enriched_goal, language="markdown")

        guidelines = crew.generate_interview_guidelines(
            business_idea=business_idea,
            business_type_hint=None,
            interview_goal=enriched_goal,
            max_questions=max_questions,
        )

        guidelines = crew.enrich_guidelines_with_evidence(guidelines, max_questions=max_questions)
        guidelines = _ensure_guidelines_cover_segments(crew, guidelines)
        crew.segment_guidelines = guidelines

    with guide_box:
        gdf = df_guidelines(guidelines)
        if gdf.empty:
            st.warning("No guidelines returned.")
        else:
            for seg_name, seg_df in gdf.groupby("Segment"):
                st.markdown(f"**{seg_name}**")
                st.table(seg_df.drop(columns=["Segment"]).reset_index(drop=True))

    # --- Preview before interviews: evidence JSON ---
    with int_box:
        st.subheader("4) Simulated interviews")
        with st.expander("Preview: evidence context that will be provided to interview agents (JSON)"):
            try:
                _ev_json_preview = json.dumps(evidence_by_segment, ensure_ascii=False, indent=2)
                st.code(_ev_json_preview, language="json")
            except Exception:
                st.caption("—")

    # 4) Interviews
    with st.spinner("Running simulated interviews."):
        os.environ["EVIDENCE_CONTEXT_BLOCK"] = json.dumps(evidence_by_segment, ensure_ascii=False)
        interviews = crew.run_interviews_per_segment(max_turns=max_turns)
    with int_box:
        render_interviews(st, interviews)

    # 5) Synthesis
    with st.spinner("Synthesizing segment findings."):
        segment_summaries = crew.synthesize_segment_findings()
    with sum_box:
        st.subheader("5) Segment synthesis")
        render_summaries(st, segment_summaries)

    # 6) Product requirements
    with st.spinner("Deriving product requirements."):
        product_requirements = crew.derive_product_requirements()
    with req_box:
        st.subheader("6) Product requirements")
        render_requirements(st, product_requirements)

    st.success("Done ✅")
    st.toast("Pipeline completed.", icon="✅")

    # Downloads
    st.markdown("---")
    st.subheader("Download artifacts")
    def _dl(payload, label, fname, as_markdown=False):
        if as_markdown:
            raw = str(payload or "").encode("utf-8")
            mime = "text/markdown"
        else:
            raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            mime = "application/json"
        st.download_button(label=label, data=raw, file_name=fname, mime=mime, use_container_width=True)

    _dl(crew.segments, "Download segments.json", "segments.json")
    _dl(crew.archetypes, "Download archetypes.json", "archetypes.json")
    _dl(guidelines, "Download guidelines.json", "guidelines.json")
    _dl(interviews, "Download interviews.json", "interviews.json")
    _dl(segment_summaries, "Download segment_summaries.json", "segment_summaries.json")
    _dl(product_requirements, "Download product_requirements.json", "product_requirements.json")

    # Sources used
    st.markdown("---")
    st.subheader("Sources used")
    links = []
    for seg, items in (evidence_by_segment or {}).items():
        for it in (items or []):
            if it.get("url"):
                links.append((it.get("title") or it["url"], it["url"]))
    seen = set(); uniq = []
    for title, url in links:
        if url not in seen:
            seen.add(url); uniq.append((title, url))
    if not uniq:
        st.caption("No web sources were used (web search disabled or no relevant results).")
    else:
        for title, url in uniq:
            st.markdown(f"- [{title}]({url})")
