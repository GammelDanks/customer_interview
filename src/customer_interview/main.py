# src/customer_interview/app.py
import os
import sys
import json
from pathlib import Path
import textwrap
import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Path repair so we can import the package whether we run:
#   streamlit run src/customer_interview/app.py
# or with PYTHONPATH=.../src
# -----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[2] if len(_THIS.parents) >= 2 else _THIS.parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -----------------------------------------------------------------------------
# Load .env and sanitize the OpenAI key
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
os.environ["OPENAI_API_KEY"] = api_key
os.environ.setdefault("MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4o-mini"))

# Encourage richer interview answers (optional dials used by your crew.py)
os.environ.setdefault("ANSWER_MIN_SENTENCES", "3")
os.environ.setdefault("ANSWER_MAX_SENTENCES", "6")
os.environ.setdefault("ENABLE_MICRO_PROBE", "1")

# -----------------------------------------------------------------------------
# Import your crew AFTER env is set
# -----------------------------------------------------------------------------
try:
    from .crew import ValidationCrew  # package-relative
except Exception:
    from customer_interview.crew import ValidationCrew  # absolute fallback

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
    # Expect: [{"segment": "Name", "questions": [q1, q2, ...]}, ...]
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
            if seg.get("notes"):
                st.markdown("**Notes**")
                st.write(seg.get("notes"))
        # Archetypes nested
        arcs = seg.get("archetypes") or []
        if arcs:
            st_container.markdown("**Customer archetypes**")
            for a in arcs:
                lab = a.get("label","")
                with st_container.expander(f"Archetype: {lab}"):
                    st.markdown("**Backstory**")
                    st.write(a.get("backstory") or "—")
                    st.markdown("**Response style**")
                    st.write(a.get("response_style") or "—")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Motivations**")
                        mv = as_list(a.get("motivations"))
                        st.markdown("\n".join(bullet(m) for m in mv) or "—")
                    with c2:
                        st.markdown("**Objections**")
                        ob = as_list(a.get("objections"))
                        st.markdown("\n".join(bullet(o) for o in ob) or "—")
                    with c3:
                        st.markdown("**Dealbreakers**")
                        db = as_list(a.get("dealbreakers"))
                        st.markdown("\n".join(bullet(d) for d in db) or "—")
        st_container.markdown("---")

def render_interviews(st_container, interviews):
    if not interviews:
        st_container.warning("No interviews returned.")
        return
    st_container.write(f"**Total transcripts:** {len(interviews)}")
    for i, itw in enumerate(interviews, start=1):
        seg = itw.get("segment", "N/A")
        lab = itw.get("customer_label", "N/A")
        with st_container.expander(f"Transcript {i}: {seg} — {lab}"):
            transcript = itw.get("transcript") or []
            for j, qa in enumerate(transcript, start=1):
                st.markdown(f"**Q{j}.** {qa.get('question','')}")
                st.write(qa.get("answer", ""))
                st.markdown("---")

def render_summaries(st_container, summaries):
    if not summaries:
        st_container.warning("No segment summaries returned.")
        return
    for i, s in enumerate(summaries, start=1):
        seg = s.get("segment", f"Segment {i}")
        st_container.markdown(f"### {i}. {seg}")
        cols = st_container.columns(2)
        with cols[0]:
            st.markdown("**Top pains**")
            st.markdown("\n".join(f"- {x}" for x in (s.get("pain_points") or [])) or "—")

            st.markdown("**Key needs**")
            st.markdown("\n".join(f"- {x}" for x in (s.get("key_needs") or s.get("needs") or [])) or "—")

            # Unified category (with fallback to legacy keys)
            barriers = (
                s.get("adoption_barriers_and_concerns")
                or s.get("constraints")
                or s.get("risks_unknowns")
                or []
            )
            st.markdown("**Adoption barriers & concerns**")
            st.markdown("\n".join(f"- {x}" for x in barriers) or "—")

        with cols[1]:
            st.markdown("**Buying signals**")
            st.markdown("\n".join(f"- {x}" for x in (s.get("buying_signals") or [])) or "—")

        quotes = s.get("representative_quotes") or s.get("notable_quotes") or []
        if quotes:
            st_container.markdown("**Representative quotes**")
            for q in quotes:
                st_container.markdown(f"> {q}")

        if s.get("narrative"):
            st_container.markdown("**Narrative**")
            st_container.write(s["narrative"])

        st_container.markdown("---")

def render_requirements(st_container, reqs: dict):
    if not reqs:
        st_container.warning("No requirements returned.")
        return

    # Cross-segment first (if present)
    xreqs = reqs.get("cross_segment_requirements") or []
    if xreqs:
        st_container.markdown("### Cross-segment requirements")
        for i, r in enumerate(xreqs, start=1):
            with st_container.expander(f"XS-{i}: {r.get('title','(untitled)')}"):
                st.markdown(f"**ID**: `{r.get('id','')}`")
                st.markdown(f"**Category**: {r.get('category','')}")
                st.markdown(f"**Priority**: {r.get('priority','')}")
                st.markdown(f"**Must-have**: {r.get('must_have', False)}")
                if r.get("rationale"):
                    st.markdown("**Rationale**")
                    st.write(r["rationale"])
                ac = as_list(r.get("acceptance_criteria"))
                if ac:
                    st.markdown("**Acceptance criteria**")
                    st.markdown("\n".join(f"- {a}" for a in ac))
                deps = as_list(r.get("depends_on"))
                if deps:
                    st.markdown("**Depends on**")
                    st.markdown("\n".join(f"- {d}" for d in deps))
                anti = as_list(r.get("anti_requirements"))
                if anti:
                    st.markdown("**DON’Ts / Anti-requirements**")
                    st.markdown("\n".join(f"- {a}" for a in anti))

    # Per-segment requirements
    per_seg = reqs.get("per_segment_requirements") or []
    if per_seg:
        st_container.markdown("### Per-segment requirements")
        for blk in per_seg:
            seg = blk.get("segment", "Segment")
            items = blk.get("requirements") or []
            with st_container.expander(f"🔹 {seg} — {len(items)} items", expanded=False):
                for i, r in enumerate(items, start=1):
                    st.markdown(f"**{i}. {r.get('title','(untitled)')}**")
                    meta = []
                    if r.get("category"):
                        meta.append(r["category"])
                    if r.get("priority"):
                        meta.append(f"prio: {r['priority']}")
                    if r.get("must_have", False):
                        meta.append("must-have")
                    if meta:
                        st.caption(", ".join(meta))
                    if r.get("rationale"):
                        st.markdown("Rationale:")
                        st.write(r["rationale"])
                    ac = as_list(r.get("acceptance_criteria"))
                    if ac:
                        st.markdown("Acceptance criteria")
                        st.markdown("\n".join(f"- {a}" for a in ac))
                    deps = as_list(r.get("depends_on"))
                    if deps:
                        st.markdown("Depends on")
                        st.markdown("\n".join(f"- {d}" for d in deps))
                    anti = as_list(r.get("anti_requirements"))
                    if anti:
                        st.markdown("DON’Ts / Anti-requirements")
                        st.markdown("\n".join(f"- {a}" for a in anti))
                    st.markdown("---")

    if reqs.get("notes"):
        st_container.caption(reqs.get("notes"))

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Interview Crew",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 Customer Interview Crew")
st.caption("Define the idea → segment customers → generate guidelines → simulate interviews → synthesize insights → derive requirements.")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    ui_api = st.text_input("OpenAI API Key", value=api_key, type="password", help="Stored only for this session.")
    if ui_api:
        os.environ["OPENAI_API_KEY"] = ui_api.strip()

    st.markdown("---")
    st.write("**Run options**")
    max_questions = st.slider("Max questions per segment", min_value=5, max_value=20, value=12, step=1)
    max_turns = st.slider("Max turns per interview", min_value=4, max_value=20, value=8, step=1)

    st.markdown("---")
    debug = st.checkbox("Verbose debug (console)", value=False, help="Sets DEBUG_CREW=1 for raw model outputs.")
    os.environ["DEBUG_CREW"] = "1" if debug else "0"

# Inputs
st.header("1) Describe the problem and the solution idea")

col1, col2, col3 = st.columns(3)
with col1:
    problem_summary = st.text_area(
        "Problem summary *(optional)*",
        placeholder="Briefly describe the user/customer problem you want to solve.",
        height=150,
    )
with col2:
    value_prop = st.text_area(
        "Solution & value proposition *(optional)*",
        placeholder="What does your solution do? Why is it effective and interesting? Who benefits and how?",
        height=150,
    )
with col3:
    core_tech = st.text_area(
        "Core technologies *(optional)*",
        placeholder="List key technologies (e.g., mobile app, LLM, wearable, computer vision, etc.).",
        height=150,
    )

# Compose business idea (with English guardrail)
business_idea = "\n".join(
    [
        ensure_english_guardrail(""),
        f"Problem: {problem_summary.strip()}" if problem_summary else "",
        f"Solution & value proposition: {value_prop.strip()}" if value_prop else "",
        f"Core technologies: {core_tech.strip()}" if core_tech else "",
    ]
).strip()

if not problem_summary and not value_prop and not core_tech:
    business_idea = ensure_english_guardrail(
        textwrap.dedent(
            """
            Problem: Busy professionals struggle to maintain healthy eating habits due to lack of time and decision fatigue.
            Solution & value proposition: An AI-powered nutrition coach that plans meals, suggests quick options, and adapts to preferences and constraints while protecting privacy.
            Core technologies: Mobi
