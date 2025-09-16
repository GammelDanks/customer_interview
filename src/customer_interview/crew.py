# src/customer_interview/crew.py
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import yaml

# --- Load ENV cleanly (no project/org needed) ---------------------------------
load_dotenv(override=True)

_api_key = os.getenv("OPENAI_API_KEY", "")
_api_key = _api_key.strip() if _api_key else ""
os.environ["OPENAI_API_KEY"] = _api_key
print("OPENAI_API_KEY loaded:", bool(_api_key), "| length:", len(_api_key), "| tail:", _api_key[-6:])

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

PKG_DIR = Path(__file__).resolve().parent


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_llm() -> LLM:
    """Create a CrewAI LLM object (OpenAI via LiteLLM)."""
    return LLM(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        temperature=0.2,
    )


def _mk_agent(name: str, spec: Dict[str, Any]) -> Agent:
    return Agent(
        role=spec.get("role", name),
        goal=spec.get("goal", ""),
        backstory=spec.get("backstory", ""),
        allow_delegation=bool(spec.get("allow_delegation", False)),
        verbose=bool(spec.get("verbose", False)),
        llm=_default_llm(),
    )


def _mk_task(description: str, expected_output: str, agent: Agent) -> Task:
    return Task(description=description, expected_output=expected_output, agent=agent)


@dataclass
class SegmentArchetype:
    segment: str
    label: str  # "critical" | "open_reflective"
    role_name: str
    backstory: str
    response_style: str
    motivations: List[str]
    objections: List[str]
    dealbreakers: List[str]


# ===== English guardrails & debug ============================================
JSON_ONLY = (
    "FORMAT:\n"
    "- You MUST return exactly ONE valid JSON object. No markdown, no code fences, no commentary.\n"
    "- Do not include explanations. Only the JSON object.\n"
)
EN_GUARD = (
    "IMPORTANT: Produce the output in ENGLISH only. "
    "Do not use any German. Keep terminology consistent and concise.\n\n"
)

DEBUG_CREW = os.getenv("DEBUG_CREW", "0") == "1"

# Answer “dials” from .env
ANSWER_MIN_SENTENCES = int(os.getenv("ANSWER_MIN_SENTENCES", "3"))
ANSWER_MAX_SENTENCES = int(os.getenv("ANSWER_MAX_SENTENCES", "6"))
ENABLE_MICRO_PROBE = os.getenv("ENABLE_MICRO_PROBE", "1") == "1"
DIVERSITY_REWRITE = os.getenv("DIVERSITY_REWRITE", "1") == "1"  # NEW

# English question banks (fallbacks)
B2C_QUESTION_BANK = [
    "Walk me through a typical day and how you currently handle this area.",
    "What’s the most frustrating part of this situation — why?",
    "What have you tried so far? What worked, what didn’t?",
    "How often does this happen and how do you usually respond?",
    "How do you decide whether to try a new product or service? What builds trust for you?",
    "If you had a magic wand, what’s the first thing you would change — and why?",
    "What trade-offs are you making today and how does that feel?",
]

B2B_QUESTION_BANK = [
    "How does the current process work? Who is involved and what tools do you use?",
    "What is the most frustrating step — with what impact on time/cost/quality?",
    "What solutions or workarounds do you use and what are their limits?",
    "How often does this come up? What happens if it isn’t solved?",
    "How do you make buying decisions for new tools? Who is involved?",
    "Is there a budget? What would it take to get one?",
    "If your current solution disappeared tomorrow, what would you do?",
]


class ValidationCrew:
    """
    Orchestrates the full problem–solution fit flow:
      - Segment identification + archetypes
      - Interview guidelines
      - (optional) Bias review
      - Simulated interviews (interviewer + customer archetypes)
      - Segment synthesis (robust, with representative quotes)
      - Product requirements derivation (cross-segment + per-segment)
    """

    def __init__(
        self,
        tasks_yaml: Optional[str | Path] = None,
        agents_yaml: Optional[str | Path] = None,
    ):
        # Default: YAMLs next to crew.py
        self.tasks_path = Path(tasks_yaml) if tasks_yaml else (PKG_DIR / "tasks.yaml")
        self.agents_path = Path(agents_yaml) if agents_yaml else (PKG_DIR / "agents.yaml")

        self.tasks_spec = _load_yaml(self.tasks_path)
        self.agents_spec = _load_yaml(self.agents_path)

        # Fixed agents
        self.agents: Dict[str, Agent] = {}
        self._init_fixed_agents()

        # Dynamic customer agents (2 per segment)
        self.customer_agents: Dict[Tuple[str, str], Agent] = {}

        # Caches
        self.segments: List[Dict[str, Any]] = []
        self.archetypes: List[Dict[str, Any]] = []
        self.segment_guidelines: List[Dict[str, Any]] = []
        self.interviews: List[Dict[str, Any]] = []
        self.segment_summaries: List[Dict[str, Any]] = []
        self.product_requirements: Dict[str, Any] = {}
        self.comparison: Dict[str, Any] = {}

    # ---------- Init ----------
    def _init_fixed_agents(self):
        for row in self.agents_spec.get("agents", []):
            name = row["name"]
            self.agents[name] = _mk_agent(name, row)
        self.customer_templates = self.agents_spec.get("customer_archetype_templates", [])

    # ---------- Normalizers & Fallbacks ----------
    def _normalize_segments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        segs = data.get("segments") or data.get("customer_segments") or data.get("segment_list") or []
        out: List[Dict[str, Any]] = []
        if isinstance(segs, list):
            for s in segs:
                if isinstance(s, dict):
                    name = s.get("name") or s.get("segment") or s.get("label") or s.get("title")
                    if name:
                        out.append({
                            "name": str(name),
                            "type": s.get("type") or s.get("business_type") or "",
                            "needs_and_concerns": s.get("needs_and_concerns") or s.get("pains") or s.get("needs") or [],
                            "adoption_likelihood": s.get("adoption_likelihood") or s.get("adoption") or "",
                            "willingness_to_pay": s.get("willingness_to_pay") or s.get("wtp") or "",
                            "notes": s.get("notes") or "",
                        })
                elif isinstance(s, str):
                    out.append({"name": s, "type": "", "needs_and_concerns": [], "adoption_likelihood": "", "willingness_to_pay": "", "notes": ""})
        elif isinstance(segs, dict):
            for k, v in segs.items():
                out.append({"name": str(k), "type": "", "needs_and_concerns": v if isinstance(v, list) else [], "adoption_likelihood": "", "willingness_to_pay": "", "notes": ""})
        return out

    def _fallback_segments(self, business_idea: str) -> List[Dict[str, Any]]:
        return [
            {"name": "Early adopters", "type": "B2C",
             "needs_and_concerns": ["Quick results", "Low setup effort"],
             "adoption_likelihood": "high", "willingness_to_pay": "medium", "notes": business_idea},
            {"name": "Price-sensitive users", "type": "B2C",
             "needs_and_concerns": ["Cost control", "Transparency"],
             "adoption_likelihood": "medium", "willingness_to_pay": "low", "notes": business_idea},
            {"name": "Quality-focused users", "type": "B2C",
             "needs_and_concerns": ["Reliability", "Support"],
             "adoption_likelihood": "medium", "willingness_to_pay": "high", "notes": business_idea},
        ]

    def _normalize_archetypes(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        arcs = data.get("archetypes") or []
        norm: List[Dict[str, Any]] = []
        if isinstance(arcs, list):
            for blk in arcs:
                seg = blk.get("segment") if isinstance(blk, dict) else None
                customers = blk.get("customers", []) if isinstance(blk, dict) else []
                cc = []
                for idx, c in enumerate(customers):
                    if not isinstance(c, dict):
                        continue
                    label = c.get("label") or ("critical" if idx == 0 else "open_reflective")
                    cc.append({
                        "label": label,
                        "backstory": c.get("backstory", ""),
                        "response_style": c.get("response_style", ""),
                        "motivations": c.get("motivations", []),
                        "objections": c.get("objections", []),
                        "dealbreakers": c.get("dealbreakers", []),
                    })
                if seg and cc:
                    labels = {c["label"] for c in cc}
                    if "critical" not in labels:
                        cc.insert(0, {
                            "label": "critical",
                            "backstory": "Skeptical, cost-conscious, has tried many tools before.",
                            "response_style": "concise, pragmatic, critical",
                            "motivations": ["Save time", "Avoid errors"],
                            "objections": ["too expensive", "too complex"],
                            "dealbreakers": ["lack of transparency", "long learning curve"],
                        })
                    if "open_reflective" not in labels:
                        cc.append({
                            "label": "open_reflective",
                            "backstory": "Curious, aims for sustainable improvements; reflective thinker.",
                            "response_style": "reflective, reasoned, uses examples",
                            "motivations": ["Quality", "Reliability"],
                            "objections": ["unclear value"],
                            "dealbreakers": ["privacy concerns"],
                        })
                    norm.append({"segment": seg, "customers": cc})
        return norm

    def _fallback_archetypes(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for s in segments[:3]:
            segname = s.get("name", "Segment")
            out.append({
                "segment": segname,
                "customers": [
                    {
                        "label": "critical",
                        "backstory": f"{segname}: skeptical, cost-conscious; tried several solutions before.",
                        "response_style": "concise, pragmatic, critical",
                        "motivations": ["Save time", "Avoid errors"],
                        "objections": ["too expensive", "too complex"],
                        "dealbreakers": ["lack of transparency", "long onboarding"],
                    },
                    {
                        "label": "open_reflective",
                        "backstory": f"{segname}: curious and open to new ideas; wants lasting improvements.",
                        "response_style": "reflective, reasoned, with examples",
                        "motivations": ["Quality", "Reliability"],
                        "objections": ["unclear value"],
                        "dealbreakers": ["privacy concerns"],
                    },
                ]
            })
        return out

    # ---------- Evidence: digest + evidence-anchored guidelines ----------
    def digest_evidence(self) -> dict:
        if not hasattr(self, "evidence_by_segment"):
            self.evidence_by_segment = {}

        links = self.evidence_by_segment or {}
        if not links:
            self.evidence_digest_by_segment = {}
            return self.evidence_digest_by_segment

        agent = self.agents.get("evidence_analyst") or self.agents.get("analyst")
        try:
            tdef = self._task_def("digest_evidence")
            description = (
                EN_GUARD
                + f"{tdef['description']}\n\n"
                + JSON_ONLY
                + "Inputs:\n"
                + f"- segments: {json.dumps(self.segments, ensure_ascii=False, indent=2)}\n"
                + f"- evidence_links_by_segment: {json.dumps(links, ensure_ascii=False, indent=2)}\n"
            )
            task = _mk_task(description, "A single JSON object.", agent)
            out = self._run_single(task)
            data = self._safe_json(out) or {}
            digest = data.get("evidence_digest_by_segment") or {}
        except Exception:
            digest = {}
            for seg, items in (links or {}).items():
                refs = [{"id": i + 1, "title": it.get("title"), "url": it.get("url")} for i, it in enumerate(items or [])]
                facts = [f"Consider source [ref {r['id']}] — {r['title']}" for r in refs[:6]]
                digest[seg] = {"references": refs, "facts": facts, "implications": []}

        self.evidence_digest_by_segment = digest
        return self.evidence_digest_by_segment

    def enrich_guidelines_with_evidence(self, guidelines: list[dict], max_questions: int = 12) -> list[dict]:
        if not hasattr(self, "evidence_digest_by_segment"):
            self.evidence_digest_by_segment = {}

        agent = self.agents.get("interview_designer")
        try:
            tdef = self._task_def("enrich_guidelines_with_evidence")
            description = (
                EN_GUARD
                + f"{tdef['description']}\n\n"
                + JSON_ONLY
                + "Inputs:\n"
                + f"- current_guidelines: {json.dumps(guidelines, ensure_ascii=False, indent=2)}\n"
                + f"- evidence_digest_by_segment: {json.dumps(self.evidence_digest_by_segment, ensure_ascii=False, indent=2)}\n"
                + f"- max_questions: {max_questions}\n"
            )
            task = _mk_task(description, "A single JSON object.", agent)
            out = self._run_single(task)
            data = self._safe_json(out) or {}
            new_gl = data.get("segment_guidelines")
            if new_gl:
                return new_gl
        except Exception:
            pass

        by_seg = {}
        for block in (guidelines or []):
            seg = block.get("segment")
            qs = list(block.get("questions") or [])
            digest = (self.evidence_digest_by_segment or {}).get(seg) or {}
            facts = list(digest.get("facts") or [])
            if not qs or not facts:
                by_seg.setdefault(seg, qs)
                continue
            half = max(1, min(len(qs) // 2, len(facts)))
            new_qs = []
            for i, q in enumerate(qs):
                if i < half and i < len(facts):
                    new_qs.append(f"{facts[i]} → {q}")
                else:
                    new_qs.append(q)
            by_seg.setdefault(seg, new_qs[:max_questions])

        out = [{"segment": s, "questions": q} for s, q in by_seg.items()]
        return out or guidelines

    # ---------- Steps ----------
    def identify_customer_segments(
        self,
        business_idea: str,
        market_context: str = "",
        constraints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        agent = self.agents["segmentation_expert"]
        tdef = self._task_def("identify_customer_segments")

        description = (
            EN_GUARD
            + f"{tdef['description']}\n\n"
            + JSON_ONLY
            + "Output schema hint: {\"segments\": [...], \"assumptions_and_risks\": [...]}.\n\n"
            + "Inputs:\n"
            + f"- business_idea: {business_idea}\n"
            + f"- market_context: {market_context}\n"
            + f"- constraints: {constraints or []}\n"
        )
        task = _mk_task(description, "A single JSON object.", agent)

        out = self._run_single(task)
        data = self._safe_json(out)
        segs = self._normalize_segments(data)

        if not segs:
            retry_description = (
                EN_GUARD
                + "[STRICT RETRY]\n"
                "Return ONLY one JSON object with key 'segments' as array. Each item needs at least {name:string}.\n\n"
                + description
            )
            retry_task = _mk_task(retry_description, "A single JSON object.", agent)
            out2 = self._run_single(retry_task)
            data2 = self._safe_json(out2)
            segs = self._normalize_segments(data2)

        if not segs:
            segs = self._fallback_segments(business_idea)

        self.segments = segs
        return self.segments

    def propose_customer_archetypes(self) -> List[Dict[str, Any]]:
        agent = self.agents["segmentation_expert"]
        tdef = self._task_def("propose_customer_archetypes")

        description = (
            EN_GUARD
            + f"{tdef['description']}\n\n"
            + JSON_ONLY
            + "Output schema hint: {\"archetypes\": [{\"segment\": str, \"customers\": [...]}]}.\n\n"
            + "Segments:\n"
            + json.dumps(self.segments, indent=2, ensure_ascii=False)
            + "\n"
        )
        task = _mk_task(description, "A single JSON object.", agent)

        out = self._run_single(task)
        data = self._safe_json(out)
        arcs = self._normalize_archetypes(data)

        if not arcs:
            retry_desc = (
                EN_GUARD
                + "[STRICT RETRY]\n"
                "Return ONLY one JSON object with key 'archetypes' = array of {segment, customers:[{label, backstory,...}]}.\n\n"
                + description
            )
            retry_task = _mk_task(retry_desc, "A single JSON object.", agent)
            out2 = self._run_single(retry_task)
            data2 = self._safe_json(out2)
            arcs = self._normalize_archetypes(data2)

        if not arcs:
            arcs = self._fallback_archetypes(self.segments)

        self.archetypes = arcs
        self._instantiate_customer_agents(self.archetypes)

        if DEBUG_CREW:
            by_seg: Dict[str, List[str]] = {}
            for (seg, lab), _ag in self.customer_agents.items():
                by_seg.setdefault(seg, []).append(lab)
            print("[DEBUG_CREW] Customer agents per segment:", by_seg)

        return self.archetypes

    def _normalize_guidelines(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        gl = data.get("segment_guidelines") or data.get("guidelines") or data.get("per_segment_guidelines") or []
        norm: List[Dict[str, Any]] = []

        if isinstance(gl, list):
            for item in gl:
                if isinstance(item, dict):
                    seg = item.get("segment") or item.get("name") or item.get("segment_name")
                    qs = item.get("questions") or item.get("qs") or []
                    qq: List[str] = []
                    for q in qs:
                        if isinstance(q, str):
                            qq.append(q)
                        elif isinstance(q, dict) and "q" in q:
                            qq.append(str(q["q"]))
                    if seg and qq:
                        norm.append({"segment": seg, "questions": qq})

        if not norm and isinstance(gl, dict):
            for seg, qs in gl.items():
                if isinstance(qs, list) and qs:
                    qq = [q["q"] if isinstance(q, dict) and "q" in q else str(q) for q in qs]
                    norm.append({"segment": str(seg), "questions": [x for x in qq if x]})

        return norm

    def generate_interview_guidelines(
        self,
        business_idea: str,
        business_type_hint: Optional[str] = None,
        interview_goal: str = "",
        max_questions: int = 15,
    ) -> List[Dict[str, Any]]:
        agent = self.agents["interview_designer"]
        tdef = self._task_def("generate_interview_guidelines")

        description = (
            EN_GUARD
            + f"{tdef['description']}\n\n"
            + JSON_ONLY
            + "Output schema hint: {\"business_type\": \"B2C|B2B\", \"segment_guidelines\": "
              "[{\"segment\": str, \"questions\": [str,...]}], \"notes\": str}.\n\n"
            + "Inputs:\n"
            + f"- business_idea: {business_idea}\n"
            + f"- segments: {json.dumps(self.segments, indent=2, ensure_ascii=False)}\n"
            + f"- business_type_hint: {business_type_hint or 'None'}\n"
            + f"- interview_goal: {interview_goal}\n"
            + f"- max_questions: {max_questions}\n"
        )
        task = _mk_task(description, "A single JSON object.", agent)

        out = self._run_single(task)
        data = self._safe_json(out)
        normalized = self._normalize_guidelines(data)

        if not normalized:
            retry_description = (
                EN_GUARD
                + "[STRICT RETRY]\n"
                "Return ONLY one JSON object. Do NOT wrap in code fences.\n"
                "Required keys: business_type (\"B2C\"|\"B2B\"), segment_guidelines "
                "([{segment, questions[]}]), notes.\n\n"
                + description
            )
            retry_task = _mk_task(retry_description, "A single JSON object.", agent)
            out2 = self._run_single(retry_task)
            data2 = self._safe_json(out2)
            normalized = self._normalize_guidelines(data2)

        if not normalized:
            def _guess_type(segname: str, segtype: str) -> str:
                t = (segtype or "").upper()
                if t in ("B2B", "B2C"):
                    return t
                if any(k in (segname or "").lower() for k in ["company", "team", "enterprise", "business", "b2b"]):
                    return "B2B"
                return "B2C"

            normalized = []
            for seg in self.segments:
                segname = seg.get("name", "Segment")
                kind = _guess_type(segname, seg.get("type", ""))
                bank = B2B_QUESTION_BANK if kind == "B2B" else B2C_QUESTION_BANK
                normalized.append({
                    "segment": segname,
                    "questions": bank[: min(len(bank), max_questions)]
                })

        self.segment_guidelines = normalized
        return self.segment_guidelines

    def review_guidelines_for_bias(self) -> Dict[str, Any]:
        if not self.segment_guidelines:
            return {"issues_found": [], "revised_guidelines": []}
        agent = self.agents["interview_designer"]
        tdef = self._task_def("review_guidelines_for_bias")

        description = (
            EN_GUARD
            + f"{tdef['description']}\n\n"
            + JSON_ONLY
            + "Output schema hint: {\"issues_found\": [str], \"revised_guidelines\": "
              "[{\"segment\": str, \"questions\": [str]}]}.\n\n"
            + "Segment Guidelines:\n"
            + json.dumps(self.segment_guidelines, indent=2, ensure_ascii=False)
            + "\n"
        )
        task = _mk_task(description, "A single JSON object.", agent)

        out = self._run_single(task)
        return self._safe_json(out)

    # ---------- small sanitizer (remove boilerplate & tame % overload) ----------
    def _sanitize_text(self, s: str) -> str:
        if not s:
            return s
        txt = str(s).strip()

        patterns = [
            r"^\s*i\s+now\s+can\s+give\s+(you\s+)?a\s+great\s+answer[:\-–—,.\s]*",
            r"^\s*as\s+(an|a)\s+(ai|language\s+model)[^.!?]*[.!?]\s*",
            r"^\s*(sure|certainly|of\s+course|honestly|to\s+answer\s+your\s+question)[,:\-–—\s]+",
        ]
        for pat in patterns:
            txt = re.sub(pat, "", txt, flags=re.IGNORECASE).strip()

        # Limit % signs: keep first %, convert others to "about N"
        matches = list(re.finditer(r"(\b\d{1,3})\s?%", txt))
        if len(matches) > 1:
            offset = 0
            for m in matches[1:]:
                start, end = m.start(0) + offset, m.end(0) + offset
                num = m.group(1)
                repl = f"about {num}"
                txt = txt[:start] + repl + txt[end:]
                offset += len(repl) - (end - start)

        txt = re.sub(r"\s+", " ", txt).strip()
        return txt

    # ---------- Anti-overlap helpers ----------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        parts = re.split(r'(?<=[.!?])\s+', (text or "").strip())
        return [p.strip() for p in parts if p and not p.isspace()]

    @staticmethod
    def _tok(s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", (s or "").lower())

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _key_terms_from_question(q: str, k: int = 2) -> List[str]:
        stop = {
            "the","a","an","and","or","but","if","when","how","what","why","where","which","who",
            "to","for","with","without","on","in","of","at","by","about","from","is","are","do","does",
            "this","that","these","those","it","its","be","been","being","as","into","over","under","up","down"
        }
        toks = [t for t in ValidationCrew._tok(q) if len(t) >= 4 and t not in stop]
        seen, kept = set(), []
        for t in toks:
            if t not in seen:
                seen.add(t); kept.append(t)
            if len(kept) >= k:
                break
        return kept or (ValidationCrew._tok(q)[:1] or [])

    @staticmethod
    def _extract_topics(answer: str) -> Set[str]:
        """Grab durations, frequencies, and common wellness/usage activities + bigrams."""
        a = (answer or "").lower()
        topics: Set[str] = set()

        # durations & frequencies
        for m in re.findall(r"\b\d{1,3}\s*(minutes?|hours?)\b", a):
            topics.add(m.strip())
        for m in re.findall(r"\b\d{1,3}\s*(x|times)\s*(per\s+)?(day|week|month)\b", a):
            topics.add("".join(m).strip())
        for m in re.findall(r"\b(daily|every\s+morning|each\s+morning|every\s+day)\b", a):
            topics.add(m.strip())

        # common activities (extendable)
        activity_words = [
            "meditation", "meditate", "mindfulness", "yoga", "breathing", "breathwork",
            "running", "jogging", "gym", "workout", "journaling", "journal",
            "meal prep", "tracking", "track my calories", "fasting"
        ]
        for w in activity_words:
            if w in a:
                topics.add(w)

        # bigrams as soft topics
        toks = [t for t in re.findall(r"[a-z0-9]+", a) if len(t) >= 3]
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            topics.add(bg)

        # normalize short composites like "15 minutes meditation"
        for m in re.findall(r"\b\d{1,3}\s*minutes?\s+[a-z]{3,}\b", a):
            topics.add(m.strip())
        return topics

    def _deoverlap_and_align(self, answer: str, history: List[str], question: str) -> str:
        """Drop near-duplicate sentences vs. earlier answers and make sure we reference the question."""
        prev_sents: List[str] = []
        for h in history[-6:]:
            prev_sents.extend(self._split_sentences(h))
        prev_norm = [self._tok(s) for s in prev_sents]

        cand_sents = self._split_sentences(answer)
        unique_sents: List[str] = []
        for s in cand_sents:
            tok_s = self._tok(s)
            too_close_prev = any(self._jaccard(tok_s, ps) >= 0.80 for ps in prev_norm)
            too_close_self = any(self._jaccard(tok_s, self._tok(u)) >= 0.85 for u in unique_sents)
            if not too_close_prev and not too_close_self:
                unique_sents.append(s)

        if len(unique_sents) < 2 and cand_sents:
            unique_sents = cand_sents[:2]

        key = self._key_terms_from_question(question)
        if key:
            key_hit = any(any(k in self._tok(s) for k in key) for s in unique_sents)
            if not key_hit and unique_sents:
                unique_sents[0] = f"Regarding {', '.join(key)}: " + unique_sents[0].lstrip()

        out = " ".join(unique_sents).strip()
        return out or answer

    def _should_rewrite(self, ans: str, banlist: Set[str], history: List[str]) -> bool:
        a = ans.lower()
        if any(b in a for b in banlist):
            return True
        # high similarity vs last two answers
        last = " ".join(history[-2:])
        if last:
            jac = self._jaccard(self._tok(a), self._tok(last))
            if jac >= 0.45:
                return True
        return False

    def _rewrite_for_diversity(self, question: str, draft: str, banlist: Set[str], customer: Agent) -> str:
        if not DIVERSITY_REWRITE:
            return draft
        forbidden = ", ".join(sorted(banlist))[:800] if banlist else ""
        prompt = (
            "You are the same interviewee.\n"
            "Rewrite the following answer so that it directly addresses the current QUESTION, "
            "and avoids ANY mention or close paraphrase of the FORBIDDEN topics.\n"
            "- Keep 3–5 sentences, natural tone.\n"
            "- Use at most one number; prefer ranges or absolute units (€, minutes/week, #tools).\n"
            "- Provide a fresh, distinct example not used earlier.\n"
            "- Start straight with your point; no boilerplate.\n\n"
            f"QUESTION: {question}\n"
            f"FORBIDDEN: {forbidden if forbidden else '(none)'}\n"
            f"CURRENT DRAFT: {draft}\n"
            "Return only the rewritten answer."
        )
        t = _mk_task(prompt, "A 3–5 sentence rewritten answer.", customer)
        out = str(self._run_single(t)).strip()
        out = self._sanitize_text(out)
        return out if out else draft

    # ---------- intern ----------
    def _simulate_dialogue(
        self,
        interviewer: Agent,
        customer: Agent,
        questions: List[str],
        max_turns: int = 20,
        segment_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        transcript: List[Dict[str, str]] = []
        turns = 0
        history: List[str] = []
        banlist: Set[str] = set()  # <- accumulates topics to AVOID repeating

        for q in questions:
            if turns >= max_turns:
                break
            transcript.append({"question": q, "answer": ""})

            # Mini-evidence hint & brand extraction
            facts: List[str] = []
            brands: List[str] = []
            try:
                digest = getattr(self, "evidence_digest_by_segment", {}) or {}
                if segment_name and segment_name in digest:
                    facts = list(digest[segment_name].get("facts") or [])[:2]
                    refs = list(digest[segment_name].get("references") or [])[:6]
                    seen = set()
                    for r in refs:
                        u = (r.get("url") or "").strip()
                        host = urlparse(u).netloc.lower().replace("www.", "")
                        if not host:
                            continue
                        parts = host.split(".")
                        brand = parts[-2] if len(parts) >= 2 else host
                        if brand and brand not in seen and brand.isalpha():
                            seen.add(brand)
                            brands.append(brand)
                        if len(brands) >= 3:
                            break
            except Exception:
                pass

            evidence_hint = ""
            if facts:
                evidence_hint = (
                    "\n\n(Background trends you might have casually heard about; do NOT sound like an expert — "
                    "reflect them only if it feels natural: " + " | ".join(facts) + ")"
                )
            brand_hint = ""
            if brands:
                brand_hint = (
                    "\n\n(If relevant, you may mention tools/services you've seen/used, e.g.: "
                    + ", ".join(brands) + "; only if it genuinely fits your experience.)"
                )

            history_snippet = ""
            if history:
                last = " ".join(history[-2:])
                history_snippet = f"\n\nYour previous points (for consistency): {last}"

            # Explicit do-not-mention clause
            forbidden_list = ", ".join(sorted(list(banlist)))[:900] if banlist else ""
            forbidden_clause = (
                f"\n\nDO NOT mention or allude to ANY of these earlier topics: {forbidden_list}."
                "\nIf the question is about one of them, switch to a clearly different angle (new time window, different metric, different activity)."
                if forbidden_list else ""
            )

            cust_desc = (
                "You are the interviewee from the specified customer segment.\n"
                "ANSWER FORMAT:\n"
                f"- {ANSWER_MIN_SENTENCES} to {ANSWER_MAX_SENTENCES} full sentences in natural English.\n"
                "- Make the answer SPECIFIC to the current question; mirror 1–2 key terms from the question.\n"
                "- Include at least one brief, concrete anecdote, but DO NOT reuse earlier anecdotes/activities.\n"
                "- Use numbers sparingly. At most ONE percentage per answer; prefer ranges or absolute units (€, minutes/week, #tools). If unsure, say 'roughly'.\n"
                "- Anchor statements in the last 3–6 months when relevant (recency).\n"
                "- Start directly; no boilerplate like 'I now can give a great answer'.\n"
                "- No bullet points.\n\n"
                f"Role/segment: {customer.role}\n"
                f"Backstory: {getattr(customer, 'backstory', '')}\n"
                f"Question: {q}\n"
                "Answer realistically and consistently with backstory/response style, motivations, objections, and dealbreakers."
                + history_snippet
                + evidence_hint
                + brand_hint
                + forbidden_clause
            )

            cust_task = _mk_task(cust_desc, "A short, credible answer (3–6 sentences).", customer)
            ans = str(self._run_single(cust_task)).strip()
            ans = self._sanitize_text(ans)
            ans = self._deoverlap_and_align(ans, history, q)

            # Enforce diversity with a rewrite if needed
            if self._should_rewrite(ans, banlist, history):
                ans = self._rewrite_for_diversity(q, ans, banlist, customer)
                ans = self._sanitize_text(ans)
                ans = self._deoverlap_and_align(ans, history, q)

            # Optional micro-probe with repetition guard
            if ENABLE_MICRO_PROBE and (turns + 1 < max_turns):
                probe = (
                    "Can you ground that in one fresh, concrete situation with a rough number (€, minutes/week, or times/week)? "
                    "Avoid repeating earlier examples or activities."
                )
                probe_desc = (
                    "You are still the interviewee. "
                    "Answer this follow-up in 2–3 sentences with one brief, concrete example. "
                    "Use numbers sparingly; avoid more percentages and do not reuse prior anecdotes.\n"
                    f"Follow-up: {probe}"
                )
                probe_task = _mk_task(probe_desc, "A brief follow-up answer (2–3 sentences).", customer)
                probe_ans = str(self._run_single(probe_task)).strip()
                probe_ans = self._sanitize_text(probe_ans)
                probe_ans = self._deoverlap_and_align(probe_ans, history + [ans], q)

                # If the probe reintroduced banned content, rewrite it too
                if self._should_rewrite(probe_ans, banlist, history + [ans]):
                    probe_ans = self._rewrite_for_diversity(q, probe_ans, banlist, customer)
                    probe_ans = self._sanitize_text(probe_ans)
                    probe_ans = self._deoverlap_and_align(probe_ans, history + [ans], q)

                ans = (ans + " " + probe_ans).strip()

            # Tiny resonance check sentence (if we had facts), also diversity-guarded
            if facts:
                chk = (
                    "If you think about it: does any of this resonate with things you've seen/heard recently? "
                    "Feel free to say 'not sure' if it doesn't."
                )
                chk_task = _mk_task(
                    "You are still the interviewee. In ONE short sentence, react casually to this prompt: " + chk,
                    "One casual sentence.",
                    customer,
                )
                chk_ans = str(self._run_single(chk_task)).strip()
                chk_ans = self._sanitize_text(chk_ans)
                chk_ans = self._deoverlap_and_align(chk_ans, history + [ans], q)
                if self._should_rewrite(chk_ans, banlist, history + [ans]):
                    chk_ans = self._rewrite_for_diversity(q, chk_ans, banlist, customer)
                    chk_ans = self._sanitize_text(chk_ans)
                    chk_ans = self._deoverlap_and_align(chk_ans, history + [ans], q)
                ans = (ans + " " + chk_ans).strip()

            # Final store
            transcript[-1]["answer"] = ans
            history.append(ans)

            # Update banlist with new topics found in this answer to avoid reuse
            banlist |= self._extract_topics(ans)

            # Strongly blacklist the prototypical repetition: "15 minutes meditation" / "30 minutes yoga" etc.
            hard_repeats = re.findall(r"\b\d{1,2}\s*minutes?\s*(of\s+)?(meditation|mindfulness|yoga|breathing|breathwork)\b", ans.lower())
            for _ in hard_repeats:
                banlist.add("minutes meditation")
                banlist.add("minutes mindfulness")
                banlist.add("minutes yoga")
                banlist.add("minutes breathing")
                banlist.add("breathwork")
                # also generic activity words to steer away
                banlist.add("meditation")
                banlist.add("mindfulness")
                banlist.add("yoga")

            turns += 1

        return transcript

    def _instantiate_customer_agents(self, archetypes: List[Dict[str, Any]]):
        t_crit = next(
            (t for t in self.customer_templates if t.get("template_name") == "customer_critical_template"),
            None,
        )
        t_open = next(
            (t for t in self.customer_templates if t.get("template_name") == "customer_open_reflective_template"),
            None,
        )

        def build_from_tpl(tpl: Dict[str, Any], seg_name: str, cust: Dict[str, Any]) -> Agent:
            seg_slug = seg_name.lower().replace(" ", "_")
            role = tpl.get("role", "").replace("{{segment_name}}", seg_name)
            goal = tpl.get("goal", "").replace("{{segment_name}}", seg_name)
            _ = tpl.get("name", "customer").replace("{{segment_slug}}", seg_slug)
            backstory = cust.get("backstory", "")
            response_style = cust.get("response_style", "")
            motivations = ", ".join(cust.get("motivations", []))
            objections = ", ".join(cust.get("objections", []))
            dealbreakers = ", ".join(cust.get("dealbreakers", []))
            return Agent(
                role=role,
                goal=goal,
                backstory=(
                    f"{backstory}\n\n"
                    f"Response style: {response_style}\n"
                    f"Motivations: {motivations}\n"
                    f"Objections: {objections}\n"
                    f"Dealbreakers: {dealbreakers}\n"
                ),
                allow_delegation=False,
                verbose=False,
                llm=LLM(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0.5, max_tokens=300),
            )

        for block in archetypes:
            seg = block.get("segment")
            customers = block.get("customers", [])
            for idx, cust in enumerate(customers):
                label = cust.get("label") or ("critical" if idx == 0 else "open_reflective")
                if label == "critical" and t_crit:
                    self.customer_agents[(seg, "critical")] = build_from_tpl(t_crit, seg, cust)
                elif label == "open_reflective" and t_open:
                    self.customer_agents[(seg, "open_reflective")] = build_from_tpl(t_open, seg, cust)

    def _task_def(self, name: str) -> Dict[str, Any]:
        for t in self.tasks_spec.get("tasks", []):
            if t.get("name") == name:
                return t
        raise KeyError(f"Task '{name}' not found in {self.tasks_path}.")

    def _run_single(self, task: Task) -> Any:
        crew = Crew(agents=[task.agent], tasks=[task], process="sequential")
        out = crew.kickoff()
        if DEBUG_CREW:
            print("\n" + "=" * 60)
            print(f"[DEBUG_CREW] RAW OUTPUT for task ({task.agent.role}):")
            try:
                s = str(out)
                print(s[:2000] + ("..." if len(s) > 2000 else ""))
            except Exception:
                print(repr(out))
            print("=" * 60 + "\n")
        return out

    def _safe_json(self, text_out: Any) -> Dict[str, Any]:
        s = str(text_out).strip()

        try:
            return json.loads(s)
        except Exception:
            pass

        if "```" in s:
            parts = s.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{") and p.endswith("}"):
                    try:
                        return json.loads(p)
                    except Exception:
                        pass

        first = s.find("{")
        last = s.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = s[first:last + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return {}

    # ---------- Public API (remaining steps) ----------
    def run_interviews_per_segment(self, max_turns: int = 20) -> List[Dict[str, Any]]:
        interviewer = self.agents.get("interviewer")
        if interviewer is None:
            raise RuntimeError("Interviewer agent not found. Check agents.yaml (name: interviewer).")

        results: List[Dict[str, Any]] = []

        # Normalize guidelines to [{"segment": str, "questions": [..]}]
        if self.segment_guidelines and isinstance(self.segment_guidelines, list) and \
           isinstance(self.segment_guidelines[0], dict) and "questions" in self.segment_guidelines[0]:
            normalized_guidelines = self.segment_guidelines
        else:
            normalized_guidelines = self._normalize_guidelines({"segment_guidelines": self.segment_guidelines})

        # map: segment -> questions
        q_map: Dict[str, List[str]] = {
            g["segment"]: g.get("questions", [])
            for g in normalized_guidelines
            if isinstance(g, dict) and g.get("segment") and g.get("questions")
        }

        if not q_map:
            if DEBUG_CREW:
                print("[DEBUG_CREW] run_interviews_per_segment: No questions found in guidelines.")
            self.interviews = []
            return self.interviews

        # Ensure customer agents exist for each segment and run
        for segment, questions in q_map.items():
            if not self.customer_agents.get((segment, "critical")) and not self.customer_agents.get((segment, "open_reflective")):
                block = next((a for a in self.archetypes if a.get("segment") == segment), None)
                if block:
                    self._instantiate_customer_agents([block])

            for label in ("critical", "open_reflective"):
                customer = self.customer_agents.get((segment, label))
                if not customer:
                    if DEBUG_CREW:
                        print(f"[DEBUG_CREW] Missing customer agent for segment='{segment}' label='{label}' — skipping.")
                    continue

                transcript = self._simulate_dialogue(
                    interviewer=interviewer,
                    customer=customer,
                    questions=questions,
                    max_turns=max_turns,
                    segment_name=segment,
                )

                results.append(
                    {
                        "segment": segment,
                        "customer_label": label,
                        "transcript": transcript,
                        "metadata": {},
                    }
                )

        self.interviews = results
        return self.interviews

    # ---------- Synthesis ----------
    def _validate_and_fill_summaries(self, summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        required_arrays = ["pain_points", "key_needs", "constraints", "buying_signals", "risks_unknowns"]
        out = []
        known_segments = {s.get("name") for s in self.segments}
        for s in summaries or []:
            seg = s.get("segment") or next(iter(known_segments), "Segment")
            block = {
                "segment": seg,
                "pain_points": s.get("pain_points") or [],
                "key_needs": s.get("key_needs") or [],
                "constraints": s.get("constraints") or [],
                "buying_signals": s.get("buying_signals") or [],
                "risks_unknowns": s.get("risks_unknowns") or [],
                "narrative": s.get("narrative") or "",
                "representative_quotes": s.get("representative_quotes") or [],
            }
            for k in required_arrays:
                if not block[k]:
                    block[k] = ["(no strong signals extracted)"]
            if not block["representative_quotes"]:
                block["representative_quotes"] = []
            out.append(block)
        return out

    def _fallback_summaries_from_interviews(self) -> List[Dict[str, Any]]:
        by_seg: Dict[str, List[str]] = {}
        for rec in self.interviews or []:
            seg = rec.get("segment", "Segment")
            for qa in rec.get("transcript") or []:
                ans = (qa.get("answer") or "").strip()
                if ans:
                    by_seg.setdefault(seg, []).append(ans)

        out = []
        for s in self.segments:
            name = s.get("name", "Segment")
            quotes = (by_seg.get(name) or [])[:3]
            out.append({
                "segment": name,
                "pain_points": ["(no strong signals extracted)"],
                "key_needs": ["(no strong signals extracted)"],
                "constraints": ["(no strong signals extracted)"],
                "buying_signals": ["(no strong signals extracted)"],
                "risks_unknowns": ["(no strong signals extracted)"],
                "narrative": "",
                "representative_quotes": quotes,
            })
        return out

    def synthesize_segment_findings(self) -> List[Dict[str, Any]]:
        analyst = self.agents["analyst"]
        tdef = self._task_def("synthesize_segment_findings")

        description = (
            "IMPORTANT: Produce the output in ENGLISH only. Do NOT use any German.\n\n"
            f"{tdef['description']}\n\n"
            "FORMAT:\n- Return exactly ONE JSON object.\n"
            "Schema:\n"
            "{\n"
            '  "segment_summaries": [\n'
            '    {"segment":"<name>","pain_points":[str,...],"key_needs":[str,...],"constraints":[str,...],'
            '"buying_signals":[str,...],"risks_unknowns":[str,...],"representative_quotes":[str,...],"narrative":"short paragraph"}\n'
            "  ]\n"
            "}\n"
            "- Do NOT leave arrays empty. If you are uncertain, add a cautious, concise item.\n\n"
            "Inputs:\n"
            f"- segments:\n{json.dumps(self.segments, ensure_ascii=False, indent=2)}\n"
            f"- interviews:\n{json.dumps(self.interviews, ensure_ascii=False, indent=2)}\n"
        )

        task = _mk_task(description, "A single JSON object.", analyst)
        out = self._run_single(task)
        data = self._safe_json(out)
        summaries = data.get("segment_summaries") or []

        if not summaries:
            retry_desc = (
                "IMPORTANT: ENGLISH ONLY. Do NOT leave arrays empty. "
                "Return exactly ONE JSON object with key 'segment_summaries' as specified.\n\n"
                + description
            )
            retry_task = _mk_task(retry_desc, "A single JSON object.", analyst)
            out2 = self._run_single(retry_task)
            data2 = self._safe_json(out2)
            summaries = data2.get("segment_summaries") or []

        if not summaries:
            summaries = self._fallback_summaries_from_interviews()

        summaries = self._validate_and_fill_summaries(summaries)
        self.segment_summaries = summaries
        return self.segment_summaries

    # ---------- Requirements ----------
    def _req_item_defaults(self, r: Dict[str, Any], idx: int) -> Dict[str, Any]:
        def _listify(x):
            if x is None:
                return []
            if isinstance(x, list):
                return [str(i) for i in x if i is not None]
            return [str(x)]
        return {
            "id": r.get("id") or f"REQ-{idx:03d}",
            "title": r.get("title") or r.get("name") or "Unspecified requirement",
            "category": r.get("category") or "unspecified",
            "priority": r.get("priority") or "should",
            "must_have": bool(r.get("must_have")) if isinstance(r.get("must_have"), bool) else str(r.get("must_have") or "").lower() in ("true", "yes", "1", "must", "must-have"),
            "acceptance_criteria": _listify(r.get("acceptance_criteria")),
            "rationale": r.get("rationale") or "",
            "depends_on": _listify(r.get("depends_on")),
            "anti_requirements": _listify(r.get("anti_requirements") or r.get("anti_req")),
        }

    def _normalize_requirements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cross = data.get("cross_segment_requirements") or data.get("cross_requirements") or []
        per_seg = data.get("per_segment_requirements") or data.get("requirements_by_segment") or []

        norm_cross: List[Dict[str, Any]] = []
        for i, r in enumerate(cross, start=1):
            if isinstance(r, dict):
                norm_cross.append(self._req_item_defaults(r, i))

        norm_per: List[Dict[str, Any]] = []
        for blk in per_seg:
            if not isinstance(blk, dict):
                continue
            seg = blk.get("segment") or blk.get("name") or ""
            reqs = blk.get("requirements") or blk.get("items") or []
            norm_items: List[Dict[str, Any]] = []
            for j, r in enumerate(reqs, start=1):
                if isinstance(r, dict):
                    norm_items.append(self._req_item_defaults(r, j))
            if seg:
                norm_per.append({"segment": seg, "requirements": norm_items})

        return {
            "cross_segment_requirements": norm_cross,
            "per_segment_requirements": norm_per,
        }

    def derive_product_requirements(self) -> Dict[str, Any]]:
        product_owner = (
            self.agents.get("product_owner")
            or self.agents.get("strategist")
            or self.agents.get("analyst")
        )
        if product_owner is None:
            raise RuntimeError("No suitable agent found (product_owner/strategist/analyst). Check agents.yaml.")

        tdef = self._task_def("derive_product_requirements")

        description = (
            EN_GUARD
            + f"{tdef['description']}\n\n"
            + JSON_ONLY
            + "Output schema:\n"
            + "{\n"
            + '  "cross_segment_requirements": [\n'
            + '    {"id": "REQ-001", "title": "...", "category": "security|data|ux|performance|integration|compliance|operations|...","priority":"must|should|could","must_have": true,\n'
            + '     "acceptance_criteria": ["Given... When... Then..."], "rationale":"...", "depends_on": ["REQ-000"], "anti_requirements": ["avoid ..."]}\n'
            + "  ],\n"
            + '  "per_segment_requirements": [\n'
            + '    {"segment": "<name>", "requirements": [ { ... requirement item ... }, ... ]}\n'
            + "  ]\n"
            + "}\n\n"
            + "Inputs:\n"
            + f"- segments: {json.dumps(self.segments, ensure_ascii=False)}\n"
            + f"- segment_summaries: {json.dumps(self.segment_summaries, ensure_ascii=False)}\n"
            + "Guidance:\n"
            + "- Translate pains/needs/barriers into clear REQs, each with acceptance criteria.\n"
            + "- Include non-functionals (security, privacy, performance, availability, data residency, energy efficiency, accessibility).\n"
            + "- Add anti-requirements (what to explicitly avoid: dark patterns, over-collection, brittle UX, etc.).\n"
        )

        task = _mk_task(description, "A single JSON object.", product_owner)
        out = self._run_single(task)
        data = self._safe_json(out)
        norm = self._normalize_requirements(data)

        if not norm["cross_segment_requirements"] and not norm["per_segment_requirements"]:
            skeleton_cross: List[Dict[str, Any]] = []
            idx = 1
            for s in self.segment_summaries or []:
                for need in (s.get("key_needs") or []):
                    skeleton_cross.append(self._req_item_defaults({"title": f"Satisfy need: {need}", "category": "functional", "priority": "should"}, idx))
                    idx += 1
            norm = {"cross_segment_requirements": skeleton_cross[:5], "per_segment_requirements": []}

        self.product_requirements = norm
        return self.product_requirements

    # ---------- Deprecated ----------
    def cross_segment_comparison(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        self.comparison = {}
        return self.comparison

    # ---------- UI helper (missing earlier) ----------
    def segments_with_archetypes(self) -> List[Dict[str, Any]]:
        """
        Utility to merge archetypes into segments for nicer UI rendering.
        Does not change pipeline behavior.
        """
        by_seg = {a["segment"]: a.get("customers", []) for a in self.archetypes}
        merged: List[Dict[str, Any]] = []
        for s in self.segments:
            segname = s.get("name")
            block = dict(s)
            block["archetypes"] = by_seg.get(segname, [])
            merged.append(block)
        return merged
