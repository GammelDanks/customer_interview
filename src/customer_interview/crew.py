# src/customer_interview/crew.py
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# litellm wird als Abhängigkeit von crewai installiert
try:
    import litellm
except Exception:  # Fallback, falls lib nicht geladen werden kann
    litellm = None


# ---------------------------------------------------------------------------
# Kleine Utilities
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Versucht, JSON aus einem LLM-Output zu extrahieren:
    1) direkter json.loads()
    2) größter {...}-Block
    3) größtes [...]-Array
    """
    if not text:
        return None
    # Direkt versuchen
    try:
        obj = json.loads(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:
        pass

    # Größter {}-Block
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
    except Exception:
        pass

    # Größtes []-Array
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return {"_array": obj}
    except Exception:
        pass

    return None


def _llm_json(
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """
    Ruft das LLM über litellm auf und gibt robust JSON zurück.
    Bei Fehlern: liefert leeres Dict.
    """
    model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if not litellm or not api_key:
        # Kein LLM verfügbar -> leeres Ergebnis
        return {}

    try:
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # litellm standardisiert die Antwort
        text = resp.choices[0].message.get("content", "") if hasattr(resp, "choices") else ""
        data = _extract_json(text) or {}
        if isinstance(data, dict):
            return data
        # Falls nur Array extrahiert wurde
        if isinstance(data, list):
            return {"_array": data}
        return {}
    except Exception:
        return {}


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    try:
        val = d.get(key, default)
        return default if val is None else val
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Datenhaltung
# ---------------------------------------------------------------------------
@dataclass
class ValidationCrew:
    # vom LLM befüllt
    segments: List[Dict[str, Any]] = field(default_factory=list)
    archetypes: List[Dict[str, Any]] = field(default_factory=list)
    segment_guidelines: List[Dict[str, Any]] = field(default_factory=list)
    evidence_by_segment: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    evidence_digest_by_segment: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # 1) Segmente erkennen
    # -----------------------------------------------------------------------
    def identify_customer_segments(
        self,
        business_idea: str,
        market_context: Optional[str] = None,
        constraints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        system = (
            "You are a senior market researcher. "
            "Return concise JSON. Do not include commentary, only JSON."
        )
        user = f"""
Return JSON with:
{{
  "segments": [
    {{
      "name": "...",
      "type": "B2C" | "B2B",
      "characteristics": ["..."],
      "needs_and_concerns": ["..."],
      "pain_points": ["..."],
      "attitude": "...",
      "likelihood_to_adopt": "...",
      "buying_behavior": ["..."],
      "willingness_to_pay": "...",
      "differentiators": ["..."]
    }}
  ],
  "assumptions_and_risks": ["..."]
}}

Context:
- Business idea:\n{business_idea.strip()}
- Market context:\n{(market_context or '').strip()}
- Constraints:\n{json.dumps(constraints or [])}
Keep outputs short, practical, in English.
        """.strip()

        data = _llm_json(system, user)
        segs = _safe_get(data, "segments", [])
        if not isinstance(segs, list):  # Fallback mini-Segmente
            segs = [
                {
                    "name": "Time-pressed consumers",
                    "type": "B2C",
                    "characteristics": ["mobile-first", "privacy-aware"],
                    "needs_and_concerns": ["low friction", "clear value"],
                    "pain_points": ["decision fatigue"],
                    "attitude": "curious but skeptical",
                    "likelihood_to_adopt": "medium",
                    "buying_behavior": ["trial first", "cancel if noisy"],
                    "willingness_to_pay": "low-to-medium",
                    "differentiators": ["privacy", "convenience"],
                }
            ]
        self.segments = segs
        return self.segments

    # -----------------------------------------------------------------------
    # 2) Archetypen
    # -----------------------------------------------------------------------
    def propose_customer_archetypes(self) -> List[Dict[str, Any]]:
        if not self.segments:
            self.segments = [{"name": "General users", "type": "B2C"}]

        system = (
            "You are a UX researcher. Return JSON only."
        )
        user = f"""
For each segment below, define two interview customers:
- labels: "critical" and "open_reflective"
- include: role_name, backstory, response_style, motivations[], objections[], dealbreakers[]

Segments:
{json.dumps(self.segments, ensure_ascii=False)}
Return:
{{ "archetypes": [ {{ "segment": "Segment Name", "customers": [{{...}}, {{...}}] }} ] }}
Keep it brief, in English.
        """.strip()

        data = _llm_json(system, user)
        arch = _safe_get(data, "archetypes", [])
        if not isinstance(arch, list) or not arch:
            # Fallback
            arch = []
            for s in self.segments:
                name = s.get("name", "Segment")
                arch.append(
                    {
                        "segment": name,
                        "customers": [
                            {
                                "label": "critical",
                                "role_name": f"Customer ({name}) — critical",
                                "backstory": "Skeptical, budget-conscious.",
                                "response_style": "short, to the point",
                                "motivations": ["save time", "save money"],
                                "objections": ["privacy", "hype"],
                                "dealbreakers": ["hidden costs"],
                            },
                            {
                                "label": "open_reflective",
                                "role_name": f"Customer ({name}) — open & reflective",
                                "backstory": "Curious, reflective.",
                                "response_style": "thoughtful, examples",
                                "motivations": ["learn", "improve routine"],
                                "objections": ["complexity"],
                                "dealbreakers": ["poor UX"],
                            },
                        ],
                    }
                )
        self.archetypes = arch
        return self.archetypes

    def segments_with_archetypes(self) -> List[Dict[str, Any]]:
        # Für die UI reicht es, die Segmente zurückzugeben (Archetypen werden nicht direkt gerendert)
        return self.segments or []

    # -----------------------------------------------------------------------
    # 3) Evidence-Digest
    # -----------------------------------------------------------------------
    def digest_evidence(self) -> Dict[str, Any]:
        """
        Erwartet: self.evidence_by_segment = { "Segment": [ {"title": "...", "url": "..."} ] }
        Liefert:  self.evidence_digest_by_segment = { seg: { references, facts, implications } }
        """
        digest: Dict[str, Any] = {}
        if not self.evidence_by_segment:
            self.evidence_digest_by_segment = {}
            return self.evidence_digest_by_segment

        system = "You are a meticulous desk researcher. Return JSON only."
        for seg, links in (self.evidence_by_segment or {}).items():
            # Referenzen mit IDs vorbereiten
            refs = []
            for i, it in enumerate(links or [], start=1):
                if not it.get("url"):
                    continue
                title = it.get("title") or it["url"]
                refs.append({"id": i, "title": title[:180], "url": it["url"]})

            # Wenn keine Links -> leeres Digest
            if not refs:
                digest[seg] = {"references": [], "facts": [], "implications": []}
                continue

            user = f"""
Build a compact evidence digest in JSON:
{{
  "references": [{{"id": N, "title": "...", "url": "..."}}],
  "facts": ["short, checkable claims, each ending with [ref N]"],
  "implications": ["optional bullets for interviews"]
}}
Use only what is plausibly supported by titles/landing pages. Neutral tone. English.

References:
{json.dumps(refs, ensure_ascii=False, indent=2)}
            """.strip()

            data = _llm_json(system, user, temperature=0.1, max_tokens=1000)
            # Robust extrahieren
            out = {
                "references": _safe_get(data, "references", refs),
                "facts": _safe_get(data, "facts", []),
                "implications": _safe_get(data, "implications", []),
            }
            digest[seg] = out

        self.evidence_digest_by_segment = digest
        return self.evidence_digest_by_segment

    # -----------------------------------------------------------------------
    # 4) Leitfäden
    # -----------------------------------------------------------------------
    def generate_interview_guidelines(
        self,
        business_idea: str,
        business_type_hint: Optional[str] = None,
        interview_goal: Optional[str] = None,
        max_questions: int = 12,
    ) -> List[Dict[str, Any]]:
        if not self.segments:
            self.segments = [{"name": "General users", "type": "B2C"}]

        system = "You are an interview designer. Return JSON only."
        user = f"""
For each segment, produce up to {max_questions} neutral, non-leading interview questions.
Include these mandatory questions verbatim:
1) "What would your dream product do?"
2) "What are the key features, qualities, functionalities, effects that you expect and why? What are the motivations and constraints behind those requests?"

Return:
{{ "business_type": "B2C|B2B",
  "segment_guidelines": [{{"segment": "...", "questions": ["...", "..."]}}]
}}
Use concise English. Avoid duplication.

Business idea:\n{business_idea.strip()}

Interview goal:\n{(interview_goal or '').strip()}

Segments:\n{json.dumps(self.segments, ensure_ascii=False)}
        """.strip()

        data = _llm_json(system, user, temperature=0.2, max_tokens=2000)
        sg = _safe_get(data, "segment_guidelines", [])
        if not isinstance(sg, list) or not sg:
            # Fallback-Minileitfaden
            sg = []
            mandatory = [
                "What would your dream product do?",
                "What are the key features, qualities, functionalities, effects that you expect and why? What are the motivations and constraints behind those requests?",
            ]
            for s in self.segments:
                base = [
                    "Walk me through your current process / routine and where this problem appears.",
                    "What have you already tried? What worked, what didn't?",
                    "What trade-offs are you making today?",
                ]
                questions = (mandatory + base)[:max_questions]
                sg.append({"segment": s.get("name", "Segment"), "questions": questions})

        # cap auf max_questions
        for blk in sg:
            qs = blk.get("questions", [])
            blk["questions"] = (qs or [])[:max_questions]

        self.segment_guidelines = sg
        return sg

    def enrich_guidelines_with_evidence(
        self,
        current_guidelines: List[Dict[str, Any]],
        max_questions: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Mind. 50% der Fragen sollen [ref N] aus dem Evidence-Digest enthalten.
        """
        if not current_guidelines:
            return []

        system = "You are an interview designer. Return JSON only."
        user = f"""
You will revise interview guidelines per segment so that >=50% questions reference evidence facts using [ref N].
Do not invent references. Keep questions neutral and within {max_questions} max.

Current guidelines:
{json.dumps(current_guidelines, ensure_ascii=False)}

Evidence digest:
{json.dumps(self.evidence_digest_by_segment or {}, ensure_ascii=False)}

Return:
{{ "segment_guidelines": [{{"segment":"...", "questions":["..."]}}] }}
        """.strip()

        data = _llm_json(system, user, temperature=0.2, max_tokens=1800)
        sg = _safe_get(data, "segment_guidelines", [])
        if not isinstance(sg, list) or not sg:
            # Falls LLM nichts liefert, die aktuellen Guidelines minimal anreichern
            enriched: List[Dict[str, Any]] = []
            for blk in current_guidelines:
                seg = blk.get("segment", "Segment")
                facts = (_safe_get(self.evidence_digest_by_segment.get(seg, {}), "facts", []) or [])[:3]
                qs = blk.get("questions", [])[:max_questions]
                # bezeichne einige Fragen mit [ref 1] wenn möglich
                ref_tag = "[ref 1]" if facts else ""
                for i in range(len(qs)):
                    if i % 2 == 1 and ref_tag and ref_tag not in qs[i]:
                        qs[i] = f"{qs[i]} {ref_tag}".strip()
                enriched.append({"segment": seg, "questions": qs})
            sg = enriched

        # cap
        for blk in sg:
            blk["questions"] = (blk.get("questions") or [])[:max_questions]

        self.segment_guidelines = sg
        return sg

    # -----------------------------------------------------------------------
    # 5) Interviews simulieren
    # -----------------------------------------------------------------------
    def run_interviews_per_segment(
        self,
        max_turns: int = 8,
        collect_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        if not self.segment_guidelines:
            self.segment_guidelines = [{"segment": "Segment", "questions": ["Tell me about it."]}]
        if not self.archetypes:
            self.propose_customer_archetypes()

        system = "You are a neutral interviewer and simulate short interviews. Return JSON only."
        user = f"""
Simulate two interviews per segment: one with customer label "critical" and one "open_reflective".
Use 1 question per turn from the provided guidelines (up to {max_turns} turns total).
Short follow-ups allowed. Be concise.

Guidelines:
{json.dumps(self.segment_guidelines, ensure_ascii=False)}

Archetypes:
{json.dumps(self.archetypes, ensure_ascii=False)}

Return:
{{
  "interviews": [
    {{
      "segment": "...",
      "customer_label": "critical" | "open_reflective",
      "transcript": [{{"question":"...", "answer":"..."}}],
      "metadata": {{"started_at":"...", "ended_at":"...", "notes":"..."}}
    }}
  ]
}}
        """.strip()

        data = _llm_json(system, user, temperature=0.3, max_tokens=2600)
        interviews = _safe_get(data, "interviews", [])
        if not isinstance(interviews, list) or not interviews:
            # Fallback: superkurze pseudo-Interviews
            interviews = []
            for g in self.segment_guidelines[:3]:
                seg = g.get("segment", "Segment")
                qs = (g.get("questions") or [])[:max_turns]
                for label in ("critical", "open_reflective"):
                    transcript = []
                    for q in qs:
                        transcript.append({"question": q, "answer": "Short, plausible answer."})
                    interviews.append(
                        {
                            "segment": seg,
                            "customer_label": label,
                            "transcript": transcript,
                            "metadata": {
                                "started_at": _now_iso(),
                                "ended_at": _now_iso(),
                                "notes": "",
                            },
                        }
                    )

        return interviews

    # -----------------------------------------------------------------------
    # 6) Synthese
    # -----------------------------------------------------------------------
    def synthesize_segment_findings(self) -> List[Dict[str, Any]]:
        system = "You are a qualitative analyst. Return JSON only."
        user = f"""
Synthesize interviews into findings per segment.
Return:
{{ "segment_summaries": [
  {{
    "segment": "...",
    "pain_points": ["..."],
    "key_needs": ["..."],
    "adoption_barriers_and_concerns": ["..."],
    "constraints": ["..."],
    "risks_unknowns": ["..."],
    "buying_signals": ["..."],
    "representative_quotes": ["..."],
    "narrative": "..."
  }}
] }}

Interviews:
{json.dumps(getattr(self, "interviews_cache", []), ensure_ascii=False)}

If no interviews provided, infer lightly from guidelines.
        """.strip()

        # Interviews können direkt aus run_interviews_per_segment kommen
        # Halte eine Cache-Variable bereit (App übergibt die Ergebnisliste zwar,
        # aber wir möchten robust sein).
        if not hasattr(self, "interviews_cache") or not self.interviews_cache:
            # nichts gesetzt -> versuche, sie aus letzter Ausführung zu lesen, ansonsten leer
            pass

        data = _llm_json(system, user, temperature=0.2, max_tokens=2200)
        summaries = _safe_get(data, "segment_summaries", [])
        if not isinstance(summaries, list) or not summaries:
            # Fallback
            summaries = []
            for s in (self.segments or [{"name": "Segment"}])[:3]:
                name = s.get("name", "Segment")
                summaries.append(
                    {
                        "segment": name,
                        "pain_points": ["fragmented workflow", "time pressure"],
                        "key_needs": ["clarity", "low friction"],
                        "adoption_barriers_and_concerns": ["privacy concerns", "unclear ROI"],
                        "constraints": ["budget", "time"],
                        "risks_unknowns": ["data reliability"],
                        "buying_signals": ["clear time savings"],
                        "representative_quotes": ["I just need the simplest path."],
                        "narrative": "Users want tangible relief without extra cognitive load.",
                    }
                )
        return summaries

    # -----------------------------------------------------------------------
    # 7) Anforderungen ableiten
    # -----------------------------------------------------------------------
    def derive_product_requirements(self) -> Dict[str, Any]:
        system = "You are a product owner. Return JSON only."
        user = f"""
Derive concrete, testable product requirements from segment findings (and interviews if present).
Return:
{{
  "cross_segment_requirements": [
    {{
      "id":"REQ-...","title":"...","category":"security|data|ux|performance|integration|compliance|operations|constraint|dont",
      "priority":"MUST|SHOULD|COULD|WONT","must_have":true,
      "description":"...",
      "acceptance_criteria":["Given/When/Then ..."],
      "rationale":"...",
      "depends_on":["REQ-..."],
      "anti_requirements":["..."],
      "effort_tshirt":"XS|S|M|L|XL",
      "tags":["..."]
    }}
  ],
  "per_segment_requirements":[
    {{
      "segment":"...",
      "requirements":[ {{ same structure as above }} ]
    }}
  ],
  "notes":"..."
}}

Segment summaries:
{json.dumps(getattr(self, "segment_summaries_cache", []), ensure_ascii=False)}

Interviews (optional):
{json.dumps(getattr(self, "interviews_cache", []), ensure_ascii=False)}
        """.strip()

        data = _llm_json(system, user, temperature=0.25, max_tokens=2600)
        # Robust auslesen
        out = {
            "cross_segment_requirements": _safe_get(data, "cross_segment_requirements", []),
            "per_segment_requirements": _safe_get(data, "per_segment_requirements", []),
            "notes": _safe_get(data, "notes", ""),
        }

        # Fallback minimal
        if not out["cross_segment_requirements"] and not out["per_segment_requirements"]:
            out["cross_segment_requirements"] = [
                {
                    "id": "REQ-PRIV-001",
                    "title": "Privacy-by-design defaults",
                    "category": "security",
                    "priority": "MUST",
                    "must_have": True,
                    "description": "Minimal data retention, clear consent, local processing where possible.",
                    "acceptance_criteria": [
                        "Given a new user, when onboarding, then consent is explicit and revocable.",
                        "Given user requests deletion, when confirmed, then all data is deleted within 30 days.",
                    ],
                    "rationale": "Common privacy concern across segments.",
                    "depends_on": [],
                    "anti_requirements": ["No dark patterns"],
                    "effort_tshirt": "M",
                    "tags": ["privacy", "trust"],
                }
            ]
            out["per_segment_requirements"] = []

        return out

    # -----------------------------------------------------------------------
    # Helfer, um App-Zwischenstände zu halten (optional)
    # -----------------------------------------------------------------------
    def set_interviews_cache(self, interviews: List[Dict[str, Any]]) -> None:
        self.interviews_cache = interviews

    def set_segment_summaries_cache(self, summaries: List[Dict[str, Any]]) -> None:
        self.segment_summaries_cache = summaries
