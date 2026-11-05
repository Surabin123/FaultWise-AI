"""Deterministic sufficiency and structural grounding checks."""

import re

from .schemas import EvidenceItem, GeneratedContent

PROMPT_VERSION = "diagnostic-v3"
_UNSUPPORTED = re.compile(r"\b(confirmed|failed|damaged|replace|repair|manufacturer|manual|guaranteed|probability|confidence)\b|%", re.I)
_NUMBER = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")
_COMPONENT = re.compile(r"\b(turbine|blade|bearing|compressor|combustor|rotor|fan|gearbox|shaft|fuel pump|nozzle)\b", re.I)
_DIRECTIVE = re.compile(r"\b(ignore|disregard|bypass|override|invent|fabricate|pretend|reveal|expose|leak)\b", re.I)
_SECRET = re.compile(r"\b(api[_ -]?key|gemini_api_key|credential|secret|password|bearer|access[_ -]?token)\b|\b(?:sk-[A-Za-z0-9]{8,}|AIza[A-Za-z0-9_-]{16,})\b", re.I)


def missing_evidence(evidence: list[EvidenceItem], *, anomaly_analyzed: bool, has_anomaly: bool) -> list[str]:
    types = {item.type for item in evidence}
    missing: list[str] = []
    if "prediction" not in types:
        missing.append("RUL prediction")
    if "observation" not in types:
        missing.append("latest sensor measurements")
    if not anomaly_analyzed:
        missing.append("sufficient trajectory for anomaly analysis")
    if not has_anomaly:
        missing.append("corroborating unusual sensor behavior for a diagnostic concern")
    return missing


def validate_generated(content: GeneratedContent, evidence: list[EvidenceItem]) -> list[str]:
    valid_ids = {item.id for item in evidence}
    anomaly_ids = {item.id for item in evidence if item.type == "anomaly"}
    # The current evidence contract has no component-level diagnostic source.
    component_ids: set[str] = set()
    errors: list[str] = []
    for claim in content.concerns + content.recommended_investigation:
        if not set(claim.evidence_ids).issubset(valid_ids):
            errors.append("claim references an unknown evidence ID")
        if _UNSUPPORTED.search(claim.text) or _NUMBER.search(claim.text):
            errors.append("claim contains an unsupported factual or numerical assertion")
        if (claim.scope == "component_condition" or _COMPONENT.search(claim.text)) and not set(claim.evidence_ids) & component_ids:
            errors.append("component-specific claim lacks component-level evidence")
        if _DIRECTIVE.search(claim.text):
            errors.append("claim contains an instruction to override evidence rules")
        if _SECRET.search(claim.text):
            errors.append("claim contains a secret or credential reference")
    for concern in content.concerns:
        if not set(concern.evidence_ids) & anomaly_ids:
            errors.append("diagnostic concern lacks anomaly evidence")
    for value in [content.uncertainty, *content.limitations]:
        if not value.strip():
            errors.append("empty uncertainty or limitation")
        if _UNSUPPORTED.search(value) or _NUMBER.search(value):
            errors.append("uncertainty or limitation contains an unsupported assertion")
        if _DIRECTIVE.search(value) or _SECRET.search(value) or _COMPONENT.search(value):
            errors.append("uncertainty or limitation contains unsafe or unsupported content")
    if not content.concerns:
        errors.append("a supported hypothesis is required")
    return errors
