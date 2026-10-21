"""Deterministic local retrieval over policy content already encoded in the project."""

from __future__ import annotations

import re
from typing import Any, Iterable, Protocol

from models.policy_retrieval import PolicyDocument


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


class PolicyRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[PolicyDocument]: ...


class LocalPolicyRetriever:
    """Keyword retriever requiring no embeddings, network, or credentials."""

    def __init__(self, documents: Iterable[PolicyDocument] | None = None):
        self.documents = tuple(documents or ())

    @classmethod
    def from_rules(cls, rules) -> "LocalPolicyRetriever":
        documents = []
        for rule in rules:
            content = (
                f"{rule.description}. Evaluation: {rule.state_path} "
                f"{rule.operator.value} {rule.expected_value!r}. "
                f"Failure outcome: {rule.on_fail}. Mandatory: {rule.mandatory}."
            )
            documents.append(PolicyDocument(
                id=rule.rule_id,
                title=rule.name,
                content=content,
                source="internal_policy_rule",
                metadata={
                    "rule_id": rule.rule_id,
                    "state_path": rule.state_path,
                    "operator": rule.operator.value,
                    "expected_value": rule.expected_value,
                    "mandatory": rule.mandatory,
                    "on_fail": rule.on_fail,
                },
            ))
        return cls(documents)

    def retrieve(self, query: str, *, top_k: int = 5,
            metadata: dict[str, Any] | None = None) -> list[PolicyDocument]:
        query_tokens = _tokens(query)
        if not query_tokens or top_k <= 0:
            return []
        filters = metadata or {}
        ranked = []
        for document in self.documents:
            if any(document.metadata.get(key) != value for key, value in filters.items()):
                continue
            searchable = " ".join((document.id, document.title, document.content,
                " ".join(str(value) for value in document.metadata.values())))
            overlap = query_tokens & _tokens(searchable)
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            if document.id.casefold() in query.casefold():
                score += 1.0
            ranked.append((score, document.id, document))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [document.model_copy(update={"score": round(score, 6)})
            for score, _id, document in ranked[:top_k]]


def context_for_policy_results(retriever: PolicyRetriever, results, *, top_k: int = 5) -> list[PolicyDocument]:
    """Retrieve failed/referred rule context, or all evaluated rules on a clean pass."""
    relevant = [result for result in results if result.status != "PASS"] or list(results)
    found: dict[str, PolicyDocument] = {}
    for result in relevant:
        for document in retriever.retrieve(result.rule_id, top_k=1):
            found[document.id] = document
    return [found[key] for key in sorted(found)][:top_k]
