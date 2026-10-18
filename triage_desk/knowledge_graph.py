"""A provenance graph over explicit source facts, not diagnoses or semantic search.

This adapts entity-seeded local graph expansion to engineering documents. An
edge says that a source literally mentions an API identifier or carries a
metadata value. It never says two issues are duplicates or a release fixes one.
"""
from __future__ import annotations

from collections import Counter, deque
import json
import re
import textwrap
from typing import Any

import networkx as nx


_API = re.compile(r"(?<![A-Za-z0-9_])st\.[A-Za-z_][A-Za-z0-9_]*")
_VERSION = re.compile(r"(?<![A-Za-z0-9_.])v?\d+\.\d+(?:\.\d+)?(?:[ab]\d+|rc\d+)?(?:[+~-][A-Za-z0-9.-]+)?(?![A-Za-z0-9_.])")
_VERSION_VALUE = re.compile(r"v?\d+\.\d+(?:\.\d+)?(?:[ab]\d+|rc\d+)?(?:[+~-][A-Za-z0-9.-]+)?")
_RELATIONS = {"MENTIONS_API", "REPORTS_VERSION", "HAS_KIND", "HAS_STATE"}
_MAX_EXPANSION_NODES = 600
_NOTES = [
    "Edges describe literal source mentions or explicit metadata; they do not establish causality, duplicate issues, compatibility, affected-version ranges, or fixes.",
    "REPORTS_VERSION uses only an issue's reported_versions field. Versions in prose, api_version, source URLs and release titles do not create reported-version edges.",
    "An API mention is not an official API definition. documented_api_subject requires an exact st.<identifier> documentation title; other mentions retain their source kind.",
    "Entity linking is exact and local; graph retrieval has no semantic or synonym fallback. Traversal is bounded to two hops in either direction while preserving edge orientation.",
]


class EngineeringKnowledgeGraph:
    def __init__(self, sources: list[dict[str, Any]]):
        if not isinstance(sources, list):
            raise ValueError("Graph sources must be a list of source records")
        self.graph = nx.MultiDiGraph()
        self._sources: dict[str, dict[str, Any]] = {}
        self._warnings: list[str] = []
        for source in sources:
            if not isinstance(source, dict) or any(
                not isinstance(source.get(field), str) for field in ("source_id", "title", "kind", "text")
            ):
                raise ValueError("Graph source records require text source_id, title, kind and text fields")
            source_id = source["source_id"]
            if not source_id.strip() or source_id in self._sources:
                raise ValueError("Graph source IDs must be nonempty and unique")
            if not source["kind"].strip():
                raise ValueError("Graph source kinds must be nonempty")
            self._sources[source_id] = dict(source)
            self._add_source(source)
        self._finish_api_metadata()

    @staticmethod
    def _provenance(source: dict[str, Any]) -> dict[str, Any]:
        return {field: source.get(field) for field in
                ("source_id", "url", "snapshot_at", "source_commit", "version_scope")}

    def _edge(self, source: dict[str, Any], target: str, relation: str, **details: Any) -> None:
        self.graph.add_edge("source:" + source["source_id"], target, key=relation,
                            relation=relation, provenance=self._provenance(source), **details)

    def _add_source(self, source: dict[str, Any]) -> None:
        source_id, kind = source["source_id"], source["kind"]
        self.graph.add_node("source:" + source_id, type="source", label=source["title"],
                            source_id=source_id, source_kind=kind,
                            **{field: source.get(field) for field in
                               ("url", "snapshot_at", "source_commit", "version_scope")})
        kind_id = "kind:" + kind
        self.graph.add_node(kind_id, type="source_kind", label=kind, value=kind)
        self._edge(source, kind_id, "HAS_KIND", basis="source.kind")

        counts = Counter(_API.findall(source["title"] + "\n" + source["text"]))
        title_subject = source["title"].strip().strip("`")
        for symbol, count in sorted(counts.items()):
            node_id = "api:" + symbol
            self.graph.add_node(node_id, type="api", label=symbol, value=symbol)
            origin_fields = [field for field in ("title", "text") if symbol in _API.findall(source[field])]
            self._edge(source, node_id, "MENTIONS_API", basis="literal_text_mention",
                       literal=symbol, mention_count=count, origin_fields=origin_fields,
                       source_kind=kind,
                       documented_api_subject=(kind == "documentation" and title_subject == symbol))

        if kind != "issue":
            return
        state = source.get("state")
        if isinstance(state, str) and state.strip():
            state = state.strip().lower()
            state_id = "state:" + state
            self.graph.add_node(state_id, type="issue_state", label=state, value=state)
            self._edge(source, state_id, "HAS_STATE", basis="source.state",
                       meaning="Issue state at the source snapshot; closed is not proof of a fix")

        versions = source.get("reported_versions", [])
        if not isinstance(versions, list):
            self._warnings.append(f"{source_id}: ignored malformed reported_versions; expected a list")
            return
        for original in sorted({value for value in versions if isinstance(value, str)}):
            value = original.strip()
            if not _VERSION_VALUE.fullmatch(value):
                self._warnings.append(f"{source_id}: ignored a non-version reported_versions value")
                continue
            version = value.removeprefix("v")
            version_id = "version:" + version
            self.graph.add_node(version_id, type="reported_version", label="Streamlit " + version + " (reported)",
                                value=version)
            self._edge(source, version_id, "REPORTS_VERSION", basis="source.reported_versions",
                       reported_value=original,
                       meaning="Explicit reporter-version metadata; not a compatibility or fix assertion")

    def _finish_api_metadata(self) -> None:
        for node_id, attributes in self.graph.nodes(data=True):
            if attributes["type"] != "api":
                continue
            subjects, docs, issues = [], [], []
            for source_node, _, relation, edge in self.graph.in_edges(node_id, keys=True, data=True):
                if relation != "MENTIONS_API":
                    continue
                source_id = self.graph.nodes[source_node]["source_id"]
                if edge["documented_api_subject"]:
                    subjects.append(source_id)
                if edge["source_kind"] == "documentation":
                    docs.append(source_id)
                elif edge["source_kind"] == "issue":
                    issues.append(source_id)
            attributes.update(documented_api_subject=bool(subjects),
                              api_reference_source_ids=sorted(subjects),
                              documentation_mention_source_ids=sorted(docs),
                              issue_mention_source_ids=sorted(issues))

    def summary(self) -> dict[str, Any]:
        node_types = Counter(data["type"] for _, data in self.graph.nodes(data=True))
        relations = Counter(data["relation"] for _, _, data in self.graph.edges(data=True))
        return {"node_count": self.graph.number_of_nodes(), "edge_count": self.graph.number_of_edges(),
                "source_count": node_types["source"], "api_count": node_types["api"],
                "reported_version_count": node_types["reported_version"],
                "source_kind_count": node_types["source_kind"], "issue_state_count": node_types["issue_state"],
                "node_types": dict(sorted(node_types.items())), "relations": dict(sorted(relations.items())),
                "notes": [*_NOTES, *self._warnings]}

    @staticmethod
    def _query(query: str) -> str:
        if not isinstance(query, str) or len(query) > 6000 or "\x00" in query:
            raise ValueError("Graph query must be text without nulls, at most 6,000 characters")
        return query.strip()

    def _seeds(self, query: str, *, allow_metadata: bool = False) -> list[str]:
        seeds = {"api:" + value for value in _API.findall(query)}
        seeds.update("version:" + value.removeprefix("v") for value in _VERSION.findall(query))
        if allow_metadata:
            # Explicit node IDs enable source/kind/state inspection without
            # interpreting ordinary prose words such as "issue" as filters.
            if query in self.graph:
                seeds.add(query)
            if "source:" + query in self.graph:
                seeds.add("source:" + query)
        return sorted(seed for seed in seeds if seed in self.graph)

    def _overview_seeds(self) -> list[str]:
        by_kind: dict[str, list[str]] = {}
        for node_id, data in self.graph.nodes(data=True):
            if data["type"] == "source":
                by_kind.setdefault(data["source_kind"], []).append(node_id)
        return [sorted(nodes, key=lambda node: (-self.graph.degree(node), node))[0]
                for _, nodes in sorted(by_kind.items())]

    def _expand(self, seeds: list[str], max_nodes: int) -> tuple[list[str], bool]:
        selected: list[str] = []
        seen = set()
        queue: deque[tuple[str, int]] = deque()
        truncated = False
        for seed in seeds:
            if seed in seen:
                continue
            if len(selected) >= max_nodes:
                truncated = True
                break
            selected.append(seed)
            seen.add(seed)
            queue.append((seed, 0))
        while queue:
            node, depth = queue.popleft()
            if depth >= 2:
                continue
            neighbors = sorted(set(self.graph.predecessors(node)) | set(self.graph.successors(node)))
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                if len(selected) >= max_nodes:
                    truncated = True
                    continue
                seen.add(neighbor)
                selected.append(neighbor)
                queue.append((neighbor, depth + 1))
        return selected, truncated

    def explore(self, query: str = "", max_nodes: int = 60) -> dict[str, Any]:
        query = self._query(query)
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= 600:
            raise ValueError("max_nodes must be an integer from 1 to 600")
        seeds = self._seeds(query, allow_metadata=True) if query else self._overview_seeds()
        notes = [*_NOTES, *self._warnings]
        if query and not seeds:
            notes.append("No exact known API/version seed or explicit node ID matched; no semantic fallback was applied.")
        elif not query:
            notes.append("Overview starts with a representative source from each source kind; it is not a relevance ranking.")
        selected, truncated = self._expand(seeds, max_nodes)
        included = set(selected)
        nodes = [{"id": node, **dict(self.graph.nodes[node])} for node in sorted(selected)]
        edges = []
        for origin, target, key, data in self.graph.edges(keys=True, data=True):
            if origin in included and target in included:
                edges.append({"source": origin, "target": target, **dict(data)})
        edges.sort(key=lambda item: (item["source"], item["target"], item["relation"]))
        if truncated:
            notes.append("Node budget reached; this is a partial two-hop view, not the entire connected graph.")
        return {"nodes": nodes, "edges": edges, "notes": notes,
                "seed_node_ids": seeds, "truncated": truncated, "max_hops": 2,
                "total_node_count": self.graph.number_of_nodes()}

    def search_sources(self, question: str, limit: int = 12) -> list[str]:
        question = self._query(question)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        seeds = self._seeds(question)
        if not seeds:
            return []
        nodes, _ = self._expand(seeds, _MAX_EXPANSION_NODES)
        candidates = [node for node in nodes if self.graph.nodes[node]["type"] == "source"]
        # This score is a count of exact graph seeds adjacent to the source, not
        # a probability, semantic score, or claim that the source answers fully.
        hits = lambda node: sum(self.graph.has_edge(node, seed) or self.graph.has_edge(seed, node) for seed in seeds)
        ranked = sorted(candidates, key=lambda node: (-hits(node), self.graph.nodes[node]["source_id"]))
        return [self.graph.nodes[node]["source_id"] for node in ranked[:limit]]

    def to_dot(self, query: str = "", max_nodes: int = 60) -> str:
        view = self.explore(query, max_nodes)
        aliases = {node["id"]: f"n{index}" for index, node in enumerate(view["nodes"])}
        shapes = {"source": "box", "api": "ellipse", "reported_version": "diamond",
                  "source_kind": "hexagon", "issue_state": "octagon"}
        quoted = lambda text: json.dumps(str(text), ensure_ascii=False)
        lines = ["digraph EngineeringKnowledgeGraph {", '  graph [rankdir="LR", bgcolor="transparent", nodesep="0.3"];',
                 '  node [fontname="Arial", fontsize="12", color="#839b93", fontcolor="#18322d", style="filled", margin="0.12,0.08"];',
                 '  edge [fontname="Arial", fontsize="10", color="#8aa198", fontcolor="#45645a"];']
        colors = {'source': '#edf3f0', 'api': '#d2e9fa', 'reported_version': '#fff0d2', 'source_kind': '#e2e8e5', 'issue_state': '#f0dfe8'}
        for node in view["nodes"]:
            label = node["label"]
            if len(label) > 70:
                label = label[:67] + "..."
            label = '\n'.join(textwrap.wrap(label, width=32, break_long_words=False, break_on_hyphens=False))
            if node['type'] == 'source':
                label = node.get('source_kind', 'source').title() + '\n' + label
            lines.append(f'  {aliases[node["id"]]} [label={quoted(label)}, shape={shapes[node["type"]]}, fillcolor={quoted(colors[node["type"]])}];')
        for edge in view["edges"]:
            lines.append(f'  {aliases[edge["source"]]} -> {aliases[edge["target"]]} [label={quoted(edge["relation"])}];')
        lines.append("}")
        return "\n".join(lines)
