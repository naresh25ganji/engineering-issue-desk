import json
import re

import pytest

from triage_desk.knowledge_graph import EngineeringKnowledgeGraph


@pytest.fixture
def graph_sources():
    provenance = {"snapshot_at": "2026-10-18T00:00:00Z", "source_commit": "fixture-doc-commit",
                  "version_scope": "Fixture snapshot; not a compatibility guarantee."}
    return [
        {**provenance, "source_id": "doc-state", "title": "st.session_state", "kind": "documentation",
         "url": "https://docs.streamlit.io/example/state", "text": "Use st.session_state between reruns. See st.button callbacks.",
         "api_version": "1.66.0"},
        {**provenance, "source_id": "issue-state", "title": "State report", "kind": "issue", "state": "closed",
         "url": "https://github.com/streamlit/streamlit/issues/1", "source_commit": None,
         "text": "The reporter used st.session_state and st.button. They mention Python 3.12.0 and a possible st.ghost_widget.",
         "reported_versions": ["1.44.1"]},
        {**provenance, "source_id": "issue-other", "title": "Upload report", "kind": "issue", "state": "open",
         "url": "https://github.com/streamlit/streamlit/issues/2", "source_commit": None,
         "text": "st.file_uploader on an example environment 1.44.1; this prose is not a version metadata field.",
         "reported_versions": []},
        {**provenance, "source_id": "release", "title": "Release 1.66.0", "kind": "release",
         "url": "https://docs.streamlit.io/example/releases", "text": "This release mentions st.file_uploader."},
    ]


def test_edges_preserve_explicit_provenance_without_fix_or_causality(graph_sources):
    graph = EngineeringKnowledgeGraph(graph_sources)
    view = graph.explore("st.session_state", max_nodes=60)
    relations = {edge["relation"] for edge in view["edges"]}
    assert relations <= {"MENTIONS_API", "REPORTS_VERSION", "HAS_KIND", "HAS_STATE"}
    version = [edge for edge in view["edges"] if edge["relation"] == "REPORTS_VERSION"]
    assert len(version) == 1
    assert version[0]["source"] == "source:issue-state"
    assert version[0]["provenance"]["source_commit"] is None
    assert version[0]["provenance"]["url"].endswith("/issues/1")
    assert version[0]["basis"] == "source.reported_versions"
    assert graph.graph.has_edge("source:issue-state", "state:closed", "HAS_STATE")
    assert any("closed is not proof of a fix" in edge.get("meaning", "") for edge in view["edges"])
    assert not any(node["id"] in {"version:3.12.0", "version:1.66.0"} for node in view["nodes"])


def test_issue_only_symbol_is_not_promoted_to_documented_api(graph_sources):
    graph = EngineeringKnowledgeGraph(graph_sources)
    symbols = {node["id"]: node for node in graph.explore("st.ghost_widget")["nodes"]}
    assert symbols["api:st.ghost_widget"]["documented_api_subject"] is False
    assert symbols["api:st.ghost_widget"]["issue_mention_source_ids"] == ["issue-state"]
    canonical = graph.graph.nodes["api:st.session_state"]
    assert canonical["documented_api_subject"] is True
    assert canonical["api_reference_source_ids"] == ["doc-state"]
    assert graph.graph.nodes["api:st.button"]["documented_api_subject"] is False


def test_version_search_uses_metadata_only_and_combines_exact_seeds(graph_sources):
    graph = EngineeringKnowledgeGraph(graph_sources)
    assert graph.search_sources("Reported Streamlit version 1.44.1") == ["issue-state"]
    assert graph.search_sources("st.session_state in 1.44.1", limit=2) == ["issue-state", "doc-state"]
    assert graph.search_sources("3.12.0") == []
    assert graph.search_sources("1.66.0") == []
    assert graph.search_sources("1.44") == []  # No inferred affected-version range.


def test_no_semantic_fallback_or_generic_issue_word_expansion(graph_sources):
    graph = EngineeringKnowledgeGraph(graph_sources)
    assert graph.search_sources("My screen changed after I clicked something") == []
    assert graph.search_sources("closed issue") == []
    view = graph.explore("My screen changed after I clicked something")
    assert view["nodes"] == []
    assert view["edges"] == []
    assert any("No exact" in note for note in view["notes"])


def test_two_hop_view_does_not_expand_through_kind_to_unrelated_source(graph_sources):
    graph = EngineeringKnowledgeGraph(graph_sources)
    view = graph.explore("st.session_state")
    ids = {node["id"] for node in view["nodes"]}
    assert "source:issue-state" in ids
    assert "kind:issue" in ids
    assert "source:issue-other" not in ids  # Would require a third hop.
    assert view["max_hops"] == 2


@pytest.mark.parametrize("budget", [1, 5, 20])
def test_view_is_bounded_deterministic_and_json_serializable(graph_sources, budget):
    graph = EngineeringKnowledgeGraph(graph_sources)
    first = graph.explore("st.session_state st.file_uploader", max_nodes=budget)
    assert first == graph.explore("st.session_state st.file_uploader", max_nodes=budget)
    ids = {node["id"] for node in first["nodes"]}
    assert len(ids) <= budget
    assert all(edge["source"] in ids and edge["target"] in ids for edge in first["edges"])
    json.dumps(first)


def test_graph_naturally_exceeds_twenty_nodes_without_invented_relationships():
    sources = [{"source_id": f"doc-{i}", "kind": "documentation", "title": f"API {i}",
                "text": f"Literal example st.widget_{i}.", "url": f"https://docs.example.test/{i}"}
               for i in range(25)]
    graph = EngineeringKnowledgeGraph(sources)
    summary = graph.summary()
    assert summary["source_count"] == 25
    assert summary["api_count"] == 25
    assert summary["node_count"] == 51
    assert summary["relations"] == {"HAS_KIND": 25, "MENTIONS_API": 25}


def test_dot_escapes_untrusted_labels_and_uses_internal_node_aliases(graph_sources):
    hostile = 'Bad"]; injected [label="owned"];\nsecond line'
    graph_sources[0]["title"] = hostile
    graph = EngineeringKnowledgeGraph(graph_sources)
    dot = graph.to_dot("st.session_state")
    assert hostile not in dot
    assert '\\"' in dot and '\\n' in dot
    assert not re.search(r"^\s*injected\s*\[", dot, re.MULTILINE)
    labels = [json.loads(match) for match in re.findall(r'label=("(?:[^"\\]|\\.)*")', dot)]
    assert any(' '.join(hostile.split()) in ' '.join(label.split()) for label in labels)
    assert any(node.get('label') == hostile for node in graph.explore('st.session_state')['nodes'])


def test_invalid_version_metadata_is_not_silently_interpreted(graph_sources):
    graph_sources[1]["reported_versions"] = "1.44.1"
    graph = EngineeringKnowledgeGraph(graph_sources)
    assert graph.summary()["reported_version_count"] == 0
    assert any("malformed reported_versions" in note for note in graph.summary()["notes"])


def test_duplicate_source_id_is_rejected(graph_sources):
    with pytest.raises(ValueError, match="unique"):
        EngineeringKnowledgeGraph([graph_sources[0], graph_sources[0]])
