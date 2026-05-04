---
name: flysuite-fos-graph
description: "Use the read-only FlySuiteFOS graph MCP to inspect FHM ontology, entities, relations, timelines, and object-set aggregates."
version: 0.1.0
author: FlySuite
license: MIT
metadata:
  hermes:
    tags: [FlySuite, FOS, FHM, OMS, WMS, ERP, Graph, Ontology]
    category: flysuite
---

# FlySuite FOS Graph

Use this skill whenever the user asks about FlySuite FOS, FHM ontology, OMS/WMS/ERP relationships, graph traversal, entity state, timelines, stock, orders, sales, products, vendors, warehouses, or how objects connect across FlySuite systems.

## Safety Rules

- Use only the read-only FOS graph tools. Do not ask for raw SQL and do not claim write access.
- Keep `schema` as `fhm`.
- Keep `include_pii` false unless the user explicitly asks for PII and the environment permits it. Secrets are never valid output.
- Treat `sourceRefs`, `lineage`, and read-model notes as important evidence. Mention them when they affect confidence or source-of-truth decisions.
- Do not tell the user about internal tool wiring, MCP transport, plugins, hooks, local ports, or memory sync.

## Query Pattern

1. For an unfamiliar domain or object name, start with `fos_describe_ontology` using `q`, `domain`, or `type`.
2. For a specific SKU, document number, tracking number, UUID, or identifier, use `fos_search_entities`.
3. After finding a target entity, use `fos_get_entity` with `include_claims=true` and `include_relations=true` when relationship context matters.
4. Use `fos_traverse_graph` when the user asks "what is connected to this", "where did this come from", or "how does it relate".
5. Use `fos_get_timeline` when the user asks what happened over time.
6. Use `fos_explain_path` before multi-hop reasoning between entity types.
7. Use `fos_query_object_set` for counts, aggregates, grouped reporting, and filtered object lists. Its DSL must use registered entity types and claim predicates.

## Source-Of-Truth Notes

- Current state lives in active entities plus `current_claims` and `current_relations`.
- For current inventory questions, prefer the `warehouse_stock_balance` read model semantics when available; `warehouse_stock` is graph context.
- For period reporting, prefer the business date called out by ontology metadata, not write-time `created_at`, when the registry says so.
- If the graph result and user expectation conflict, explain what the graph currently shows and identify the entity type, relation, claim, or read model that produced it.

## Response Style

- Answer in the user's language.
- Start with the concrete result, then give the relationship path or evidence.
- If the result is incomplete, say exactly which lookup would narrow it further.
- Keep graph details readable: use names, labels, predicates, counts, and lineage rather than raw JSON.
