#!/usr/bin/env python3
"""Generate PRPACK numerical-fidelity fixtures for graphrag-rs PPR tests.

Uses python-igraph's PRPACK implementation as the ground-truth reference,
matching HippoRAG's canonical personalized_pagerank(implementation='prpack').

Usage:
    python3 tests/fixtures/generate_ppr_fixtures.py > tests/fixtures/ppr_fixtures.rs

Requires: python-igraph >= 0.11  (Nix: python3Packages.igraph)
  If missing, install with: pip install python-igraph
  On NixOS: nix-shell -p python3 python3Packages.igraph
"""

import sys

try:
    import igraph as ig
except ImportError:
    print(
        "ERROR: python-igraph is not installed.\n"
        "Install with: pip install python-igraph\n"
        "On NixOS:    nix-shell -p python3 python3Packages.igraph\n"
        "Then re-run: python3 tests/fixtures/generate_ppr_fixtures.py > tests/fixtures/ppr_fixtures.rs",
        file=sys.stderr,
    )
    sys.exit(1)


def compute_prpack(n, edges, reset, damping):
    """Compute PPR via igraph PRPACK.

    Parameters
    ----------
    n      : int   — number of nodes
    edges  : list of (src, dst, weight)
    reset  : list of (node_idx, weight)  — sparse personalisation vector
    damping: float — damping factor α

    Returns
    -------
    list of float — dense PPR scores, length n
    """
    g = ig.Graph(n=n, directed=True)
    if edges:
        srcs, dsts, wts = zip(*edges)
        g.add_edges(list(zip(srcs, dsts)))
        g.es["weight"] = list(wts)

    # Build dense reset vector (igraph takes dense list)
    reset_dense = [0.0] * n
    total_reset = sum(w for _, w in reset)
    for idx, w in reset:
        reset_dense[idx] = w / total_reset if total_reset > 0 else 1.0 / n

    use_weights = "weight" if edges else None
    scores = g.personalized_pagerank(
        reset=reset_dense,
        damping=damping,
        directed=True,
        weights=use_weights,
        implementation="prpack",
    )
    return list(scores)


def fmt_f64(v):
    """Format a float as a Rust f64 literal with enough precision."""
    return f"{v:.20e}"


def fmt_fixture(name, n, edges, reset, damping, scores):
    """Render one PprFixture Rust constant."""
    # Sort edges by (src, dst) for determinism
    sorted_edges = sorted(edges, key=lambda e: (e[0], e[1]))
    # Sort reset by node index
    sorted_reset = sorted(reset, key=lambda r: r[0])

    edges_str = ", ".join(f"({s}, {d}, {fmt_f64(w)})" for s, d, w in sorted_edges)
    reset_str = ", ".join(f"({i}, {fmt_f64(w)})" for i, w in sorted_reset)
    scores_str = ", ".join(fmt_f64(s) for s in scores)

    return f"""    PprFixture {{
        name: "{name}",
        n: {n},
        edges: &[{edges_str}],
        reset: &[{reset_str}],
        damping: {fmt_f64(damping)},
        expected_scores: &[{scores_str}],
    }}"""


def main():
    # ── Fixture definitions ──────────────────────────────────────────────────
    #
    # Each entry: (name, n, edges [(src,dst,weight)], reset [(idx,weight)], damping)
    # Sorted by name for determinism.

    fixtures_input = []

    # F1: chain_5 — linear chain A→B→C→D→E, personalised at A (node 0)
    # Expected: geometric decay, ~0.516, 0.258, 0.129, 0.065, 0.032
    fixtures_input.append((
        "chain_5",
        5,
        [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
        [(0, 1.0)],
        0.5,
    ))

    # F2: clique_5 — fully connected 5-node graph, uniform personalisation
    # Expected: uniform ~0.2 each (symmetry)
    n2 = 5
    clique_edges = [(i, j, 1.0) for i in range(n2) for j in range(n2) if i != j]
    fixtures_input.append((
        "clique_5",
        n2,
        clique_edges,
        [(i, 1.0) for i in range(n2)],  # uniform
        0.5,
    ))

    # F3: dangling_6 — several sinks (out_degree=0) with incoming edges.
    # Topology: 0→1→2(sink), 0→3→4(sink), 1→5(sink). Personalised at node 0.
    # Tests that dangling-mass redistribution gives non-zero to all reachable nodes.
    fixtures_input.append((
        "dangling_6",
        6,
        [(0, 1, 1.0), (0, 3, 1.0), (1, 2, 1.0), (1, 5, 1.0), (3, 4, 1.0)],
        [(0, 1.0)],
        0.5,
    ))

    # F4: disconnected_5 — two components: {0,1,2} cycle and {3,4} isolated.
    # Personalised only in component {0,1,2}. Verifies score doesn't leak to {3,4}.
    # Note: nodes 3,4 have no edges at all — they are fully isolated.
    fixtures_input.append((
        "disconnected_5",
        5,
        [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
        [(0, 1.0)],
        0.5,
    ))

    # F5: kg_18 — KG-shaped ~18 nodes, mixed degree, asymmetric, weighted edges.
    # Closest to real entity graphs in miniature. Personalised at node 0.
    kg_edges = [
        (0, 1, 1.0),  (0, 2, 0.5),  (1, 3, 1.0),  (1, 4, 0.5),
        (2, 4, 1.0),  (2, 5, 0.5),  (3, 6, 1.0),  (4, 6, 0.5),
        (4, 7, 1.0),  (5, 8, 0.5),  (6, 9, 1.0),  (7, 9, 0.5),
        (7, 10, 1.0), (8, 11, 0.5), (9, 12, 1.0), (10, 12, 0.5),
        (10, 13, 1.0), (11, 14, 0.5), (12, 15, 1.0), (13, 15, 0.5),
        (14, 16, 1.0), (15, 17, 0.5), (16, 17, 1.0), (17, 0, 0.5),
        (6, 1, 1.0),  (12, 3, 0.5),
    ]
    fixtures_input.append((
        "kg_18",
        18,
        kg_edges,
        [(0, 1.0)],
        0.5,
    ))

    # Sort fixtures by name for determinism
    fixtures_input.sort(key=lambda x: x[0])

    # ── Compute PRPACK scores ────────────────────────────────────────────────
    computed = []
    for name, n, edges, reset, damping in fixtures_input:
        scores = compute_prpack(n, edges, reset, damping)
        computed.append((name, n, edges, reset, damping, scores))

    # ── Emit Rust source ─────────────────────────────────────────────────────
    lines = []
    lines.append("// Auto-generated by tests/fixtures/generate_ppr_fixtures.py")
    lines.append("// DO NOT EDIT manually.")
    lines.append("//")
    lines.append("// Regenerate with:")
    lines.append("//   nix-shell -p python3 python3Packages.igraph \\")
    lines.append("//     --run 'python3 tests/fixtures/generate_ppr_fixtures.py \\")
    lines.append("//     > tests/fixtures/ppr_fixtures.rs'")
    lines.append("//")
    lines.append(f"// igraph version: {ig.__version__}")
    lines.append("")
    lines.append("/// One ground-truth PPR fixture computed by igraph PRPACK.")
    lines.append("pub struct PprFixture {")
    lines.append("    /// Human-readable fixture name")
    lines.append("    pub name: &'static str,")
    lines.append("    /// Number of nodes")
    lines.append("    pub n: usize,")
    lines.append("    /// Edges as (src, dst, weight)")
    lines.append("    pub edges: &'static [(usize, usize, f64)],")
    lines.append("    /// Sparse personalisation vector as (node_idx, weight)")
    lines.append("    pub reset: &'static [(usize, f64)],")
    lines.append("    /// Damping factor α")
    lines.append("    pub damping: f64,")
    lines.append("    /// Ground-truth PPR scores from igraph PRPACK, dense, length n")
    lines.append("    pub expected_scores: &'static [f64],")
    lines.append("}")
    lines.append("")
    lines.append("/// All PPR fixtures, sorted by name.")
    lines.append("pub const FIXTURES: &[PprFixture] = &[")

    fixture_strs = []
    for name, n, edges, reset, damping, scores in computed:
        fixture_strs.append(fmt_fixture(name, n, edges, reset, damping, scores))
    lines.append(",\n".join(fixture_strs))

    lines.append("];")
    lines.append("")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
