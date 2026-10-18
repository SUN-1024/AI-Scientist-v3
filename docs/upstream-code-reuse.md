# Actual V1, V2, and V3 code reuse

The Harness runs adapted V1 and V2 code inside the existing V3 executor and
evidence pipeline. It does not require either upstream checkout at runtime.
The audit checkouts in `scratch/` are reference copies, not runtime dependencies.

| Version | Pinned source | Reused implementation | Live integration |
| --- | --- | --- | --- |
| V1 | `SakanaAI/AI-Scientist`, `1de1dbc1f4ee2c5f61e9c94348d55eb51d7fa2eb` | `generate_ideas.py`: bounded iterative idea reflection, parse each round, preserve the original idea, archive the resulting proposal, stop when no change is needed; serialize previous ideas into the next proposal prompt | `src/research/upstream/v1_reflection.py`; agenda drafting calls the existing provider with full frozen source and selected-code context for every reflection round. Each call keeps its own receipt. Whole-idea portfolio prompts use `render_idea_archive` to retain earlier cards and negative outcomes. |
| V2 | `SakanaAI/AI-Scientist-v2`, `96bd51617cfdbb494a9fc283af00fe090edfae48` | `journal.py`: draft/debug/improve node semantics, consecutive debug depth, good/buggy node filters, best-node metric selection; `utils/metric.py`: optimize in the declared direction | `src/research/upstream/v2_journal.py`; `SearchTree.choose` ranks actual host-validated nodes, selects a bounded buggy leaf if no correct implementation exists, and supplies complete journal evidence to ideation. |
| V3 | Existing project code and its preserved upstream history | Task contracts, isolated coding and evaluation, native evaluation receipts, campaign execution, final confirmation, ablations, retrospective analysis, and manuscript generation/audit | The current task, investigation, pipeline, and paper modules remain the execution foundation. |

## Why the adaptations differ from upstream

The V1 reflection loop uses the existing agenda schema. It gets the complete
frozen evidence on each request. The selected candidate packet, when present,
includes actual source and observed evaluator outputs. A new author proposal
cannot silently become evidence that an experiment ran. At most three total
rounds run by default, including the initial draft. A parsed `_reflection.done`
flag stops the loop only when the normalized agenda is unchanged. A quoted
`I am done` string cannot stop it. Provider failures propagate to the existing
Harness failure handling. The reflection trace remains author work; independent
novelty review is still required.

The V2 adaptation uses only host-validated search records. It removes upstream
provider calls and optional numerical/serialization libraries. A missing metric
is worse than a valid metric. Non-finite values, different metric names,
conflicting optimization directions, and different dataset sets cannot be
compared. Multi-dataset values mean the same metric on the same datasets; loss
and accuracy are never averaged together. Replication nodes do not compete for
promotion. An explicit failed plot check removes a node from promotion, but the
adapter does not invent a successful plot check when no visual audit was run.
Candidate analysis and visual feedback stay in the selection record even when
the metric exists. The host's persisted lease pins the chosen parent on resume.

The integration does not copy V2's manager state bugs, implicit dataset-count
thresholds, first-three-seed aggregation, or free-form LLM winner selection.
The existing adaptive controller and new whole-idea portfolio controller use
separate execution modes. They retain responsibility for stage transitions,
protocol identity, repeated measurements, and handoff to final evaluation.

New Harness workspaces use a frozen whole-idea policy. The portfolio screens
several complete ideas, promotes a cohort, and jointly tunes each promoted
idea's allowed parameters from its own best result. It reuses V3's parallel
worker processes, GPU allocation, isolated evaluator, and validated receipts.
The selected candidate is frozen before repeatability checks. Selection scores
and repeat means remain separate. A development handoff does not establish an
independent result, causal component effect, or publication readiness.

The archive renderer adapts V1's previous-idea serialization in
`generate_ideas.py` lines 112–172. It preserves host root IDs and complete cards,
including unsuccessful trials. Bounded input validation rejects an oversized
archive instead of silently removing earlier ideas. The V2 journal adapter
continues to run in legacy search mode; portfolio scheduling uses its own
explicit cohort and per-idea state rather than claiming to execute V2's full
manager unchanged.

The bounded review-retrieval loop in `src/research/review_retrieval.py` follows
V1's `check_idea_novelty` and V2's `generate_temp_free_idea` query/result feedback
pattern. It is new host code, not a copied upstream search implementation.
Only replay-verified reviewer requests can start it. New primary full texts are
added in separate evidence views. The author then revises the interpretation,
and fresh independent reviewers assess it. All earlier results remain visible.
The loop has cumulative limits and never turns exhausted retries into a pass.
The existing V1 reflection and V2 journal adaptations above remain the actual
upstream code reuse; this retrieval loop does not claim to run either full
upstream research manager.

CLI-agent paper suggestions have no structured web-tool receipt in the current
Headless transport. `agent_literature.py` labels them as unverified suggestions.
The retrieval host downloads and checks each accepted PDF separately. This is
source acquisition and title matching, not proof of exhaustive search or truth.

## License and provenance

V3's existing code keeps Apache-2.0. Both pinned V1 and V2 snapshots carry the
**AI Scientist Source Code License, Version 1.0, December 2025**. The adapted
modules remain subject to that license. They are not relicensed as Apache-2.0.

Each `third_party/ai_scientist_v1/` and `third_party/ai_scientist_v2/` directory
contains the exact full upstream `LICENSE`, an attribution/modification `NOTICE`,
and `SOURCE.json` with pinned URLs and SHA-256 hashes of the original files.
The same source receipts are installed as
`src/research/upstream/provenance.json`. The wheel includes both licenses and
notices. The distribution's license expression lists both Apache-2.0 and
`LicenseRef-AI-Scientist-Source-1.0`.

The upstream license includes use restrictions and requires a prominent AI
Scientist disclosure in generated manuscripts, research papers, and technical
reports. Keep the full license and restrictions with derived distributions.
The Harness manuscript prompt and task disclosure must retain this disclosure;
the license file alone does not satisfy the manuscript requirement.

The Harness writes `research_context/upstream_integration.json` before freezing
the prepared task. This marker carries the pinned V1/V2 source identities and
the required statement. Snapshot ignore rules may not remove it. Paper drafting
receives the statement through the existing task context. Compilation fails with
repair feedback if the rendered PDF lacks attribution in its Abstract, Methods,
or a dedicated Disclosure section. TeX comments and watermarks do not pass this
check. Readiness inspects the current PDF again using the frozen input marker.
The checker does not insert or rewrite manuscript text. Legacy V3 tasks without
the marker retain their previous behavior. This check covers the specified
attribution statement; it does not certify full legal compliance or human review.

## Verification

`tests/research/test_upstream_code_reuse.py` checks reflection convergence,
retention of full frozen context, malformed output and provider errors,
direction-aware metric ordering, incomparable data rejection, replication and
plot eligibility, graph cycle rejection, debug-depth limits, and live search
lease replay. Existing search-tree and adaptive parent tests check compatibility.
`tests/research/test_novelty_gate.py` covers the agenda-provider integration and
candidate-bound independent reviews. These are software checks; only an actual
case run can establish the research behavior and resulting evidence quality.
