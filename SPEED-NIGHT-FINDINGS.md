# GLM-5.2 Speed Night — Findings (2026-07-09)

**Question:** is GLM-5.2 decode on 4× DGX Spark at its physics limit (28.8 tok/s), or is there attackable overhead?
**Answer: NOT physics. 63% of every decode step is attackable overhead.** But most of it is unreachable with config flags — collecting it requires custom communication engineering.

---

## The headline measurement: step-time decomposition (k=0 test)

Running with speculation OFF makes tok/s = engine steps/sec directly:

| | value |
|---|---|
| k=0 single-stream | 14.9 tok/s → **67ms per raw engine step** |
| physics floor (weights @ 273GB/s, ~6GB/node/step) | **~25ms (37%)** |
| **overhead (NCCL latency, kernel launch, sync)** | **~42ms (63%)** |

Second finding from the same test: **MTP speculation is doing 2× of the work** (14.9 → ~29 tok/s). And comparing step times k=4 vs k=0 shows each drafter pass costs ~14ms — vs ~2ms of actual compute — so **even the drafter is overhead-bound.** Any per-step overhead reduction compounds across target + 4 draft passes.

**Ceiling math:** overhead fully eliminated → k=4 ≈ 50 tok/s theoretical. Realistic partial win → mid-30s.

## Experiment results (all vs baseline 26.5–31.1 c1 / 60.8 c6, k=4, 512-tok temp-0 prose)

| config | c1 | c6 agg | verdict |
|---|---|---|---|
| baseline (k=4, FLASHMLA 200K) | 26.5–31.1 | 60.8 | reference |
| k=0 (no spec) | 14.9 | 41.8 | decomposition probe — not a serving config |
| NCCL_PROTO=LL | 26.3–28.5 | 61.9 | **neutral** — NCCL already auto-selects LL for small messages |
| **fuse_gemm_comms** | 28.4–29.7 | **63.0** | **small real win on aggregate (+2 c6); c1 within noise. KEPT — it's free** |
| expert-parallel | — | — | **FLEET-KILLER: OOM'd all 4 nodes into swap-death; required physical power-cycle.** EP's MoE layout blows the <1GB-free memory budget. Do not retry without full memory retune (lower gmu, smaller KV). |
| k=5 | not run | — | skipped after the EP incident; theory says marginal (position-5 acceptance ~0.45 vs +1 draft pass/step). Optional future cycle. |

**Shipped config after the night: k=4 + fuse_gemm_comms** (`~/glm-5.2-gb10/speednight-fuse.sh`).

## The RDMA-allreduce verdict (the strategic question)

**BUILD-WORTHY — the data now justifies it.** Reasoning:
- The 42ms/step overhead is real and measured, not hypothesized.
- Config-level attacks are exhausted: NCCL flags neutral (auto-tuned already), fuse pass collects only ~1–2ms, EP structurally infeasible on this memory budget.
- The remaining overhead lives in per-call NCCL/RoCE latency across ~156 tiny allreduces per step. lukealonso's b12x proved the same attack works on PCIe (single-box); nobody has built the RoCE-fabric equivalent for Spark clusters.
- Prize: 5–10 tok/s single-stream (28.8 → mid-30s), compounding through the drafter. It would be the defining community contribution for every multi-Spark owner.
- Cost: weeks-class kernel/verbs engineering. Next step if pursued: profile with torch-profiler to get exact per-allreduce latency, then prototype a one-shot RC-verbs allreduce for the 24KB decode message size.

## Operational lessons
- **EP incident:** untested memory-layout flags on nodes running <1GB free can swap-wedge the entire fleet beyond SSH recovery. Rule: any experiment that changes weight/KV layout gets a reduced gmu first boot. (Cost: one fleet power-cycle, no data loss.)
- Staged monitors with 2.5-min pings (Tony's cadence preference) worked well — kept visibility through every cycle including the outage.

## Scoreboard vs community (for context)
Our 28.8–31 median remains the fastest known sustained single-stream for this stack on 4 nodes; Zatz 640K: 19.6–25.7 (peaks 33–37); Cosmic DCP2 328K: 20–36. Everyone's k=4 now. The differences between rigs are measurement-basis (prose vs synthetic) more than config.

---

# Speed-Night 2 (2026-07-17)

Baseline going in (same-day bench, warm): c1 31.1 code / 34.2 math; c6 aggregate ~60.5.

## Results by lever (each its own relaunch; 512-tok completions, temp 0, warm)

| Config | c1 code | c1 math | c1 prose | c1 mean | c6 agg (2 rounds) |
|---|---|---|---|---|---|
| KNOWNGOOD (k=4, 8192 chunks) | 31.1 | 34.2 | ~27.5 | ~31 | ~60.5 |
| P1: +capture-sizes[5..30] +atomic-add +4096 chunks | 30.8 | 32.9 | 27.5 | 30.4 | **66.2 / 75.0** |
| P2: k=5 +capture[6..36] +atomic-add, 8192 chunks | **32.0** | **36.0** | **29.6** | **32.5** | 64.9 / 71.2 |

## Verdicts

- **cudagraph_capture_sizes incl. the c6 batch size (30 or 36): KEEP — the single biggest c6 win (+10-24% aggregate).** Root cause: default capture sizes round to multiples of (k+1) and skipped c6's token count entirely, so c6 decode ran piecewise (no full graph). Check `max_cudagraph_capture_size` vs `max_num_seqs*(k+1)` on any config change.
- **MTP k=5: KEEP — new c1 records across all three workloads (mean 32.5, math 36.0).** Per-position acceptance at pos-5 measured 0.547 on prose-like loads — above the ~0.51 break-even. Mixed/agentic windows drop to ~0.36 accept at pos-4/5, so k=5 is workload-sensitive; prose/code/math all won.
- **--max-num-batched-tokens 4096: REVERT — cost ~1-2 tok/s c1**, and did NOT fix the short-prompt-behind-long-ingest wait (35.0s vs 33.8s baseline). The real fix (`--max-num-partial-prefills 2`) is **NotImplementedError on this vLLM pin** — re-test after any re-pin.
- **VLLM_MARLIN_USE_ATOMIC_ADD=1: kept** (log-recommended; not isolated, rode along with both winners).
- **GDR finding:** NCCL reports `GPU Direct RDMA Disabled` on all HCAs — `dlvsym(mlx5dv_reg_dmabuf_mr, MLX5_1.25)` fails against Ubuntu rdma-core 50.0 (host and container identical). All cross-node traffic host-stages. On unified-memory GB10 the copy is same-silicon so the penalty is muted; revisit only if chasing the last few tok/s (needs a rdma-core rebuild or NCCL that probes unversioned symbols).
- **Felt latency:** see README — `chat_template_kwargs: {"enable_thinking": false}` takes first visible token from 7-10s to 0.36s. Biggest perceived-speed lever on the whole stack.
- **NCCL_ALGO=Tree: DEAD — boot fails with `DistBackendError: NCCL error ... invalid usage` (NCCL 2.30.4) at graph capture.** Do not retry on this stack.
- **Dual-rail RoCE: untested tonight** (deliberately skipped after the Tree failure — second rail is verified UP on all 4 nodes; a future candidate, expected 0-2 c1).

## Final serving config (end of night)

`speednight-k5.sh` = KNOWNGOOD + `cudagraph_capture_sizes [6,12,18,24,30,36]` + MTP `k=5` +
`VLLM_MARLIN_USE_ATOMIC_ADD=1` (chunks 8192, NCCL WARN). **c1 mean 32.5 (records on all three
workloads), c6 aggregate 65-71 vs ~60.5 baseline.**

## Ranked: what to do next

1. **DFlash single-pass drafter lane** — the ceiling math says ~48ms of every ~110ms step is the
   k sequential drafter passes (piecewise-only by code). DFlash collapses them to one pass:
   **est. +10-20 c1, the 34→50 path.** Days of work; a `speednight-dflash.sh` lane already exists.
2. **FlashInfer 0.6.14 sparse-MLA port** — jasl's vLLM #41834 posted 41.9 tok/s decode / 1757 tok/s
   prefill on 2× GB10 (DSA-family model) with it. Could replace the slower parts of the b12x
   Triton overlay AND is the only known prefill-rate lever (~800 → potentially 1500+ tok/s).
3. **Remaining cherry-picks onto the pin**: vLLM #47448 (MTP post-final-norm fix — may lift
   acceptance further, compounding with k=5), #47410 (FP32 gate). #46862 already deployed.
4. **Dual-rail RoCE test** (cheap, one relaunch, do it next quiet window).
5. **RDMA one-shot allreduce** — still build-worthy (+5-10) but weeks; do after 1-2.
6. Re-pin to v0.25.1 only when #45317 (native sm_121 DSA) merges or the DFlash/dspark plumbing
   is needed; revalidation protocol is in "Upgrading beyond the pin".

**Post-script:** the end-of-night re-verify of the final k=5 boot read c1 ~22 — because six live
user streams were on the endpoint during the bench (num_requests_running=6; the box was serving
~52 tok/s aggregate to real traffic at that moment). Solo numbers for this exact config stand from
the identical quiet-box boot the same morning (c1 32.5 mean / c6 65-75). Benchmark discipline rule
worth keeping: `curl /metrics | grep num_requests_running` must read 0, or your c1 number is
actually a cN number.
