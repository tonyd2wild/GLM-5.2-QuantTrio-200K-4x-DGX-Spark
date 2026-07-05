# GLM-5.2 (unpruned QuantTrio Int4-Int8Mix) at 200K context on a 4× NVIDIA DGX Spark (GB10) cluster

A complete, follower-replicable recipe for serving **GLM-5.2** — the **unpruned** QuantTrio
`GLM-5.2-Int4-Int8Mix` checkpoint, all 256 experts intact — across **4× NVIDIA DGX Spark (GB10)**
nodes over a RoCE fabric, with **200,000-token context**, **MTP speculative decoding (k=4)**,
fp8 sparse-MLA KV cache, and full CUDA graphs. vLLM native multi-node (no Ray), one plain
`docker run` per node.

**Measured: 28.8 tok/s single-stream (median), 60.5 tok/s aggregate at 6 concurrent, 200K
context, unpruned.**

### Measured performance (final, 2026-07-05)

| Concurrency | Aggregate tok/s | Per-stream avg | Per-stream min | MTP accept len |
|---|---|---|---|---|
| 1 (warm, median of 3) | **28.8** | 28.8 | 27.3 | 3.3–3.6 |
| 2 | 37.6 | 20.2 | 18.8 | 3.50 |
| 3 | 39.3 | 13.6 | 13.1 | 3.22 |
| 4 | 53.5 | 14.1 | 13.4 | 3.28 |
| 5 | 59.1 | 12.5 | 11.8 | 3.22 |
| 6 | **60.5** | 10.6 | 10.1 | 3.23 |

*(512-token generations, temp 0, low-depth context. c1 is the warm median of 3 runs —
27.3/29.0/28.8; the cold first request after boot reads lower (16–22 tok/s). The c3+ rows
exist only because of `patches/fix-indexer-mtp-overhang.py` — unpatched vLLM crashes at 3
concurrent. Context-depth tables (16K/32K prefill/decode) are the one remaining PENDING
item — see [Benchmarks](#8-benchmarks).)*

> **⚠️ Benchmark caveat (temp 0):** GLM-5.2 is a reasoning model. Independent testing on
> Apple Silicon (MLX) found that `temperature=0` can trigger TP-collective deadlocks on this
> architecture — the engine hangs rather than errors. This was not observed in the vLLM/GB10
> runs above (which completed cleanly), but if you replicate at higher concurrency or deeper
> context and hit a silent hang, try `temperature=0.6+`. See Gotcha 8 below.

---

## 1. Credits & lineage

This recipe stands entirely on the shoulders of the people below. If you use it, their work is
what you are using.

| Who | What |
|---|---|
| **[CosmicRaisins](https://github.com/CosmicRaisins/glm-5.2-gb10)** | The whole sm_121 sparse-MLA port: the `glm-5.2-gb10` repo, the 10 Triton kernels, the DeepGEMM bypass, the launch harness this repo's `launch.sh` derives from. Apache-2.0. Nothing here works without this. |
| **Zatz** | The unpruned QuantTrio recipe — proving the full 256-expert Int4-Int8Mix checkpoint fits and flies on 4× GB10 (forum thread [374125](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125), posts #57 and #84). |
| **back199640** | Tuning that closed the gap (thread 374125, posts #80 and #89): `--async-scheduling`, MTP k=4 with `draft_tensor_parallel_size: 1`, the head-pad trick, and explicit `--kv-cache-memory-bytes` for deterministic boot. |
| **ciprianveg** | The baked-mod scripts in `mods/` (`glm52-sparse.zip`, thread 374125 post #34) that replicate CosmicRaisins' mods, and the NCCL channel-narrowing find (`NCCL_MIN/MAX_NCHANNELS=4`, post #107). |
| **[eugr / spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** | The image build harness (`build-and-copy.sh`) used to build the vLLM container for GB10. |
| **[QuantTrio](https://huggingface.co/QuantTrio)** | The `GLM-5.2-Int4-Int8Mix` checkpoint itself. |
| **p33zy** | Explored the alternative NVFP4 quantization path and GB10 hardware-acceleration trade-offs (thread 374125). |
| **aidendle94** | Shared container/image resources (originally for DeepSeek on GB10) that partially carried over to the GLM-5.2 bring-up (thread 374125). |
| **Claude Code** | Technical clarifications on the thread: sm_121 capability detection, cudagraph capture safety, b12x install requirements, and the sparse-MLA indexer path (thread 374125). |
| **[indexer-bf16](https://github.com/chadhurley25075-web/GLM-5.2-indexer-BF16-MLX)** | Independent finding: indexer precision governs long-context coherence in DSA models. Mixed-precision recipe (BF16 indexer/attn/embed/router + 4-bit experts) with proof across three quants. Operational findings: temp=0 TP deadlocks, thinking-mode token budgets, cudagraph KV starvation on unified memory. See Gotchas 8–10 and the Context depth warning. |

**Contributions from this deployment** (things we found during bring-up, offered back):

- The **indexer MTP-overhang fix** ([`patches/fix-indexer-mtp-overhang.py`](patches/fix-indexer-mtp-overhang.py)):
  the DSA indexer under-sizes its expanded block-table buffer by one block when
  `max_model_len` is an exact multiple of the block size and MTP is enabled — crashes the
  engine at ≥3 concurrent requests.
- The **load-phase page-cache-drop procedure** (Gotcha 6): periodic `drop_caches` on every
  node during weight load, which unsticks GB10 kernel-reclaim stalls.
- The **memory-budget numbers** for exactly 200K context on this checkpoint (Gotcha 7 and the
  config table): gmu 0.91 + `--kv-cache-memory-bytes 10950000000`.

Forum threads (read both — they are the primary sources):

- https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125
- https://forums.developer.nvidia.com/t/followup-mystery-solved-4x-spark-glm-5-2-nfp4-24tp-s-128k-ctx-no-reap/375416

## 2. Hardware requirements

- **4× NVIDIA DGX Spark (GB10)** — 121 GB unified memory each.
- **RoCE fabric between the four nodes.** We use a CRS812 switch, fabric subnet
  `192.168.192.0/24`, **MTU 9000 (jumbo frames) end-to-end** — on the switch AND every NIC.
  A direct-cabled mesh works too if NCCL can see the IB devices.
- **~420 GB free disk per node** — 405 GB of weights plus image, caches, and slack.

## 3. Why this checkpoint

**`QuantTrio/GLM-5.2-Int4-Int8Mix` — unpruned.**

- **Zero quality compromise.** No REAP/expert pruning: all **256 experts intact**. You serve the
  actual model, not a surgically reduced one.
- **Real memory headroom.** 405 GB of weights → **~95 GiB weights per node** on TP=4. On a
  121 GB unified-memory GB10 that leaves genuine room for KV cache, CUDA graphs, and the OS.
  Contrast the 429 GB NVFP4 hybrid: ~107 GB/node — a knife-edge that OOMs the moment page
  cache or warmup allocations breathe on it.
- **MTP drafter is in-checkpoint** (layer 78). No separate drafter model to download, align,
  or version-match — `--speculative-config` just points at the checkpoint itself.

## 4. Repo contents

| File | What it is |
|---|---|
| `launch.sh` | The 4-node launcher (adapted from CosmicRaisins, Apache-2.0). Edit the `EDIT`-marked config block, run from the head node. |
| `mods/glm52-sm12x-sparse/run.sh` | Baked-in-image mod: installs Triton sparse-MLA kernels + DeepGEMM bypass (verbatim from ciprianveg's zip). |
| `mods/glm52-b12x-sparse/run.sh` | Baked-in-image mod: installs b12x for CUDA-graph-safe sparse-MLA decode (verbatim from ciprianveg's zip). |
| `patches/fix-indexer-mtp-overhang.py` | Baked-in-image patch: fixes the DSA indexer's expanded block-table buffer being one block too small under MTP (required for `--max-num-seqs >= 3`). See step (h). |
| `LICENSE`, `NOTICE` | Apache-2.0 + attribution. |

**Not vendored: the 10 Triton kernels.** Get them from the upstream repo,
**[CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10)** (Apache-2.0),
directory `kernels/`. The 10 files you need:

```
b12x_sparse_helpers.py
deepseek_v2.py
flashmla_sparse.py
patch_flashmla_ops.py
sm12x_deep_gemm_fallbacks.py
sm12x_mqa.py
sm12x_sparse_mla_attn.py
sparse_attn_indexer.py
sparse_mla_env.py
sparse_mla_kernels.py
```

## 5. Step-by-step

Steps (a)–(b) and (h) run on one build machine (any of the Sparks works). Steps (c)–(f) touch
**every node**. Step (g) runs from the head node only. Do step (h) — the indexer patch — as
part of the same bake as (b), before you distribute the image in (c).

### a. Build the vLLM image (~35–60 min)

Clone [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and build against the
pinned vLLM commit:

```bash
git clone https://github.com/eugr/spark-vllm-docker
cd spark-vllm-docker
./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \
  -t vllm-node-tf5-glm52-b12x:probe --tf5
```

### b. Bake the mods into the image

Get the 10 kernel files from
[CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) `kernels/` into
`~/glm-triton` on the build machine, then run both mod scripts inside a container and commit the
result. **The kernels must be mounted at `/root/models/models15/glm-triton`** — that is the path
`mods/glm52-sm12x-sparse/run.sh` expects (`KERNELS=` at the top of the script). Exactly as we
ran it:

```bash
docker run -d --name glm52-modding \
  -v ~/glm-triton:/root/models/models15/glm-triton:ro \
  -v $(pwd)/mods/glm52-sm12x-sparse:/mods/glm52-sm12x-sparse:ro \
  -v $(pwd)/mods/glm52-b12x-sparse:/mods/glm52-b12x-sparse:ro \
  vllm-node-tf5-glm52-b12x:probe sleep infinity

docker exec glm52-modding bash /mods/glm52-sm12x-sparse/run.sh
docker exec glm52-modding bash /mods/glm52-b12x-sparse/run.sh

docker commit \
  --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
  --change 'CMD []' \
  glm52-modding vllm-node-tf5-glm52-b12x:probe-modded
docker rm -f glm52-modding
```

> **WARNING — two docker traps that both bit us:**
>
> 1. `docker commit` inherits `--entrypoint` overrides from the patch container. If the
>    container you're committing was started with an entrypoint override (or with a bare
>    command like `sleep infinity`), the committed image carries it forward and will not
>    boot vLLM. **Always** commit with
>    `--change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' --change 'CMD []'`
>    (as shown above).
> 2. When piping stdin scripts into containers, use `docker exec -i`. Without `-i` the
>    script **silently no-ops** — no error, nothing runs, and you commit an unpatched image.

Both scripts print `✓` lines; the sm12x one must end with `=== glm52-sm12x-sparse complete ===`
and the b12x one must show a successful `import b12x`.

### c. Distribute the image to all nodes

```bash
# from the build machine, for each OTHER node:
docker save vllm-node-tf5-glm52-b12x:probe-modded | \
  ssh <user>@<node> docker load
```

(Over the RoCE fabric this is minutes, not hours. `pigz` in the middle helps on slower links.)

### d. Weights: download once, rsync to all nodes

Download the checkpoint **once** (405 GB — do this on the node with the best internet):

```bash
hf download QuantTrio/GLM-5.2-Int4-Int8Mix \
  --local-dir /var/tmp/models/glm52-int4-int8mix
```

Then fan out **over the RoCE fabric** (not your uplink):

```bash
# from the node holding the weights, for each other node's fabric IP:
rsync -a --info=progress2 /var/tmp/models/glm52-int4-int8mix/ \
  <user>@192.168.192.X:/var/tmp/models/glm52-int4-int8mix/
```

Create the hub-layout symlink on **every** node (the serve path is
`/cache/huggingface/hub/glm52-int4-int8mix` inside the container):

```bash
mkdir -p /var/tmp/models/hub
ln -sfn ../glm52-int4-int8mix /var/tmp/models/hub/glm52-int4-int8mix
```

### e. Stage NCCL 2.30.4 on every node

The image's bundled NCCL is replaced at runtime via `LD_PRELOAD`. On **each** node:

```bash
pip download nvidia-nccl-cu13==2.30.4 -d /tmp/nccl --no-deps
mkdir -p /var/tmp/models/hub/nccl-2.30.4
cd /tmp/nccl && unzip -o nvidia_nccl_cu13-2.30.4*.whl 'nvidia/nccl/lib/libnccl.so.2'
cp nvidia/nccl/lib/libnccl.so.2 /var/tmp/models/hub/nccl-2.30.4/
```

### f. Kernels on every node

Copy the 10 `.py` files from CosmicRaisins/glm-5.2-gb10 `kernels/` to `~/glm-triton` on **every**
node (launch.sh bind-mounts them file-by-file over the vLLM tree, read-only):

```bash
git clone https://github.com/CosmicRaisins/glm-5.2-gb10
for node in 192.168.192.1 192.168.192.2 192.168.192.3 192.168.192.4; do
  rsync -a glm-5.2-gb10/kernels/ <user>@$node:~/glm-triton/
done
```

### g. Launch

Edit the `EDIT`-marked config block in `launch.sh` (node IPs, SSH user/key, HCA + interface
names), then from the head node:

```bash
./launch.sh --dry-run   # sanity-check the generated docker commands first
./launch.sh
```

Plain `docker run` per node; vLLM native multi-node (`--nnodes/--node-rank`), **no Ray**.
Workers start headless first, head last. Expect **~12 min weight load + ~10 min cudagraph
warmup** before `curl http://<head>:8210/v1/models` answers.

### h. Bake the indexer MTP-overhang patch (required for `--max-num-seqs >= 3`)

Do this during the same bake session as step (b), before committing and distributing the
image. `patches/fix-indexer-mtp-overhang.py` fixes a vLLM bug where the DSA indexer sizes
its expanded block-table buffer from `max_model_len` alone; MTP spec tokens can extend a
request one block past it, and at ≥3 concurrent requests the engine crashes with
`RuntimeError: The expanded size of the tensor (3125) must match the existing size (3126)`.
See the patch's docstring for the full story.

**VALIDATED (2026-07-05):** with the patch verified in-image, the full concurrency sweep
(c1–c6) completed with zero crashes — including the c3+ levels that reliably crashed an
unpatched engine. See [Benchmarks](#8-benchmarks).

Bake it exactly the same way as the mods — mount it into the patch container and run it
before the `docker commit`:

```bash
docker run -d --name glm52-modding \
  ... \
  -v $(pwd)/patches:/patches:ro \
  vllm-node-tf5-glm52-b12x:probe sleep infinity

# (after the two mod scripts from step b)
docker exec glm52-modding python3 /patches/fix-indexer-mtp-overhang.py

docker commit \
  --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
  --change 'CMD []' \
  glm52-modding vllm-node-tf5-glm52-b12x:probe-modded
docker rm -f glm52-modding
```

It prints `patched: .../indexer.py` on success and is idempotent (safe to re-run).

## 6. Key serve config, with rationale

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | One GB10 per TP rank; 405 GB / 4 ≈ 95 GiB weights per node. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}` | MTP drafter is in-checkpoint (layer 78). k=4 with draft TP=1 (back199640, #89): the tiny drafter doesn't benefit from TP, and draft TP=1 removes cross-node hops from every speculation step. |
| `--kv-cache-dtype` | `fp8_ds_mla` | fp8 sparse-MLA KV: halves KV footprint, enables 200K on 10.5 GB/node of cache. |
| `--compilation-config` | `{"cudagraph_mode":"FULL"}` | Full CUDA graphs for decode. Requires the b12x mod — without it, graph capture crashes (`torch.full` under capture). |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution (back199640, #80) — meaningful tok/s on GB10. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk size: big enough for ~700+ tok/s prefill, small enough not to blow memory at depth. |
| `--gpu-memory-utilization` + `--kv-cache-memory-bytes` | `0.91` + `10950000000` | **Deterministic boot + KV budget for exactly 200K.** gmu alone lets vLLM size KV off *currently free* memory, which on GB10 unified memory varies with page cache — same command OOMs or boots depending on cache state. And gmu 0.90 leaves only 9.78 GiB for KV where 200000 ctx needs 10.19 GiB (see Gotcha 7). gmu 0.91 with KV pinned to 10.95 GB boots a 200,064-token pool every time. |
| `--max-model-len` | `200000` | 200K context, fits in the pinned KV budget with fp8_ds_mla. |
| `--max-num-seqs` | `6` | Up to 6 concurrent streams. Requires the indexer MTP-overhang patch (step h) — unpatched, the engine crashes at ≥3 concurrent requests. Drop to 1 for a pure single-stream latency build. |
| `NCCL_MIN/MAX_NCHANNELS` | `4` | ciprianveg (#107): narrowing NCCL channels on GB10 RoCE cuts contention; more channels is slower here. |
| `--reasoning-parser` / `--tool-call-parser` | `glm45` / `glm47` | Correct parsers for GLM-5.2's reasoning traces and tool-call format. |
| `--distributed-executor-backend` | `mp` | Native multiprocessing + `--nnodes/--node-rank` rendezvous. No Ray. |

## 7. Gotchas (hard-won)

1. **RoCE fabric IP must live on the right interface — and persist in netplan.** If the fabric
   IP is added ad hoc, a link-local address (169.254.x.x) can squat the port after a
   reboot/link-flap, which **shifts the GID table** — your `NCCL_IB_GID_INDEX` now points at the
   wrong GID and NCCL either fails or silently degrades. Put the fabric IP in netplan and verify
   with `show_gids` after any reboot.
2. **IB device passthrough is required.** Without `--device /dev/infiniband` + `--cap-add
   IPC_LOCK` + `--ulimit memlock=-1:-1`, NCCL **silently** falls back to TCP over the socket
   interface. Everything works; decode is ~12 tok/s instead of 30+. If numbers look halved,
   check `NCCL_DEBUG=INFO` output for `NET/IB` vs `NET/Socket`.
3. **Page-cache pressure on GB10 unified memory.** Loading ~95 GiB of weights fills the page
   cache on a machine where CPU and GPU share the same 121 GB. Before launch, on each node:
   `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`. This plus explicit
   `--kv-cache-memory-bytes` is what makes boot deterministic.
4. **Jumbo frames on the switch, not just the NICs.** MTU 9000 must be set end-to-end; a switch
   port at 1500 silently fragments and craters bus bandwidth.
5. **Don't trust `--gpu-memory-utilization` alone.** See the config table: pin
   `--kv-cache-memory-bytes` explicitly, or identical launches will OOM ~sometimes~ depending on
   what the page cache looked like at profiling time.
6. **Page-cache thrash during weight load.** On GB10, big loads stall at 100% CPU in kernel
   reclaim even with 14–18 GB "free". Run an unconditional
   `sync; echo 3 > /proc/sys/vm/drop_caches` every 60s on **every** node during the load phase
   (and once right before launch). Symptom: shard progress freezes mid-load; a manual drop
   unsticks it within seconds.
7. **KV budget for exactly 200K.** `--gpu-memory-utilization 0.90` leaves only 9.78 GiB for
   KV — 200000 ctx needs 10.19 GiB. Use gmu `0.91` with `--kv-cache-memory-bytes 10950000000`;
   boot allocates a 200,064-token pool.

8. **temperature=0 can deadlock TP collectives on reasoning models.** GLM-5.2 reasons heavily
into `reasoning_content`. At `temperature=0`, the deterministic sampling path can trigger a
TP-collective deadlock — the engine hangs silently rather than erroring. This was observed
independently on MLX/tensor-parallel (not vLLM), but the architecture is the same. If you hit
silent hangs at higher concurrency or deeper context, use `temperature >= 0.6`. The benchmarks
above ran at temp 0 without issue, but this is a known landmine at scale.
9. **`enable_thinking:false` gives sub-second response vs 40s+ with think-trace.** For inference
and tool-use turns where you don't need the reasoning trace, disable thinking explicitly.
The model otherwise spends its entire token budget in `reasoning_content` and returns near-empty
`content` — which looks broken but isn't. On small `max_tokens` budgets (80–600), reasoning eats
the whole allocation.
10. **cudagraph + low utilization starves KV cache on unified memory.** On architectures where
CPU and GPU share memory (GB10, Apple Silicon), piecewise cudagraph can reserve enough to push
KV below the minimum threshold. If you see OOM during graph capture that doesn't match your
memory math, try eager mode + MTP speculative decoding instead — it can be both more
memory-efficient AND faster.

## 8. Benchmarks

**Final concurrency results (2026-07-05)** — measured on this cluster with the final serve
config (gmu 0.91, KV 10.95 GB, max-num-seqs 6, MTP k=4), on a boot with the indexer patch
verified in-image. All runs: 512-token generations, temperature 0, low-depth context. All 6
concurrency levels completed with **zero crashes**.

### Decode by concurrency (final)

| Concurrency | Aggregate tok/s | Per-stream avg | Per-stream min | MTP accept len |
|---|---|---|---|---|
| 1 (warm, median of 3) | 28.8 | 28.8 | 27.3 | 3.3–3.6 |
| 2 | 37.6 | 20.2 | 18.8 | 3.50 |
| 3 | 39.3 | 13.6 | 13.1 | 3.22 |
| 4 | 53.5 | 14.1 | 13.4 | 3.28 |
| 5 | 59.1 | 12.5 | 11.8 | 3.22 |
| 6 | 60.5 | 10.6 | 10.1 | 3.23 |

Notes:

- **c1 is the warm median of 3 runs (27.3 / 29.0 / 28.8).** The cold first request after
  boot reads lower (16–22 tok/s) — quote the warm median, with this caveat.
- **The c3–c6 rows exist only because of the indexer patch** (`patches/fix-indexer-mtp-overhang.py`,
  step h): without it, the engine crashes at 3 concurrent requests. This run validates the
  patch — 6/6 concurrency levels, zero crashes.

### Context depth (16K / 32K)

**PENDING** — the one remaining open item: prefill/decode tables at 16K and 32K context depth.

> **⚠️ Indexer precision warning:** When running these tests, watch output quality closely,
> not just tok/s. Independent mixed-precision quantization experiments (MLX-side) found that
> the DSA lightning indexer is the component most sensitive to quantization — it governs
> which KV positions each token attends to, and degrading it corrupts long-range attention
> routing. In testing across three checkpoint variants:
>
> | Quant variant | Indexer precision | Observed collapse point |
> |---|---|---|
> | 8-bit uniform | 8-bit | ~4,700 tokens → degraded output |
> | DQ4plus (8-bit indexer) | 8-bit | ~3,000–3,700 tokens → degraded output |
> | BF16 indexer + 4-bit experts | **BF16** | 5,846+ tokens clean — natural stop |
>
> The routed MoE experts compress aggressively with no quality loss; the indexer, MLA
> attention, embeddings, router gate, and lm_head do not. If your 16K/32K benchmarks show
> coherent degradation (repeated words, salad, loss of topic) before any crash, indexer
> precision is the likely variable. Recipe and proof:
> https://github.com/chadhurley25075-web/GLM-5.2-indexer-BF16-MLX

### Boot telemetry (verified)

- Weights: **98.07 GiB per node**.
- KV pool: **200,064 tokens**, `fp8_ds_mla`.
- Steady state: MemAvailable 0.6–0.9 GB + 3.6–4.5 GB swap parked (**by design** — matches
  the upstream author's memory profile for this config class).

## 9. License

**Apache-2.0** — see [LICENSE](LICENSE). Required and deliberate: `launch.sh` derives from
CosmicRaisins' Apache-2.0 `launch.sh` (copyright notice preserved in the file header), and the
`mods/` scripts replicate his Apache-2.0 mods. See [NOTICE](NOTICE) for attribution.
