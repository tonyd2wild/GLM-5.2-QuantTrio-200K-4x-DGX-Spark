# GLM-5.2 (unpruned QuantTrio Int4-Int8Mix) on 4× DGX Spark — 28.8 tok/s single-stream, 200K context

> The unpruned QuantTrio `GLM-5.2-Int4-Int8Mix` checkpoint — all 256 experts intact — served
> across **4× NVIDIA DGX Spark (GB10)** over a RoCE fabric at **200,000-token context**, with
> MTP speculative decoding (k=4), fp8 sparse-MLA KV cache, and full CUDA graphs. vLLM native
> multi-node (no Ray), one plain `docker run` per node.

## TL;DR

- **What you get:** full-quality GLM-5.2 — the unpruned QuantTrio `GLM-5.2-Int4-Int8Mix`
  checkpoint with **all 256 experts intact** (no REAP/expert pruning), served across 4× GB10 at
  200K context with MTP k=4 speculative decode, `fp8_ds_mla` KV, full CUDA graphs, and vLLM
  native multi-node (no Ray).
- **The numbers:** **28.8 tok/s single-stream** (warm median), **60.5 tok/s aggregate at 6
  concurrent**, 200,000-token context, unpruned.
- **In-checkpoint drafter:** the MTP drafter lives in the checkpoint (layer 78) — no separate
  drafter model to download, align, or version-match; `--speculative-config` just points at the
  checkpoint itself.
- **Who it's for:** anyone with a 4× DGX Spark (GB10) RoCE cluster who wants to reproduce
  full-quality GLM-5.2 serving rather than a surgically reduced model.

## Hardware

- **4× NVIDIA DGX Spark (GB10)** — 121 GB unified memory each.
- **RoCE fabric between the four nodes.** We use a CRS812 switch, fabric subnet
  `192.168.192.0/24`, **MTU 9000 (jumbo frames) end-to-end** — on the switch AND every NIC.
  A direct-cabled mesh works too if NCCL can see the IB devices.
- **~420 GB free disk per node** — 405 GB of weights plus image, caches, and slack.

**Memory budget.** 405 GB of weights → **~95 GiB weights per node** on TP=4 (98.07 GiB measured).
On a 121 GB unified-memory GB10 that leaves genuine room for KV cache, CUDA graphs, and the OS.
Contrast the 429 GB NVFP4 hybrid: ~107 GB/node — a knife-edge that OOMs the moment page cache or
warmup allocations breathe on it.

## Quick start

Prerequisites (each covered in [Setup](#setup-detailed)): the modded vLLM image on every node, the
405 GB checkpoint staged at `/var/tmp/models` on every node, the 10 Triton kernels in
`~/glm-triton` on every node, and NCCL 2.30.4 staged. Then, from the **head node**:

```bash
# 1. Edit the EDIT-marked config block in launch.sh (node IPs, SSH user/key, HCA + interface names).
# 2. Sanity-check the generated docker commands first.
./launch.sh --dry-run
# 3. Launch (workers start headless first, head last).
./launch.sh
# 4. Smoke test — ready after ~12 min weight load + ~10 min cudagraph warmup.
curl -s http://localhost:8210/v1/models
```

Plain `docker run` per node; vLLM native multi-node (`--nnodes/--node-rank`), **no Ray**.
`./launch.sh --stop` does `docker rm -f` the container on every node.

## Setup (detailed)

Steps (a)–(b) and (h) run on one build machine (any of the Sparks works). Steps (c)–(f) touch
**every node**. Step (g) runs from the head node only. Do step (h) — the indexer patch — as part
of the same bake as (b), before you distribute the image in (c).

### Image & mods (steps a, b, h)

**a. Build the vLLM image (~35–60 min).** Clone
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) and build against the pinned
vLLM commit:

```bash
git clone https://github.com/eugr/spark-vllm-docker
cd spark-vllm-docker
# PIN THE HARNESS. Later harness commits carry inline vLLM patches that do not
# apply to our pinned ref (the build fails at an llm_base_proposer.py hunk),
# and harness HEAD moves fast.
git checkout 4ed3ebf

# DISABLE THE HARNESS'S PRESET-PR AUTO-MERGE. By default ("auto"), when no
# --apply-vllm-pr is given, the Dockerfile merges a preset list of vLLM PRs
# fetched LIVE from GitHub. Those branches have moved since this recipe was
# written: merging them today fast-forwards the pinned ab666069 to current
# vLLM main. The build still "succeeds" -- and the engine then dies at
# KV-cache init (view 576 vs 656 B/token) with the b12x backend missing.
# (build-and-copy.sh's own APPLY_PRESET_VLLM_PRS=false default is NOT
# forwarded to docker build, so this sed is required.)
sed -i 's|^ARG VLLM_APPLY_PRESET_PRS=""|ARG VLLM_APPLY_PRESET_PRS="false"|' Dockerfile

./build-and-copy.sh --vllm-ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 \
  -t vllm-node-tf5-glm52-b12x:probe --tf5

# VERIFY THE PIN HELD. The wheel version must be a dev build AT the pin:
#   grep -oE "vllm==[0-9a-z.+]*" build.log | tail -1   ->  ...dev190+gab6660699...
# If it reads dev893+g52b6667xx (or any other suffix), the tree was silently
# un-pinned and the resulting image will NOT boot this recipe.
```

**b. Bake the mods into the image.** Get the 10 kernel files from
[CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) `kernels/` into
`~/glm-triton` on the build machine, then run both mod scripts inside a container and commit the
result. The two mods do:

- `mods/glm52-sm12x-sparse/run.sh` — installs the Triton sparse-MLA kernels + DeepGEMM bypass
  (verbatim from ciprianveg's zip).
- `mods/glm52-b12x-sparse/run.sh` — installs b12x for CUDA-graph-safe sparse-MLA decode (verbatim
  from ciprianveg's zip).

**The kernels must be mounted at `/root/models/models15/glm-triton`** — that is the path
`mods/glm52-sm12x-sparse/run.sh` expects (`KERNELS=` at the top of the script). Exactly as we ran
it:

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

The 10 kernel files you need from CosmicRaisins/glm-5.2-gb10 `kernels/` (Apache-2.0; not vendored
here):

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

> **WARNING — two docker traps that both bit us:**
>
> 1. `docker commit` inherits `--entrypoint` overrides from the patch container. If the container
>    you're committing was started with an entrypoint override (or with a bare command like
>    `sleep infinity`), the committed image carries it forward and will not boot vLLM. **Always**
>    commit with `--change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' --change 'CMD []'`
>    (as shown above).
> 2. When piping stdin scripts into containers, use `docker exec -i`. Without `-i` the script
>    **silently no-ops** — no error, nothing runs, and you commit an unpatched image.

Both scripts print `✓` lines; the sm12x one must end with `=== glm52-sm12x-sparse complete ===`
and the b12x one must show a successful `import b12x`.

**h. Bake the indexer MTP-overhang patch (required for `--max-num-seqs >= 3`).** Do this during
the same bake session as step (b), before committing and distributing the image.
`patches/fix-indexer-mtp-overhang.py` fixes a vLLM bug where the DSA indexer sizes its expanded
block-table buffer from `max_model_len` alone; MTP spec tokens can extend a request one block past
it, and at ≥3 concurrent requests the engine crashes with `RuntimeError: The expanded size of the
tensor (3125) must match the existing size (3126)`. See the patch's docstring for the full story.

> **VALIDATED (2026-07-05):** with the patch verified in-image, the full concurrency sweep (c1–c6)
> completed with zero crashes — including the c3+ levels that reliably crashed an unpatched engine.
> See [Benchmarks](#benchmarks).

Bake it exactly the same way as the mods — mount it into the patch container and run it before the
`docker commit`:

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

### c. Distribute the image to all nodes

```bash
# from the build machine, for each OTHER node:
docker save vllm-node-tf5-glm52-b12x:probe-modded | \
  ssh <user>@<node> docker load
```

(Over the RoCE fabric this is minutes, not hours. `pigz` in the middle helps on slower links.)

### d. Weights — download once, rsync to all nodes

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

Plain `docker run` per node; vLLM native multi-node (`--nnodes/--node-rank`), **no Ray**. Workers
start headless first, head last.

### Verify

Expect **~12 min weight load + ~10 min cudagraph warmup** before the API answers:

```bash
curl -s http://localhost:8210/v1/models   # serves as 'glm-5.2'
docker logs -f vllm_slot                   # follow bring-up on the head node
```

## Benchmarks

**Final concurrency results (2026-07-05)** — measured on this cluster with the final serve config
(gmu 0.91, KV 10.95 GB, max-num-seqs 6, MTP k=4), on a boot with the indexer patch verified
in-image. All runs: 512-token generations, temperature 0, low-depth context. All 6 concurrency
levels completed with **zero crashes**.

| Concurrency | Aggregate tok/s | Per-stream avg | Per-stream min | MTP accept len |
|---|---|---|---|---|
| 1 (warm, median of 3) | **28.8** | 28.8 | 27.3 | 3.3–3.6 |
| 2 | 37.6 | 20.2 | 18.8 | 3.50 |
| 3 | 39.3 | 13.6 | 13.1 | 3.22 |
| 4 | 53.5 | 14.1 | 13.4 | 3.28 |
| 5 | 59.1 | 12.5 | 11.8 | 3.22 |
| 6 | **60.5** | 10.6 | 10.1 | 3.23 |

Notes:

- **c1 is the warm median of 3 runs (27.3 / 29.0 / 28.8).** The cold first request after boot reads
  lower (16–22 tok/s) — quote the warm median, with this caveat.
- **The c3–c6 rows exist only because of the indexer patch**
  (`patches/fix-indexer-mtp-overhang.py`, step h): without it, the engine crashes at 3 concurrent
  requests. This run validates the patch — 6/6 concurrency levels, zero crashes.

### Context depth (16K / 32K)

**PENDING** — the one remaining open item: prefill/decode tables at 16K and 32K context depth.

### Boot telemetry (verified)

- Weights: **98.07 GiB per node**.
- KV pool: **200,064 tokens**, `fp8_ds_mla`.
- Steady state: MemAvailable 0.6–0.9 GB + 3.6–4.5 GB swap parked (**by design** — matches the
  upstream author's memory profile for this config class).

## Configuration

Key serve settings, with rationale:

> **Why no `index_topk_pattern` hf-override here?** Other GLM-5.2-on-GB10 recipes
> (including the 655K DCP4 companion recipe) pass
> `--hf-overrides '{"index_topk_pattern":"..."}'` -- mandatory on their pins, or
> output silently corrupts past ~2k tokens. This recipe does NOT need it because
> the pinned ref `ab666069` IS the merge commit of vLLM #45895 ("Indexer init
> skip and MTP TopK share"), which obsoletes the manual override. Corollary: do
> not mix this launch.sh with a different vLLM pin without re-checking that.

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | One GB10 per TP rank; 405 GB / 4 ≈ 95 GiB weights per node. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}` | MTP drafter is in-checkpoint (layer 78). k=4 with draft TP=1 (back199640, #89): the tiny drafter doesn't benefit from TP, and draft TP=1 removes cross-node hops from every speculation step. |
| `--kv-cache-dtype` | `fp8_ds_mla` | fp8 sparse-MLA KV: halves KV footprint, enables 200K on 10.5 GB/node of cache. |
| `--compilation-config` | `{"cudagraph_mode":"FULL"}` | Full CUDA graphs for decode. Requires the b12x mod — without it, graph capture crashes (`torch.full` under capture). |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution (back199640, #80) — meaningful tok/s on GB10. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk size: big enough for ~700+ tok/s prefill, small enough not to blow memory at depth. |
| `--gpu-memory-utilization` + `--kv-cache-memory-bytes` | `0.91` + `10950000000` | **Deterministic boot + KV budget for exactly 200K.** gmu alone lets vLLM size KV off *currently free* memory, which on GB10 unified memory varies with page cache — same command OOMs or boots depending on cache state. And gmu 0.90 leaves only 9.78 GiB for KV where 200000 ctx needs 10.19 GiB (see Troubleshooting). gmu 0.91 with KV pinned to 10.95 GB boots a 200,064-token pool every time. |
| `--max-model-len` | `200000` | 200K context, fits in the pinned KV budget with fp8_ds_mla. |
| `--max-num-seqs` | `6` | Up to 6 concurrent streams. Requires the indexer MTP-overhang patch (step h) — unpatched, the engine crashes at ≥3 concurrent requests. Drop to 1 for a pure single-stream latency build. |
| `NCCL_MIN/MAX_NCHANNELS` | `4` | ciprianveg (#107): narrowing NCCL channels on GB10 RoCE cuts contention; more channels is slower here. |
| `--reasoning-parser` / `--tool-call-parser` | `glm45` / `glm47` | Correct parsers for GLM-5.2's reasoning traces and tool-call format. |
| `--distributed-executor-backend` | `mp` | Native multiprocessing + `--nnodes/--node-rank` rendezvous. No Ray. |

## Upgrading beyond the pin (read before bumping ANY ref)

The pin is load-bearing. The 10 Triton kernels are bind-mounted over exact
`dist-packages/vllm/...` paths, the mod scripts patch specific files, and the serve
flags were validated against the internals of `ab666069` -- e.g. this recipe needs no
`index_topk_pattern` override *because* the pin includes vLLM #45895 (see the note in
Configuration). On a different ref those assumptions shift: on 2026-07 main the b12x
backend does not register at all and the fp8_ds_mla KV layout changed (656 vs
576 B/token view) -- the engine dies at KV-cache init.

Treat any ref bump as a **revalidation event**, not a rebuild:

1. Bump the harness and the vLLM ref **together** (newer harness exists to build
   newer vLLM; mixing directions is where the silent breakage lives). Keep
   `VLLM_APPLY_PRESET_PRS="false"` -- pinned builds should never merge live PRs.
2. **Gate on the wheel suffix**: the built wheel version must be a dev build at the
   ref you asked for. If the suffix is any other hash, stop -- the tree moved.
3. Smoke-test before trusting: boots on all nodes; boot log selects the expected
   sparse attention backend; `GPU KV cache size` matches expectation; temp-0
   correctness on a known prompt; c1 within ~10% of the numbers in this README.
4. Expect the kernels/mods to need re-porting on larger jumps -- diff the overlay
   target paths inside the new image first.

Long-term exit: native GLM-DSA sparse support on sm_121 in upstream vLLM
(vllm-project/vllm#45317 is the tracker). When that lands, the kernel overlay -- and
with it most of this pinning -- becomes unnecessary.

## Troubleshooting

1. **RoCE fabric IP must live on the right interface — and persist in netplan.** If the fabric IP
   is added ad hoc, a link-local address (169.254.x.x) can squat the port after a reboot/link-flap,
   which **shifts the GID table** — your `NCCL_IB_GID_INDEX` now points at the wrong GID and NCCL
   either fails or silently degrades. Put the fabric IP in netplan and verify with `show_gids`
   after any reboot.
2. **IB device passthrough is required.** Without `--device /dev/infiniband` + `--cap-add IPC_LOCK`
   + `--ulimit memlock=-1:-1`, NCCL **silently** falls back to TCP over the socket interface.
   Everything works; decode is ~12 tok/s instead of 30+. If numbers look halved, check
   `NCCL_DEBUG=INFO` output for `NET/IB` vs `NET/Socket`.
3. **Page-cache pressure on GB10 unified memory.** Loading ~95 GiB of weights fills the page cache
   on a machine where CPU and GPU share the same 121 GB. Before launch, on each node:
   `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`. This plus explicit
   `--kv-cache-memory-bytes` is what makes boot deterministic.
4. **Jumbo frames on the switch, not just the NICs.** MTU 9000 must be set end-to-end; a switch
   port at 1500 silently fragments and craters bus bandwidth.
5. **Don't trust `--gpu-memory-utilization` alone.** See the [Configuration](#configuration) table:
   pin `--kv-cache-memory-bytes` explicitly, or identical launches will OOM ~sometimes~ depending
   on what the page cache looked like at profiling time.
6. **Page-cache thrash during weight load.** On GB10, big loads stall at 100% CPU in kernel reclaim
   even with 14–18 GB "free". Run an unconditional `sync; echo 3 > /proc/sys/vm/drop_caches` every
   60s on **every** node during the load phase (and once right before launch). Symptom: shard
   progress freezes mid-load; a manual drop unsticks it within seconds.
7. **KV budget for exactly 200K.** `--gpu-memory-utilization 0.90` leaves only 9.78 GiB for KV —
   200000 ctx needs 10.19 GiB. Use gmu `0.91` with `--kv-cache-memory-bytes 10950000000`; boot
   allocates a 200,064-token pool.

## Credits & links

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

**Contributed back from this deployment** (things we found during bring-up, offered back):

- The **indexer MTP-overhang fix**
  ([`patches/fix-indexer-mtp-overhang.py`](patches/fix-indexer-mtp-overhang.py)): the DSA indexer
  under-sizes its expanded block-table buffer by one block when `max_model_len` is an exact
  multiple of the block size and MTP is enabled — crashes the engine at ≥3 concurrent requests.
- The **load-phase page-cache-drop procedure** (Troubleshooting #6): periodic `drop_caches` on
  every node during weight load, which unsticks GB10 kernel-reclaim stalls.
- The **memory-budget numbers** for exactly 200K context on this checkpoint (Troubleshooting #7 and
  the [Configuration](#configuration) table): gmu 0.91 + `--kv-cache-memory-bytes 10950000000`.

Forum threads (read both — they are the primary sources):

- https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125
- https://forums.developer.nvidia.com/t/followup-mystery-solved-4x-spark-glm-5-2-nfp4-24tp-s-128k-ctx-no-reap/375416

### License

**Apache-2.0** — see [LICENSE](LICENSE). Required and deliberate: `launch.sh` derives from
CosmicRaisins' Apache-2.0 `launch.sh` (copyright notice preserved in the file header), and the
`mods/` scripts replicate his Apache-2.0 mods. See [NOTICE](NOTICE) for attribution.
