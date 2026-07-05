# GLM-5.2 (unpruned QuantTrio Int4-Int8Mix) at 200K context on a 4× NVIDIA DGX Spark (GB10) cluster

A complete, follower-replicable recipe for serving **GLM-5.2** — the **unpruned** QuantTrio
`GLM-5.2-Int4-Int8Mix` checkpoint, all 256 experts intact — across **4× NVIDIA DGX Spark (GB10)**
nodes over a RoCE fabric, with **200,000-token context**, **MTP speculative decoding (k=4)**,
fp8 sparse-MLA KV cache, and full CUDA graphs. vLLM native multi-node (no Ray), one plain
`docker run` per node. Target: **~23–25+ tok/s decode, ~700+ tok/s prefill** single-stream.

### Expected performance

| Metric | Depth 0 | Depth 16K | Depth 32K |
|---|---|---|---|
| Decode (tok/s, single-stream) | PENDING | PENDING | PENDING |
| Prefill (tok/s) | PENDING | PENDING | PENDING |

*(Measured numbers to be filled in after benchmarking — see [Benchmarks](#8-benchmarks). Target
based on forum-reported results for this config class: ~23–25+ tok/s decode, ~700+ tok/s prefill.)*

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

Steps (a)–(b) run on one build machine (any of the Sparks works). Steps (c)–(f) touch **every
node**. Step (g) runs from the head node only.

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

docker commit glm52-modding vllm-node-tf5-glm52-b12x:probe-modded
docker rm -f glm52-modding
```

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

## 6. Key serve config, with rationale

| Setting | Value | Why |
|---|---|---|
| `--tensor-parallel-size` | `4` | One GB10 per TP rank; 405 GB / 4 ≈ 95 GiB weights per node. |
| `--speculative-config` | `{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}` | MTP drafter is in-checkpoint (layer 78). k=4 with draft TP=1 (back199640, #89): the tiny drafter doesn't benefit from TP, and draft TP=1 removes cross-node hops from every speculation step. |
| `--kv-cache-dtype` | `fp8_ds_mla` | fp8 sparse-MLA KV: halves KV footprint, enables 200K on 10.5 GB/node of cache. |
| `--compilation-config` | `{"cudagraph_mode":"FULL"}` | Full CUDA graphs for decode. Requires the b12x mod — without it, graph capture crashes (`torch.full` under capture). |
| `--async-scheduling` | on | Overlaps CPU scheduling with GPU execution (back199640, #80) — meaningful tok/s on GB10. |
| `--max-num-batched-tokens` | `8192` | Prefill chunk size: big enough for ~700+ tok/s prefill, small enough not to blow memory at depth. |
| `--gpu-memory-utilization` + `--kv-cache-memory-bytes` | `0.90` + `10500000000` | **Deterministic boot.** gmu alone lets vLLM size KV off *currently free* memory, which on GB10 unified memory varies with page cache — same command OOMs or boots depending on cache state. Pinning KV to 10.5 GB makes every boot identical. |
| `--max-model-len` | `200000` | 200K context, fits in the pinned KV budget with fp8_ds_mla. |
| `--max-num-seqs` | `1` | Single-stream latency build; raise for concurrency at some decode cost. |
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

## 8. Benchmarks

**PENDING** — to be filled after benching (llama-benchy-style, depths 0 / 16K / 32K).

### Single-stream

| Depth | Prefill tok/s | Decode tok/s | TTFT (s) |
|---|---|---|---|
| 0 | PENDING | PENDING | PENDING |
| 16K | PENDING | PENDING | PENDING |
| 32K | PENDING | PENDING | PENDING |

### Concurrent

| Concurrency | Depth | Aggregate prefill tok/s | Aggregate decode tok/s |
|---|---|---|---|
| 2 | 0 | PENDING | PENDING |
| 2 | 16K | PENDING | PENDING |
| 4 | 0 | PENDING | PENDING |
| 4 | 16K | PENDING | PENDING |

## 9. License

**Apache-2.0** — see [LICENSE](LICENSE). Required and deliberate: `launch.sh` derives from
CosmicRaisins' Apache-2.0 `launch.sh` (copyright notice preserved in the file header), and the
`mods/` scripts replicate his Apache-2.0 mods. See [NOTICE](NOTICE) for attribution.
