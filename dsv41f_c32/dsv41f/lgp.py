"""lever LGP: the second forward-pass root `logits_processor` (cacheon FORWARD_PASS_ROOTS) as a node.

Stock (fork, TP2) for a decode / verify forward: LM head (torch.matmul bf16, [T, V/2] per rank) -> the fork's 32-CTA
multimem bf16 all-gather into its symmetric buffer -> logits_buffer.copy_() (bf16 -> fp32, the 24240-CTA direct_copy).
LGP: the LM head writes its shard (the same torch.matmul, out= a contiguous [T, V/2] view of a symmetric buffer
allocated once) and ONE Triton kernel (<= #SM CTAs) does a per-CTA cross-rank flag barrier (data ready), reads the
local shard and the peer's shard over NVLink, widens bf16 -> fp32 with the conversion torch's copy_ uses and writes
straight into the next-token logits buffer (or a fresh fp32 tensor where stock calls logits.float()), then a second
per-CTA barrier (both ranks done reading) so no later LM head can overwrite a shard the peer still reads.

Exactness: the LM-head GEMM is the same call on the same operands (only the output address differs: checked bitwise
per row count at prepare against stock's own _compute_lm_head + the stock all-gather + .float(), agreed over the
ranks); the gather is pure data movement; bf16 -> fp32 is an exact widening (kernel checked on all 65536 bf16 bit
patterns against torch's copy_ at prepare).  Every case outside the served contract (not decode / target-verify,
logprobs, DP attention, attn-TP / all-to-all LM head, softcap, logit scale, fp32 / quantized / LoRA LM head, rows
above the buffer, autotune dummy runs, traced runs) runs stock's LogitsProcessor.forward, decided from rank-replicated
facts only.  First real eager call per row count is also checked against stock (fail closed, rank-agreed).
"""
import sys

import torch
import triton
import triton.language as tl

LGP = True                 # lever switch
LGP_MAX_ROWS = 0           # 0 = the fork gatherer's bound (max_running_requests x draft tokens, floor 128)
LGP_KERNEL = "asm"         # "asm": per-thread 16-byte loads / 2 x 16-byte stores (no layout conversion) | "tl"
LGP_TILE = 16384           # "tl": bf16 columns a tile (one row, one source rank)
LGP_WARPS = 16
LGP_GRID = 0               # 0 = the SM count
LGP_EXIT = True            # second barrier: both ranks done reading before either kernel ends
_GMAX = 1024               # flag slots a phase (>= any grid)

_LP_MOD = "sglang.srt.layers.logits_processor"


# ---------------------------------------------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------------------------------------------

@triton.jit
def _flag_barrier(send0, send1, wait0, wait1):
    """Threads 0 / 1: signal rank t's slot (CAS 0 -> 1, release.sys), then consume rank t's signal on our own pad
    (CAS 1 -> 0, acquire.sys).  The other threads skip; the caller follows with a CTA barrier."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %lt, %lo;
            .reg .pred %lp0, %lp1, %lpz;
            .reg .u64 %lsa, %lwa;
            mov.u32 %lt, %tid.x;
            setp.eq.u32 %lp0, %lt, 0;
            setp.eq.u32 %lp1, %lt, 1;
            mov.b64 %lsa, $1;
            mov.b64 %lwa, $3;
            @%lp1 mov.b64 %lsa, $2;
            @%lp1 mov.b64 %lwa, $4;
            or.pred %lpz, %lp0, %lp1;
            @!%lpz bra lgp_end;
        lgp_send:
            atom.global.release.sys.cas.b32 %lo, [%lsa], 0, 1;
            setp.ne.u32 %lp0, %lo, 0;
            @%lp0 bra lgp_send;
        lgp_wait:
            atom.global.acquire.sys.cas.b32 %lo, [%lwa], 1, 0;
            setp.ne.u32 %lp0, %lo, 1;
            @%lp0 bra lgp_wait;
        lgp_end:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l,l",
        args=[send0, send1, wait0, wait1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _lgp_pull(local_ptr, out_ptr, peer_delta, flag0, flag1, n_tiles,
              W: tl.constexpr, VOCAB: tl.constexpr, TPS: tl.constexpr, TILE: tl.constexpr,
              RANK: tl.constexpr, GMAX: tl.constexpr, EXIT: tl.constexpr):
    """local_ptr: this rank's [T, W] bf16 shard (symmetric buffer); the peer's shard is at local_ptr + peer_delta
    (elements, mapped peer memory).  out_ptr: [T, VOCAB] fp32, row stride VOCAB.  flag0 / flag1: int64 addresses of
    rank 0's / rank 1's flag region (2 phases x GMAX CTAs x 2 ranks int32)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if RANK == 0:
        own = flag0
    else:
        own = flag1
    # entry: our slot pid*2+RANK on rank t's pad; wait on our pad's slot pid*2+t
    _flag_barrier(flag0 + (pid * 2 + RANK) * 4, flag1 + (pid * 2 + RANK) * 4,
                  own + (pid * 2 + 0) * 4, own + (pid * 2 + 1) * 4)
    tl.debug_barrier()
    cols = tl.arange(0, TILE)
    for t in range(pid, n_tiles, nprog):
        s = t % 2
        rest = t // 2
        j = rest % TPS
        r = rest // TPS
        c = j * TILE + cols
        m = c < W
        if s * W + (j + 1) * TILE > VOCAB:     # padded vocab: drop columns past VOCAB (stock slices them off)
            m = m & (s * W + c < VOCAB)
        delta = peer_delta * (s != RANK).to(tl.int64)
        x = tl.load(local_ptr + delta + r * W + c, mask=m, volatile=True)
        tl.store(out_ptr + r * VOCAB + s * W + c, x.to(tl.float32), mask=m)
    tl.debug_barrier()
    if EXIT:
        base = GMAX * 2
        _flag_barrier(flag0 + (base + pid * 2 + RANK) * 4, flag1 + (base + pid * 2 + RANK) * 4,
                      own + (base + pid * 2 + 0) * 4, own + (base + pid * 2 + 1) * 4)


@triton.jit
def _epoch_barrier(epoch_addr, send_addr, wait_addr):
    """Thread 0: bump this CTA's epoch (own memory), signal the peer without a round trip (red.release.sys.add on
    its receive counter), spin (ld.acquire.sys) until our receive counter from the peer reaches the epoch.  Counters
    only grow, both ranks launch the same grid every call, so the per-slot epochs stay in lockstep (graph-safe: no
    host epoch).  The caller follows with a CTA barrier."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .u32 %lt, %le, %lv;
            .reg .s32 %ld;
            .reg .pred %lp;
            mov.u32 %lt, %tid.x;
            setp.ne.u32 %lp, %lt, 0;
            @%lp bra lgp_e_end;
            ld.volatile.global.u32 %le, [$1];
            add.u32 %le, %le, 1;
            st.volatile.global.u32 [$1], %le;
            red.release.sys.global.add.u32 [$2], 1;
        lgp_e_spin:
            ld.acquire.sys.global.u32 %lv, [$3];
            sub.s32 %ld, %lv, %le;
            setp.lt.s32 %lp, %ld, 0;
            @%lp bra lgp_e_spin;
        lgp_e_end:
            mov.u32 $0, 0;
        }
        """,
        "=r,l,l,l",
        args=[epoch_addr, send_addr, wait_addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _tid():
    return tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r", [], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _ld16(addr, pred):
    """16 bytes (8 bf16) from addr when pred (volatile = relaxed.sys: coherent with the peer's writes after the
    acquire), else zeros."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %lq;
            setp.ne.s32 %lq, $5, 0;
            mov.b32 $0, 0;
            mov.b32 $1, 0;
            mov.b32 $2, 0;
            mov.b32 $3, 0;
            @%lq ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[addr, pred],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _st32(addr, a, b, c, d, pred):
    """8 bf16 (4 words, element 0 in the low half of a) widened by cvt.f32.bf16 (torch copy_'s conversion on sm_90+)
    and stored as 32 bytes at addr when pred."""
    tl.inline_asm_elementwise(
        """
        {
            .reg .pred %lq;
            .reg .b16 %lh<8>;
            .reg .f32 %lf<8>;
            setp.ne.s32 %lq, $6, 0;
            mov.b32 {%lh0, %lh1}, $2;
            mov.b32 {%lh2, %lh3}, $3;
            mov.b32 {%lh4, %lh5}, $4;
            mov.b32 {%lh6, %lh7}, $5;
            cvt.f32.bf16 %lf0, %lh0;
            cvt.f32.bf16 %lf1, %lh1;
            cvt.f32.bf16 %lf2, %lh2;
            cvt.f32.bf16 %lf3, %lh3;
            cvt.f32.bf16 %lf4, %lh4;
            cvt.f32.bf16 %lf5, %lh5;
            cvt.f32.bf16 %lf6, %lh6;
            cvt.f32.bf16 %lf7, %lh7;
            @%lq st.global.v4.f32 [$1], {%lf0, %lf1, %lf2, %lf3};
            @%lq st.global.v4.f32 [$1+16], {%lf4, %lf5, %lf6, %lf7};
            mov.b32 $0, 0;
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[addr, a, b, c, d, pred],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _chunk(ch, total, local_addr, peer_delta, out_addr,
           W: tl.constexpr, VOCAB: tl.constexpr, RANK: tl.constexpr):
    """16-byte chunk ch of the (row, source rank, chunk) space -> (source address, output address, valid)."""
    CPS: tl.constexpr = W // 8
    seg = ch // CPS
    k = ch - seg * CPS
    s = seg % 2
    r = (seg // 2).to(tl.int64)
    col = s * W + k * 8
    valid = (ch < total) & (col < VOCAB)
    src = local_addr + (peer_delta * (s != RANK).to(tl.int64) + r * W + k * 8) * 2
    dst = out_addr + (r * VOCAB + col) * 4
    return src, dst, valid.to(tl.int32)


@triton.jit
def _lgp_pull_asm(local_addr, out_addr, peer_delta, flag0, flag1, total,
                  W: tl.constexpr, VOCAB: tl.constexpr, RANK: tl.constexpr, GMAX: tl.constexpr,
                  EXIT: tl.constexpr, THREADS: tl.constexpr):
    """The "asm" kernel: every thread moves 4 chunks of 8 bf16 an iteration (4 x 16-byte loads in flight, then
    4 x 32-byte stores), grid-strided over total = rows x 2 x W / 8 chunks."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    if RANK == 0:
        own = flag0
        peer = flag1
    else:
        own = flag1
        peer = flag0
    # flag region (int32): receive counters [phase][GMAX] (written by the peer), epochs [phase][GMAX] (own)
    _epoch_barrier(own + (2 * GMAX + pid) * 4, peer + pid * 4, own + pid * 4)
    tl.debug_barrier()
    tid = _tid()
    block = THREADS * 4
    for b0 in range(pid * block, total, nprog * block):
        s0, d0, v0 = _chunk(b0 + tid, total, local_addr, peer_delta, out_addr, W, VOCAB, RANK)
        s1, d1, v1 = _chunk(b0 + THREADS + tid, total, local_addr, peer_delta, out_addr, W, VOCAB, RANK)
        s2, d2, v2 = _chunk(b0 + 2 * THREADS + tid, total, local_addr, peer_delta, out_addr, W, VOCAB, RANK)
        s3, d3, v3 = _chunk(b0 + 3 * THREADS + tid, total, local_addr, peer_delta, out_addr, W, VOCAB, RANK)
        x0 = _ld16(s0, v0)
        x1 = _ld16(s1, v1)
        x2 = _ld16(s2, v2)
        x3 = _ld16(s3, v3)
        _st32(d0, x0[0], x0[1], x0[2], x0[3], v0)
        _st32(d1, x1[0], x1[1], x1[2], x1[3], v1)
        _st32(d2, x2[0], x2[1], x2[2], x2[3], v2)
        _st32(d3, x3[0], x3[1], x3[2], x3[3], v3)
    tl.debug_barrier()
    if EXIT:
        _epoch_barrier(own + (3 * GMAX + pid) * 4, peer + (GMAX + pid) * 4, own + (GMAX + pid) * 4)


# ---------------------------------------------------------------------------------------------------------------
# symmetric state
# ---------------------------------------------------------------------------------------------------------------

class State:
    """One symmetric bf16 buffer per rank: [max_rows * W] shard rows + a zeroed int32 flag tail."""

    def __init__(self, group, rank, world, max_rows, width, vocab, device):
        import torch.distributed._symmetric_memory as symm_mem

        if world != 2:
            raise RuntimeError(f"lgp: TP2 only (world {world})")
        self.rank, self.world, self.max_rows, self.width, self.vocab = rank, world, max_rows, width, vocab
        data = max_rows * width
        data = (data + 255) // 256 * 256
        flags_i32 = 2 * _GMAX * 2
        with torch.inference_mode(False), torch.no_grad():
            buf = symm_mem.empty((data + flags_i32 * 2,), dtype=torch.bfloat16, device=device)
        hdl = symm_mem.rendezvous(buf, group=group)
        if hdl.rank != rank or hdl.world_size != world:
            raise RuntimeError(f"lgp: symm handle rank {hdl.rank}/{hdl.world_size} != {rank}/{world}")
        self.buf, self.hdl = buf, hdl
        ptrs = [int(p) for p in hdl.buffer_ptrs]
        if ptrs[rank] != buf.data_ptr():
            raise RuntimeError("lgp: local buffer pointer mismatch")
        peer = ptrs[1 - rank]
        if (peer - ptrs[rank]) % 32:
            raise RuntimeError("lgp: peer buffer not 32-byte aligned relative to ours")
        self.peer_delta = (peer - ptrs[rank]) // 2
        self.flag_addr = [p + data * 2 for p in ptrs]
        self.flags = buf[data:].view(torch.int32)
        with torch.inference_mode(False), torch.no_grad():
            self.flags.zero_()
        torch.cuda.synchronize(device)
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count

    def shard(self, rows):
        return self.buf[: rows * self.width].view(rows, self.width)

    def grid(self, rows):
        g = LGP_GRID if LGP_GRID > 0 else self.sms
        if LGP_KERNEL == "asm":
            total = rows * 2 * (self.width // 8)
            n = triton.cdiv(total, LGP_WARPS * 32 * 4)
            return min(g, n, _GMAX), total, 0
        tps = triton.cdiv(self.width, LGP_TILE)
        n = rows * 2 * tps
        return min(g, n, _GMAX), n, tps

    def pull(self, rows, out):
        """The kernel: this rank's shard rows (already written) + the peer's -> out [rows, vocab] fp32."""
        if out.dtype != torch.float32 or tuple(out.shape) != (rows, self.vocab) or not out.is_contiguous():
            raise RuntimeError("lgp: output must be a contiguous [rows, vocab] fp32 tensor")
        g, n, tps = self.grid(rows)
        if LGP_KERNEL == "asm":
            _lgp_pull_asm[(g,)](self.buf.data_ptr(), out.data_ptr(), self.peer_delta, self.flag_addr[0],
                                self.flag_addr[1], n, W=self.width, VOCAB=self.vocab, RANK=self.rank, GMAX=_GMAX,
                                EXIT=LGP_EXIT, THREADS=LGP_WARPS * 32, num_warps=LGP_WARPS)
        else:
            _lgp_pull[(g,)](self.buf, out, self.peer_delta, self.flag_addr[0], self.flag_addr[1], n,
                            W=self.width, VOCAB=self.vocab, TPS=tps, TILE=LGP_TILE, RANK=self.rank, GMAX=_GMAX,
                            EXIT=LGP_EXIT, num_warps=LGP_WARPS)
        return out


# ---------------------------------------------------------------------------------------------------------------
# the node
# ---------------------------------------------------------------------------------------------------------------

class Prepared:
    def __init__(self, module):
        self.module = module
        self.state = None
        self.status = "pending"      # pending | on | off:<reason>
        self.checked = set()         # row counts whose first real eager call matched stock
        self.calls = 0
        self.fast = 0


def _log(msg):
    print(f"CACHEON_DSV4F lever LGP: {msg}", file=sys.stderr, flush=True)


def _tp():
    from sglang.srt.distributed import get_tp_group

    return get_tp_group()


def _agree(ok: bool) -> bool:
    """MIN over the TP CPU group: every rank arms only if every rank passed."""
    group = _tp()
    t = torch.tensor([1 if ok else 0], dtype=torch.int64)
    if group.world_size > 1:
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group)
    return bool(t.item())


def _static_reason(module, lm_head):
    """Why the node can never take the fast path (module / LM-head config; identical on every rank), or None."""
    lp = sys.modules.get(_LP_MOD)
    if lp is None:
        return "no logits_processor module"
    if not LGP:
        return "switch off"
    if not module.do_tensor_parallel_all_gather or module.do_tensor_parallel_all_gather_dp_attn:
        return "no plain TP all-gather"
    if module.use_attn_tp_group or module.use_tp_lm_head_all_to_all:
        return "attn-TP / all-to-all LM head"
    if module.logit_scale is not None or module.final_logit_softcapping:
        return "logit scale / softcap"
    if module.use_fp32_lm_head or module.rl_on_policy_target is not None:
        return "fp32 LM head / rl on-policy"
    if hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"):
        return "LoRA LM head"
    qm = getattr(lm_head, "quant_method", None)
    if lp.should_apply_lm_head_quant_method(lm_head, qm):
        return "quantized LM head"
    w = getattr(lm_head, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype != torch.bfloat16 or w.dim() != 2 or not w.is_cuda:
        return "LM head weight not 2-D bf16 CUDA"
    try:
        from sglang.srt.utils.common import use_intel_amx_backend

        if use_intel_amx_backend(lm_head):
            return "AMX LM head"
    except Exception:  # noqa: BLE001
        pass
    group = _tp()
    if group.world_size != 2:
        return f"TP {group.world_size}"
    if w.shape[0] % 16 or w.shape[0] * 2 < module.vocab_size:
        return f"shard width {w.shape[0]}"
    return None


def _max_rows(module):
    if LGP_MAX_ROWS > 0:
        return int(LGP_MAX_ROWS)
    g = getattr(module, "_logits_gatherer", None)
    n = int(getattr(g, "_max_tokens", 0) or 0)
    return max(n, 128)


def gather_ref(shard, group, world):
    """[T, W] bf16 shards of every rank -> [T, world * W] (torch.distributed NCCL all-gather: data movement, the same
    bytes as the fork's multimem gather and its NCCL fallback)."""
    rows, width = shard.shape
    full = torch.empty((world, rows, width), dtype=shard.dtype, device=shard.device)
    torch.distributed.all_gather_into_tensor(full, shard.contiguous(), group=group)
    return full.permute(1, 0, 2).reshape(rows, world * width)


def _stock_logits(module, lm_head, x, group):
    """Stock's decode logits for hidden rows x: _compute_lm_head + all-gather + slice + .float()."""
    logits = module._compute_lm_head(x, lm_head, None)
    logits = gather_ref(logits, group, 2)
    if logits.shape[-1] > module.vocab_size:
        logits = logits[:, : module.vocab_size]
    return logits.float()


def _ours(st, lm_head, x, out):
    rows = x.shape[0]
    torch.matmul(x.to(lm_head.weight.dtype), lm_head.weight.T, out=st.shard(rows))
    return st.pull(rows, out)


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int32), b.view(torch.int32))


def _build(p, lm_head, hidden):
    """Allocate + self-test (eager, not capturing; collective: both ranks reach it on the same call)."""
    module = p.module
    reason = _static_reason(module, lm_head)
    if reason is not None:
        p.status = f"off:{reason}"
        _log(p.status)
        return
    group = _tp()
    dev = hidden.device
    width = int(lm_head.weight.shape[0])
    max_rows = _max_rows(module)
    ok, why = True, ""
    st = None
    try:
        st = State(group.device_group, group.rank_in_group, group.world_size, max_rows, width, module.vocab_size, dev)
    except Exception as e:  # noqa: BLE001
        ok, why = False, f"alloc: {type(e).__name__}: {e}"
    ok = _agree(ok)
    if not ok:
        p.status = f"off:{why or 'peer alloc failed'}"
        _log(p.status)
        return
    try:
        why = _self_test(st, module, lm_head, hidden, max_rows, group.device_group)
    except Exception as e:  # noqa: BLE001
        why = f"self-test raised {type(e).__name__}: {e}"
    torch.cuda.synchronize(dev)
    if not _agree(not why):
        p.status = f"off:self-test {why or 'failed on the peer'}"
        _log(p.status)
        return
    p.state = st
    p.status = "on"
    _log(f"on: rows 1..{max_rows}, shard width {width}, vocab {module.vocab_size}, grid {st.grid(max_rows)[0]} "
         f"(SMs {st.sms}), kernel {LGP_KERNEL}, warps {LGP_WARPS}, exit barrier {LGP_EXIT}; self-test passed (all row "
         f"counts vs stock LM head + all-gather + .float(); 65536 bf16 patterns vs copy_)")


def _self_test(st, module, lm_head, hidden, max_rows, group):
    """'' when every row count 1..max_rows gives stock's bytes (random hidden rows, the real LM head), and the widening
    matches torch's copy_ on every bf16 bit pattern placed in both shards."""
    dev = hidden.device
    g = torch.Generator(device=dev).manual_seed(20261016)
    h = hidden.shape[-1]
    xs = (torch.randn(max_rows, h, device=dev, generator=g) * 2.0).to(torch.bfloat16)
    for rows in range(1, max_rows + 1):
        x = xs[:rows]
        ref = _stock_logits(module, lm_head, x, group)
        out = torch.empty(rows, module.vocab_size, dtype=torch.float32, device=dev)
        _ours(st, lm_head, x, out)
        if not _bits_equal(out, ref):
            return f"rows {rows}: logits differ"
    # every bf16 bit pattern through the pull kernel (both shards), vs torch's bf16 -> fp32 copy_
    rows = min(max_rows, 2)
    pat = torch.arange(0, 1 << 16, dtype=torch.int32, device=dev).to(torch.int16)
    shard = st.shard(rows).view(torch.int16).view(-1)
    n = shard.numel()
    full = pat.repeat((n + pat.numel() - 1) // pat.numel())[:n]
    shard.copy_(full)
    out = torch.empty(rows, module.vocab_size, dtype=torch.float32, device=dev)
    st.pull(rows, out)
    ref_bf = gather_ref(st.shard(rows).clone(), group, 2)[:, : module.vocab_size]
    ref = torch.empty(rows, module.vocab_size, dtype=torch.float32, device=dev)
    ref.copy_(ref_bf)
    if not _bits_equal(out, ref):
        return "bf16 pattern widening differs from copy_"
    return ""


def _fast_ok(p, args, kwargs):
    """The rank-replicated per-call decision.  Returns the parsed call or None (stock)."""
    names = ("input_ids", "hidden_states", "lm_head", "logits_metadata", "aux_hidden_states",
             "hidden_states_before_norm")
    if len(args) > len(names) or any(k not in names for k in kwargs):
        return None
    call = dict(zip(names, args))
    for k, v in kwargs.items():
        if k in call:
            return None
        call[k] = v
    for k in names:
        call.setdefault(k, None)
    lm = call["logits_metadata"]
    hs = call["hidden_states"]
    if lm is None or not isinstance(hs, torch.Tensor) or hs.dim() != 2 or hs.dtype != torch.bfloat16:
        return None
    mode = getattr(lm, "forward_mode", None)
    if mode is None or not (mode.is_decode() or mode.is_target_verify()):
        return None
    lp = sys.modules.get(_LP_MOD)
    if lp is None or lp.get_in_autotune_dummy_run():
        return None
    try:
        envs = lp.envs
        if envs.SGLANG_TRACE_LOGITS_E2E.get() or envs.SGLANG_TRACE_LOGITS_E2E_SYNC.get():
            return None
    except Exception:  # noqa: BLE001
        return None
    if isinstance(lm, lp.ForwardBatch):
        if lm._original_forward_mode == lp.ForwardMode.IDLE:
            return None
        if lm.multi_item_delimiter_indices is not None or lm.token_indices_to_pool is not None:
            return None
    elif not isinstance(lm, lp.LogitsMetadata):
        return None
    rows = int(hs.shape[0])
    if rows < 1 or rows > p.state.max_rows:
        return None
    return call


def forward(prepared, *args, **kwargs):
    p = prepared
    module = p.module
    p.calls += 1
    capturing = torch.cuda.is_current_stream_capturing()
    if p.status == "pending" and not capturing:
        lm_head = args[2] if len(args) > 2 else kwargs.get("lm_head")
        hs = args[1] if len(args) > 1 else kwargs.get("hidden_states")
        if lm_head is not None and isinstance(hs, torch.Tensor) and hs.is_cuda:
            _build(p, lm_head, hs)
    if p.status != "on":
        return type(module).forward(module, *args, **kwargs)
    call = _fast_ok(p, args, kwargs)
    if call is None:
        return type(module).forward(module, *args, **kwargs)
    rows = int(call["hidden_states"].shape[0])
    if not capturing and rows not in p.checked:
        # first real eager call at this row count: stock answers first, ours must match it (rank-agreed)
        ref = type(module).forward(module, *args, **kwargs)
        ref_logits = ref.next_token_logits.clone()
        out = _fast(p, call)
        same = _bits_equal(out.next_token_logits, ref_logits) and _same_hidden(out.hidden_states, ref.hidden_states)
        if not _agree(same):
            p.status = f"off:first eager call at {rows} rows differs from stock"
            _log(p.status)
            out.next_token_logits.copy_(ref_logits)
            return ref
        p.checked.add(rows)
        return out
    p.fast += 1
    return _fast(p, call)


def _same_hidden(a, b):
    """The captured hidden states (stock's own helpers on the same inputs): the same tensor, else equal bytes."""
    if a is None or b is None:
        return a is None and b is None
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.data_ptr() == b.data_ptr() and a.stride() == b.stride():
        return True
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def _fast(p, call):
    """Stock's LogitsProcessor.forward for a decode / verify call without logprobs, with _get_logits' LM head +
    gather + copy replaced by the shard write + the pull kernel."""
    module = p.module
    lp = sys.modules[_LP_MOD]
    input_ids = call["input_ids"]
    hidden_states = call["hidden_states"]
    lm_head = call["lm_head"]
    logits_metadata = call["logits_metadata"]
    aux_hidden_states = call["aux_hidden_states"]
    hidden_states_before_norm = call["hidden_states_before_norm"]
    if isinstance(logits_metadata, lp.ForwardBatch):
        logits_metadata = lp.LogitsMetadata.from_forward_batch(logits_metadata)
    hidden_states, hidden_states_before_norm = lp.layernorm_sp.maybe_exit_gather(
        hidden_states=hidden_states,
        hidden_states_before_norm=hidden_states_before_norm,
        input_ids=input_ids,
        forward_mode=logits_metadata.forward_mode,
    )
    (pruned_states, pruned_states_before_norm, aux_pruned_states, sample_indices, input_logprob_indices,
     token_to_seq_idx) = module._get_pruned_states(
        hidden_states, hidden_states_before_norm, aux_hidden_states, logits_metadata)
    hidden_states_to_store = module._get_hidden_states_to_store(
        hidden_states, hidden_states_before_norm, aux_hidden_states, pruned_states, pruned_states_before_norm,
        aux_pruned_states, sample_indices, logits_metadata)
    del hidden_states
    rows = int(pruned_states.shape[0])
    st = p.state
    if (logits_metadata.extend_return_logprob or sample_indices is not None or pruned_states.dim() != 2
            or rows < 1 or rows > st.max_rows or pruned_states.dtype != torch.bfloat16):
        raise RuntimeError("lgp: decode / verify call outside the checked contract after pruning")
    vocab = module.vocab_size
    buf = logits_metadata.next_token_logits_buffer
    if (buf is not None and tuple(buf.shape) == (rows, vocab) and buf.dtype == torch.float32
            and buf.is_contiguous()):
        out = buf
    else:
        out = torch.empty((rows, vocab), dtype=torch.float32, device=pruned_states.device)
    with lp.get_forward().scoped(lora_batch_layout=lp.LoRABatchLayout.DP_LOCAL):
        torch.matmul(pruned_states.to(lm_head.weight.dtype), lm_head.weight.T, out=st.shard(rows))
    st.pull(rows, out)
    return lp.LogitsProcessorOutput(
        next_token_logits=out,
        hidden_states=hidden_states_to_store,
        mm_input_embeds=logits_metadata.mm_input_embeds,
    )


def prepare(module):
    p = Prepared(module)
    _log(f"node bound (vocab {getattr(module, 'vocab_size', None)}); symmetric buffer + self-test at the first eager "
         f"call")
    return p
