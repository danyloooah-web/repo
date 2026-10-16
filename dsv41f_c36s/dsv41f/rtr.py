"""Lever RTR on the fork (levers.RTR, CUDA unit rtr_unit): the MoE router gate GEMM of CUDA-graph decode / verify MoE
calls (the lifted forward_normal_dual_stream's self.gate(...) call, moe_fwd.py) as ONE kernel that reproduces stock's
cuBLAS result bit for bit.

Stock (fork 5e72ce5, pins below): MoEGate.forward with a bf16 weight [384, 5120] above tiny_router_gemm_max_tokens rows
(deterministic inference off, CUDA, not aiter / AMX) -> linear_bf16_fp32 (algo 'cublas') -> _linear_bf16_fp32_cublas ->
torch.mm(x, w.t(), out_dtype=torch.float32) -> cuBLASLt 13.1 on B300: an nvjet split-K GEMM over S = 2 or 3 fixed K
partitions + cublasLt::splitKreduce_kernel (two launches), or one no-split nvjet kernel (S = 1), S by row count (17-95
rows S = 2 except 32 / 64 (S = 3), 97-143 S = 1 except 112 / 128, 144 S = 2, ...). c23 profile: 8.1-8.5 us from router
start to logits ready at 72-144 rows.
Ours: rtr_unit.run: one launch, a cluster of S CTAs per (16-32 experts x 16-32 rows) tile, mma.sync chains in stock's
partition / K-step order, the partials summed in splitKreduce's order over DSMEM (rtr_unit.cu header: the bit model and
its evidence, tests/rtr/); PDL (levers.RTR_PDL): the weight's first stages are loaded before griddepcontrol.wait. The
tile config per row band (levers.RTR_TILES, else the unit's default) sets the grid: B300 runs one kernel's CTAs per SM,
so it also decides how many SMs the route window leaves to the side streams; the bits do not depend on it.

Gates (any miss: stock's own self.gate call, unchanged):
  prepare  levers.RTR; the stock sources hash to the fork read (MoEGate.forward, linear_bf16_fp32,
           _linear_bf16_fp32_cublas); algo 'cublas'; CUDA, not HIP / aiter; deterministic inference off; the unit
           imports and initializes. Per MoE replica (model_node's private copies): its gate is a stock MoEGate (class
           and bound forward), weight a bf16 [384, 5120] contiguous 16-byte aligned CUDA parameter, not AMX; the
           weight's tensor maps are encoded once (handle). Per row count of levers.RTR_ROWS above every armed gate's
           tiny_router_gemm_max_tokens: S is learned from stock's torch.mm on this rank (the A/6 probe: x[r, a] = 2^26
           and x[r, c] = 6 against a ones weight -- 6 is truncated away inside a partition's chain, rounded to +8 across
           partitions -- read for 15 column pairs and matched against the S = 1 / 2 / 3 partition models), then
           levers.RTR_SELFTEST byte checks of ours vs stock's torch.mm (random input families incl. +-0 / inf / NaN /
           overflow / subnormal rows, the replicas' real gate weights); a row count with no matching model or one
           differing byte stays stock.
  call     the row count is served on this rank and not turned off; x bf16 2-D [rows, 5120] contiguous, 16-byte
           aligned, on the weight's device; the replica's gate is the armed module, its weight the same tensor at the
           same address; no module / global hooks, no compiled call, grad off; linear_bf16_fp32 / its cuBLAS branch /
           the algo unchanged. The first eager use of every (row count, layer) runs ours AND stock's self.gate on the
           same input and compares every byte (a difference turns the row count off for the process and returns
           stock's result); inside a capture an unchecked (row count, layer) runs stock.
Exact: the same bits by construction and by the checks above. The output is a fresh torch.empty [rows, 384] fp32 as
stock's torch.mm allocates it (same stream, same caching-allocator path). Every decision depends on the row count and
static facts only, so both TP ranks capture the same thing; nothing here is collective (a rank-local fallback to stock
changes no bytes). Nothing is patched on a served module or class: the replica gets one plain attribute
(dsv4f_rtr, not callable).
"""
import time

import torch

from dsv41f import levers

E, H = 384, 5120
BF16, F32 = torch.bfloat16, torch.float32
# sha256[:16] of the exact source text (fl.source_hash) of the stock functions the kernel stands in for, fork 5e72ce5
PINNED_V2 = {"MoEGate.forward": "a25ca5bd8c775ed7"}
PINNED_BF = {"linear_bf16_fp32": "28df7f0b8a87272c", "_linear_bf16_fp32_cublas": "4216b5da7b920fe0"}
A2 = float(2 ** 26)
# (a, c) column pairs of the S probe: partition boundaries 1728 / 2560 / 3456 and far pairs
PROBE = ((0, 16), (0, 1000), (0, 1727), (0, 1728), (0, 2000), (0, 2559), (0, 2560), (0, 3000), (1728, 3455),
         (1728, 3456), (2560, 3456), (2560, 5119), (3456, 5119), (0, 4000), (0, 5119))
BOUNDS = {1: (), 2: (2560,), 3: (1728, 3456)}

_STATE = {"detail": "not installed", "on": False}
_H: dict = {}            # pinned stock objects + the unit
_S: dict = {}            # served row count -> S (learned + self-tested on this rank)
_CFG: dict = {}          # served row count -> the unit's tile config index (levers.RTR_TILES, else the default)
_CHK: dict = {}          # (row count, layer) -> first eager use == stock
_OFF: set = set()        # row counts turned off by a first-use difference
_N = {"fast": 0, "stock": 0, "checked": 0}


class _Static:
    __slots__ = ("gate", "w", "wptr", "dev", "handle", "tiny", "layer")


def _bits_equal(a, b) -> bool:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype == F32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


def _unit():
    if "unit" not in _H:
        try:
            import rtr_unit
        except ImportError as err:
            levers.announce("rtr-missing", f"bundle CUDA unit rtr_unit not importable ({err}): lever RTR off")
            return None
        _H["unit"] = rtr_unit
    return _H["unit"]


def tile_name(rows, S):
    """levers.RTR_TILES' tile config for (rows, S): the first (lo, hi, S, name) band holding it, '' = the unit's default."""
    for lo, hi, s, name in getattr(levers, "RTR_TILES", ()):
        if lo <= rows <= hi and s == S:
            return name
    return ""


def _part(S, k):
    return sum(k >= b for b in BOUNDS[S])


def learn_s(rows, dev):
    """S of stock's torch.mm at `rows` on this device (1, 2, 3), or None: no partition model fits the probe."""
    ones = torch.ones(E, H, dtype=BF16, device=dev)
    x = torch.zeros(rows, H, dtype=F32, device=dev)
    n = min(rows, len(PROBE))
    for r in range(n):
        a, c = PROBE[r]
        x[r, a] = A2
        x[r, c] = 6.0
    y = _H["cub"](x.to(BF16), ones)[:n, 0].tolist()
    seen = [v == A2 for v in y]
    if any(v not in (A2, A2 + 8.0) for v in y):
        return None
    fits = [S for S in (1, 2, 3) if all((_part(S, a) == _part(S, c)) == same for (a, c), same in zip(PROBE, seen))]
    return fits[0] if len(fits) == 1 else None


def make_x(rows, fam, g, dev):
    """Self-test / unit-test inputs [rows, 5120] bf16 (family index 0..4)."""
    fam %= 5
    x = torch.randn(rows, H, generator=g, device=dev)
    if fam == 1:      # per-row scale 1e-4 .. 1e4
        x = x * (10.0 ** (torch.rand(rows, 1, generator=g, device=dev) * 8 - 4))
    elif fam == 2:    # heavy-tailed
        x = x * torch.exp(torch.randn(rows, H, generator=g, device=dev) * 3)
    elif fam == 3:    # hidden-like: outlier channels, unit RMS
        cols = torch.randint(0, H, (16,), generator=g, device=dev)
        x[:, cols] *= 30
        x = x / x.pow(2).mean(1, keepdim=True).sqrt()
    elif fam == 4:    # special rows by r % 10: +0, -0, one nonzero, +inf, NaN, near-max, subnormal, ties, half zero
        r = torch.arange(rows, device=dev)
        kind = r % 10
        one = torch.randint(0, H, (rows,), generator=g, device=dev)
        x[kind == 0] = 0.0
        x[kind == 1] = -0.0
        m2 = kind == 2
        keep = torch.zeros(rows, H, dtype=torch.bool, device=dev)
        keep[r, one] = True
        x = torch.where(m2[:, None] & ~keep, torch.zeros_like(x), x)
        x = torch.where((kind == 3)[:, None] & keep, torch.full_like(x, float("inf")), x)
        x = torch.where((kind == 4)[:, None] & keep, torch.full_like(x, float("nan")), x)
        x = torch.where((kind == 5)[:, None], torch.sign(x) * 3.38e38, x)
        x = torch.where((kind == 6)[:, None], x * 2.0 ** -130, x)
        x = torch.where((kind == 7)[:, None], torch.round(x * 4) / 4, x)
        half = torch.arange(H, device=dev) < H // 2
        x = torch.where((kind == 8)[:, None] & half[None, :], torch.zeros_like(x), x)
    return x.to(BF16)


def _why_gate(gate) -> str:
    """Why the unit cannot serve this gate ('' = it can)."""
    H_ = _H
    if gate is None or type(gate) is not H_["gate_cls"]:
        return "gate is not a stock MoEGate"
    if getattr(gate.forward, "__func__", None) is not H_["gate_forward"]:
        return "gate forward is not MoEGate.forward"
    w = gate._parameters.get("weight")
    if not (isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == BF16 and tuple(w.shape) == (E, H)
            and w.is_contiguous() and w.data_ptr() % 16 == 0):
        return "weight not a bf16 [384, 5120] contiguous aligned CUDA tensor"
    if H_["dv2"].use_intel_amx_backend(gate):
        return "AMX gate"
    tiny = getattr(gate, "tiny_router_gemm_max_tokens", None)
    if not isinstance(tiny, int):
        return "no tiny_router_gemm_max_tokens"
    return ""


def prepare(layers) -> str:
    """model_node.prepare: pin stock, arm the MoE replicas, learn S and self-test every served row count on this rank."""
    if not getattr(levers, "RTR", False):
        _STATE["detail"] = "off (levers.RTR)"
        return _STATE["detail"]
    t_start = time.perf_counter()
    try:
        import sglang.srt.models.deepseek_v2 as dv2
        from sglang.kernels.ops.gemm import bf16_fp32 as bf
        from torch.nn.modules import module as nnm

        from dsv41f import fl

        bad = []
        for mod, pins in ((dv2, PINNED_V2), (bf, PINNED_BF)):
            for qual, want in pins.items():
                got = fl.source_hash(mod, qual)
                if got != want:
                    bad.append(f"{qual} {got}")
            fl._PARSED.pop(mod.__name__, None)
        if bad:
            raise RuntimeError("stock source differs from the fork read: " + ", ".join(bad))
        if not getattr(dv2, "_is_cuda", False) or getattr(dv2, "_use_aiter", True) or getattr(dv2, "_is_hip", True):
            raise RuntimeError("not the CUDA / non-aiter gate branch")
        if bf._linear_bf16_fp32_algo != "cublas":
            raise RuntimeError(f"SGLANG_OPT_BF16_FP32_GEMM_ALGO={bf._linear_bf16_fp32_algo!r}, not 'cublas'")
        if dv2.get_exec().deterministic.enable_deterministic_inference:
            raise RuntimeError("deterministic inference on (stock's own torch.mm branch)")
        for n in ("_global_forward_hooks", "_global_forward_pre_hooks"):
            if not hasattr(nnm, n):
                raise RuntimeError(f"torch.nn.modules.module.{n} missing")
        if not hasattr(torch.nn.Module, "_compiled_call_impl"):
            raise RuntimeError("torch.nn.Module call internals differ")
        _H.update(dv2=dv2, bf=bf, nnm=nnm, gate_cls=dv2.MoEGate, gate_forward=dv2.MoEGate.forward,
                  lin=bf.linear_bf16_fp32, cub=bf._linear_bf16_fp32_cublas)
        unit = _unit()
        if unit is None:
            raise RuntimeError("unit rtr_unit missing")
        unit.init()
        armed, why, gates, handles = [], {}, [], {}
        for r in layers:
            moe = r._modules.get("mlp")
            if moe is None or not getattr(moe, "dsv4f_dt_copy", False) or not hasattr(moe, "forward_normal_dual_stream"):
                continue
            gate = moe._modules.get("gate")
            reason = _why_gate(gate)
            if reason:
                why[reason] = why.get(reason, 0) + 1
                continue
            st = _Static()
            st.gate = gate
            st.w = gate._parameters["weight"]
            st.wptr = st.w.data_ptr()
            st.dev = st.w.device
            if st.wptr not in handles:
                handles[st.wptr] = unit.add_weight(st.wptr)
            st.handle = handles[st.wptr]
            st.tiny = int(gate.tiny_router_gemm_max_tokens)
            st.layer = getattr(moe, "layer_id", len(armed))
            moe.dsv4f_rtr = st
            armed.append(st)
        if not armed:
            raise RuntimeError(f"no MoE replica eligible ({why})")
        dev = armed[0].dev
        tiny = max(st.tiny for st in armed)
        t_arm = time.perf_counter()
        want = sorted({int(m) for m in levers.RTR_ROWS if tiny < int(m) <= unit.MAX_ROWS})
        g = torch.Generator(device=dev).manual_seed(int(getattr(levers, "RTR_SEED", 20261015)))
        learned, refused, cfgs = {}, {}, {}
        n_test = max(1, int(getattr(levers, "RTR_SELFTEST", 4)))
        stream = torch.cuda.current_stream(dev).cuda_stream
        for m in want:
            S = learn_s(m, dev)
            if S is None:
                refused[m] = "S probe fits no partition model"
                continue
            cfg = unit.config_index(m, S, tile_name(m, S))
            if cfg < 0:
                refused[m] = f"S {S}: no tile config {tile_name(m, S)!r}"
                continue
            bad_n = 0
            for t in range(n_test):
                st = armed[(t * 7 + m) % len(armed)]
                x = make_x(m, t, g, dev)
                ref = _H["cub"](x, st.w)
                out = torch.full((m, E), float("nan"), dtype=F32, device=dev)
                unit.run(st.handle, st.wptr, x.data_ptr(), out.data_ptr(), m, S, cfg, int(bool(levers.RTR_PDL)), stream)
                bad_n += not _bits_equal(out, ref)
            if bad_n:
                refused[m] = f"S {S}: {bad_n}/{n_test} self-test calls differ from stock"
                continue
            learned[m] = S
            cfgs[m] = cfg
        torch.cuda.synchronize(dev)
        _S.update(learned)
        _CFG.update(cfgs)
        if not _S:
            raise RuntimeError(f"no row count served (refused {refused})")
        _STATE["on"] = True
        by_s = {}
        for m, S in sorted(_S.items()):
            by_s.setdefault(S, []).append(f"{m}:{unit.config_name(_CFG[m])}/{unit.config_ctas(_CFG[m], m)}")
        _STATE["detail"] = (f"on: {len(armed)} MoE replicas ({len(handles)} gate weights), rows:tile/CTAs "
                            + "; ".join(f"S {S}: {' '.join(v)}" for S, v in sorted(by_s.items()))
                            + f", PDL {bool(levers.RTR_PDL)}, self-test {n_test} calls a row count == stock"
                            + f" (prepare {time.perf_counter() - t_start:.1f} s: pins + arming "
                              f"{t_arm - t_start:.1f} s)"
                            + (f"; refused {refused}" if refused else "") + (f"; not armed {why}" if why else ""))
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["on"] = False
        _S.clear()
        _CFG.clear()
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:300]})"
    return _STATE["detail"]


def _stock(moe, hidden_states, gemm_output_zero_allocator, why=None):
    _N["stock"] += 1
    if why is not None and levers.first(f"rtr-stock-{why}"):
        levers.announce(f"rtr-stock-{why}", f"lever RTR: stock gate at {int(hidden_states.shape[0])} rows ({why})")
    return moe.gate(hidden_states, gemm_output_zero_allocator)


def gate(moe, hidden_states, gemm_output_zero_allocator=None):
    """The lifted forward_normal_dual_stream's self.gate(...) call: the unit at served row counts, stock otherwise."""
    st = getattr(moe, "dsv4f_rtr", None)
    if not st or not _STATE["on"]:
        return moe.gate(hidden_states, gemm_output_zero_allocator)
    x = hidden_states
    rows = int(x.shape[0])
    S = _S.get(rows)
    if S is None or rows in _OFF or rows <= st.tiny:
        return _stock(moe, x, gemm_output_zero_allocator)
    g = st.gate
    H_ = _H
    nnm, bf = H_["nnm"], H_["bf"]
    if (moe._modules.get("gate") is not g or g._parameters.get("weight") is not st.w or st.w.data_ptr() != st.wptr
            or type(g) is not H_["gate_cls"] or getattr(g.forward, "__func__", None) is not H_["gate_forward"]
            or g._compiled_call_impl is not None or g._forward_hooks or g._forward_pre_hooks
            or nnm._global_forward_hooks or nnm._global_forward_pre_hooks or torch.is_grad_enabled()
            or bf.linear_bf16_fp32 is not H_["lin"] or bf._linear_bf16_fp32_cublas is not H_["cub"]
            or bf._linear_bf16_fp32_algo != "cublas"):
        return _stock(moe, x, gemm_output_zero_allocator, "module")
    if (x.dtype != BF16 or x.dim() != 2 or x.shape[1] != H or not x.is_contiguous() or x.data_ptr() % 16
            or x.device != st.dev):
        return _stock(moe, x, gemm_output_zero_allocator, "input")
    key = (rows, st.layer)
    ok = _CHK.get(key)
    if ok is None and torch.cuda.is_current_stream_capturing():
        return _stock(moe, x, gemm_output_zero_allocator, "unchecked-in-capture")
    out = torch.empty((rows, E), dtype=F32, device=x.device)
    H_["unit"].run(st.handle, st.wptr, x.data_ptr(), out.data_ptr(), rows, S, _CFG[rows], int(bool(levers.RTR_PDL)),
                   torch.cuda.current_stream(x.device).cuda_stream)
    if ok is None:
        # first eager use of this (row count, layer) -- the decode-graph warmups: stock's own call on the same input
        ref = moe.gate(hidden_states, gemm_output_zero_allocator)
        same = _bits_equal(out, ref)
        _CHK[key] = same
        _N["checked"] += 1
        if not same:
            _OFF.add(rows)
            levers.announce(f"rtr-off-{rows}", f"lever RTR at {rows} rows (layer {st.layer}, S {S}): != stock: OFF for "
                            f"this row count")
            return ref
        if levers.first(f"rtr-{rows}"):
            levers.announce(f"rtr-{rows}", f"lever RTR first at {rows} rows (layer {st.layer}, S {S}, "
                            f"{H_['unit'].config_name(_CFG[rows])}): == stock bit for bit: on")
    _N["fast"] += 1
    return out


def stats() -> dict:
    return {"detail": _STATE["detail"], "S": dict(_S), "cfg": dict(_CFG), "off": sorted(_OFF), "checked": dict(_CHK),
            **_N}
