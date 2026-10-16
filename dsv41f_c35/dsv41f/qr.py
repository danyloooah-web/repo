"""Lever QR (CUDA unit qr_unit): MQALayer._compute_q_b on OUR attention replicas -- the DeepSeek-V4.1 query path's
FP8 rope (q_head_norm = False, q_out float8_e4m3fn for the trtllm sparse-MLA backend) -- on a kernel whose q_out bytes
equal stock's fused_q_norm_rope<bf16, 512, 64, PosT, kUsePDL, kApplyNorm = false, kOutFp8 = true>.

Stock (fork 5e72ce5): _compute_q_b runs wq_b, views the rows as [rows, n_local_heads, 512] and, with no q head norm and
an FP8 q_out, calls elementwise.fused_q_norm_rope(q, q_out, None, freqs_cis, positions): one JIT kernel, warp per
(token, head), whose FP8 output goes through torch_fp8_e4m3fn -- a ~16-instruction integer emulation of torch's cast
per element -- so it is instruction-bound (12096-row prefill pass: ~170 us a call, 594 MB moved = 3.5 TB/s).

The unit (qr_unit.cu) is stock's kernel with the same work split, loads, stores, PDL placement and float arithmetic
(the rope contraction and rounding read from stock's sm_103a SASS), but converts with the hardware
cvt.rn.satfinite.e4m3x2.f32 plus a per-byte NaN fix-up (stock's NaN is 0x7F without sign). tests/qr_test.py proves the
conversion equal to stock's over all 65536 bf16 patterns (through stock's own kernel) and the kernel byte-identical to
stock's over row counts 1..16384, adversarial values, int32 / int64 positions, strided rows and the PDL launch.

Binding: install() sets `_compute_q_b` as an instance attribute of OUR private attention replicas (model_node's
_replica copies, whose lifted forward calls stock's _forward_prepare* -> self._compute_q_b), never on MQALayer or a
served module: cacheon's node seal refuses an audited call once a callable of a served module or of its class was
rebound after binding. The draft model (deepseek_v4_dspark) is never touched. The wrapper runs stock's method for every
call outside the one branch it mirrors (a q head norm, no / non-FP8 q_out, torch.compile, any other dtype / shape /
stride / alignment); in that branch it calls self.wq_b and the view exactly as stock does and then the unit, or stock's
own fused_q_norm_rope call.

Exactness in the engine: install() pins the source (sha256 of the AST segment) of stock's _compute_q_b,
fused_q_norm_rope and its JIT module builder, and runs a self-test (stock vs the unit on synthetic rows with the served
rope table, int32 and int64 positions, every output byte) that verifies the coarse check keys (positions dtype, heads,
contiguous rows) before any graph capture. The first eager call of every fine check key (coarse key + row-count class)
runs stock into q_out and the unit into a scratch buffer and compares every byte; one difference turns the lever off for
the process (stock from then on). Under CUDA-graph capture (decode / verify graphs, breakable prefill graphs) the unit
runs only for coarse keys already verified eagerly, else stock.
"""
import ast
import hashlib
import inspect
import sys
import types

import torch

from dsv41f import levers

D = 512
ROPE = 64
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
U8 = torch.uint8
_STATE = {"detail": "not installed", "ok": False, "fast": 0, "checks": 0, "stock": None, "pdl": 0}
_OK: dict = {}            # check key -> True (bytes == stock) / False
_FN: dict = {}            # run, fqnr, raw

# sha256[:16] of the source segment of every function the unit mirrors or this module relies on, read from the fork's
# SGLang (5e72ce5); any difference: lever QR off
PINNED = {
    ("sglang.srt.models.deepseek_v4", "MQALayer._compute_q_b"): "6c42cb734c0dfdf2",
    ("sglang.kernels.ops.attention.dsv4.elementwise", "fused_q_norm_rope"): "cf2b81bb2f0f0aad",
    ("sglang.kernels.ops.attention.dsv4.elementwise", "_jit_main_q_norm_rope_module"): "293e51ca88efb1ac",
    ("sglang.kernels.jit.utils.arch", "is_arch_support_pdl"): "d6dbc4ab73e8b3cb",
}


def source_hash(module, qualname: str, cache=None):
    """sha256[:16] of the exact source text of `qualname` ("f" or "Class.f") in `module` (None: not found)."""
    hit = cache.get(module.__name__) if cache is not None else None
    if hit is None:
        src = inspect.getsource(module)
        hit = (src, ast.parse(src))
        if cache is not None:
            cache[module.__name__] = hit
    src, tree = hit
    node = tree
    for part in qualname.split("."):
        found = None
        for child in ast.walk(node):
            if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part
                    and child is not node):
                found = child
                break
        if found is None:
            return None
        node = found
    seg = ast.get_source_segment(src, node)
    return hashlib.sha256(seg.encode()).hexdigest()[:16] if seg is not None else None


def _modules():
    """The modules PINNED names, imported (raises on a missing one)."""
    import sglang.kernels.jit.utils.arch  # noqa: F401
    import sglang.kernels.ops.attention.dsv4.elementwise  # noqa: F401
    import sglang.srt.models.deepseek_v4  # noqa: F401
    return {name: sys.modules[name] for name, _ in PINNED}


def unit():
    if "run" not in _FN:
        try:
            import qr_unit
        except ImportError as err:
            levers.announce("qr-missing", f"bundle CUDA unit qr_unit not importable ({err}): lever QR off")
            return None
        _FN["run"] = qr_unit.run
    return _FN["run"]


def install(model, layers=None) -> str:
    """Pin the stock internals, self-test the unit against stock's kernel and bind the wrapped _compute_q_b on the
    private attention replicas of `layers` (model_node.prepare's replicas, aligned with model.layers)."""
    if _STATE["stock"] is not None or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "QR", False):
        _STATE["detail"] = "off (levers.QR)"
        return _STATE["detail"]
    try:
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        dsv4 = mods["sglang.srt.models.deepseek_v4"]
        el = mods["sglang.kernels.ops.attention.dsv4.elementwise"]
        arch = mods["sglang.kernels.jit.utils.arch"]
        if dsv4._is_hip or not dsv4._is_cuda or el._is_hip or el._is_xpu:
            raise RuntimeError("not CUDA")
        if getattr(dsv4, "fused_q_norm_rope", None) is not el.fused_q_norm_rope:
            raise RuntimeError("deepseek_v4.fused_q_norm_rope is not elementwise.fused_q_norm_rope")
        layer_cls = dsv4.MQALayer
        stock = layer_cls._compute_q_b
        targets = _replica_attns(model, layers, layer_cls, stock)
        if not targets:
            raise RuntimeError("no private MQALayer replica without a q head norm to bind on (lever QR never wraps a "
                               "served module)")
        if unit() is None:
            raise RuntimeError("unit qr_unit missing")
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(fqnr=el.fused_q_norm_rope,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
        _STATE["pdl"] = 1 if arch.is_arch_support_pdl() else 0
        why = _self_test(targets)
        if why:
            raise RuntimeError(why)
    except Exception as e:  # noqa: BLE001 - any missing / changed internal or a failed self-test keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["stock"] = stock
    _STATE["ok"] = True
    for attn in targets:
        attn._compute_q_b = types.MethodType(_wrapped, attn)   # the replica's own attribute; MQALayer untouched
    keys = sorted(k for k, v in _OK.items() if v)
    _STATE["detail"] = (f"on: {len(targets)} attention replicas, FP8 q rope at {levers.QR_MIN_ROWS}..{levers.QR_MAX_ROWS} "
                        f"rows, PDL {'on' if _STATE['pdl'] else 'off'}; self-test == stock for keys {keys}; first eager "
                        f"call per key checked vs stock, captures on verified keys only")
    return _STATE["detail"]


def _replica_attns(model, layers, layer_cls, stock) -> list:
    """The private MQALayer replicas of `layers` (never the served attention a replica layer may still share) without a
    q head norm: each must resolve _compute_q_b to stock's MQALayer method, so the bound wrapper shadows exactly it."""
    out = []
    for r, s in zip(layers or (), getattr(model, "layers", ())):
        attn = r._modules.get("self_attn") if hasattr(r, "_modules") else None
        served = s._modules.get("self_attn") if hasattr(s, "_modules") else None
        if (attn is None or attn is served or not isinstance(attn, layer_cls) or getattr(attn, "q_head_norm", True)
                or getattr(type(attn), "_compute_q_b", None) is not stock):
            continue
        bound = getattr(attn, "_compute_q_b", None)
        if getattr(bound, "__func__", None) is not stock or getattr(bound, "__self__", None) is not attn:
            continue
        out.append(attn)
    return out


def _wrapped(self, q, positions, q_out=None):
    """A replica's _compute_q_b (bound by install): stock's method except in the branch the unit mirrors
    (no q head norm, FP8 q_out): there stock's own wq_b + view, then the unit or stock's fused_q_norm_rope call."""
    if (not _STATE["ok"] or q_out is None or self.q_head_norm or q_out.dtype is not FP8
            or torch.compiler.is_compiling()):
        return _STATE["stock"](self, q, positions, q_out)
    q, _ = self.wq_b(q)
    q = q.view(-1, self.n_local_heads, self.head_dim)
    if not _fast(self, q, positions, q_out):
        _FN["fqnr"](q, q_out, None, self.freqs_cis, positions)   # stock's call of this branch, verbatim
    return q_out


def _args(q, positions, q_out, freqs):
    """The unit's launch arguments for stock's fused_q_norm_rope(q, q_out, None, freqs, positions), or None when the
    call is outside what the unit mirrors (stock runs it)."""
    if (q.dtype is not BF16 or q.dim() != 3 or q.shape[2] != D or q.stride(2) != 1 or q.stride(1) != D
            or q_out.dim() != 3 or q_out.shape != q.shape or q_out.stride(2) != 1 or q_out.stride(1) != D
            or not q.is_cuda or q_out.device != q.device):
        return None
    if (not isinstance(freqs, torch.Tensor) or freqs.dtype is not torch.complex64 or freqs.dim() != 2
            or freqs.shape[1] != ROPE // 2 or not freqs.is_contiguous() or freqs.device != q.device):
        return None
    if (not isinstance(positions, torch.Tensor) or positions.dtype not in (torch.int64, torch.int32)
            or positions.dim() != 1 or positions.shape[0] != q.shape[0] or positions.stride(0) != 1
            or positions.device != q.device):
        return None
    rows, heads = q.shape[0], q.shape[1]
    if rows < max(1, int(levers.QR_MIN_ROWS)) or rows > int(levers.QR_MAX_ROWS) or heads < 1:
        return None
    si, so = q.stride(0), q_out.stride(0)
    qp, op, fp, pp = q.data_ptr(), q_out.data_ptr(), freqs.data_ptr(), positions.data_ptr()
    pb = positions.element_size()
    if qp & 15 or si & 7 or op & 7 or so & 7 or fp & 7 or pp % pb or si < heads * D or so < heads * D:
        return None
    return (qp, si, op, so, fp, pp, pb, rows, heads)


def _key(a):
    """(coarse key, fine key): positions dtype, heads, contiguous input / output rows; + the row-count class."""
    _qp, si, _op, so, _fp, _pp, pb, rows, heads = a
    coarse = (pb, heads, si == heads * D, so == heads * D)
    return coarse, coarse + (min(rows.bit_length(), 15),)


def _launch(a, out_ptr=None, out_stride=None, device_index=0):
    qp, si, op, so, fp, pp, pb, rows, heads = a
    _FN["run"](qp, si, op if out_ptr is None else out_ptr, so if out_stride is None else out_stride, fp, pp, pb, rows,
               heads, _STATE["pdl"], _FN["raw"](device_index))


def _fast(self, q, positions, q_out) -> bool:
    """True when q_out holds the call's bytes (the unit ran, or a check ran stock into it)."""
    a = _args(q, positions, q_out, self.freqs_cis)
    if a is None:
        return False
    coarse, fine = _key(a)
    idx = q.get_device()
    if torch.cuda.is_current_stream_capturing():
        if _OK.get(coarse) is not True:
            return False
        try:
            _launch(a, device_index=idx)
        except Exception as e:  # noqa: BLE001 - a failed launch: stock from now on, loudly
            _off(f"launch at {a[7]} rows under capture failed ({type(e).__name__}: {str(e)[:200]})")
            return False
        _STATE["fast"] += 1
        return True
    ok = _OK.get(fine)
    if ok is True:
        try:
            _launch(a, device_index=idx)
        except Exception as e:  # noqa: BLE001
            _off(f"launch at {a[7]} rows failed ({type(e).__name__}: {str(e)[:200]})")
            return False
        _STATE["fast"] += 1
        return True
    if ok is False:
        return False
    return _check(self, q, positions, q_out, a, coarse, fine, idx)


def _check(self, q, positions, q_out, a, coarse, fine, idx) -> bool:
    """Stock into q_out and the unit into a scratch buffer on the same input; every byte compared. q_out holds stock's
    bytes either way (True)."""
    _FN["fqnr"](q, q_out, None, self.freqs_cis, positions)
    why = ""
    try:
        # the call's own output layout (row stride included), so the check exercises the addressing the fast path uses
        got = torch.empty_strided(tuple(q_out.shape), tuple(q_out.stride()), dtype=FP8, device=q.device)
        if got.data_ptr() & 7:
            raise RuntimeError("scratch not 8-byte aligned")
        _launch(a, out_ptr=got.data_ptr(), device_index=idx)
        if not torch.equal(got.view(U8), q_out.view(U8)):
            why = "q_out bytes"
    except Exception as e:  # noqa: BLE001 - any failure keeps stock
        why = f"{type(e).__name__}: {str(e)[:200]}"
    _STATE["checks"] += 1
    _OK[fine] = not why
    if why:
        _OK[coarse] = False
        _off(f"key {fine} at {a[7]} rows: {why} differ from stock")
        return True
    _OK.setdefault(coarse, True)
    levers.announce(f"qr-{fine}", f"lever QR: key {fine} first at {a[7]} rows: q_out bytes == stock: on")
    return True


def _off(why: str) -> None:
    _STATE["ok"] = False
    print(f"CACHEON_DSV4F lever QR: {why}: lever QR OFF", file=sys.stderr, flush=True)


def _self_test(targets) -> str:
    """Stock's fused_q_norm_rope vs the unit on synthetic rows with a served rope table (int64 and int32 positions,
    1..256 rows, finite / special values): '' and the coarse keys verified, or why not."""
    attn = targets[0]
    freqs = getattr(attn, "freqs_cis", None)
    heads = int(getattr(attn, "n_local_heads", 0))
    if (not isinstance(freqs, torch.Tensor) or freqs.dtype is not torch.complex64 or freqs.dim() != 2
            or freqs.shape[1] != ROPE // 2 or not freqs.is_cuda or heads < 1 or int(attn.head_dim) != D):
        return "replica rope table / heads not as the unit expects"
    dev = freqs.device
    g = torch.Generator(device=dev)
    g.manual_seed(20261015)
    special = torch.tensor([float("nan"), -float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 3.0e38, -3.0e38,
                            448.0, -464.0, 480.0, 1.0e-40, -2.0 ** -10, 2.0 ** -9], device=dev).to(BF16)
    with torch.cuda.device(dev):
        for rows in (1, 5, 64, 129, 256):
            q = (torch.randn(rows, heads, D, device=dev, generator=g) * 4).to(BF16)
            flat = q.view(-1)
            pick = torch.randint(0, flat.numel(), (min(4096, flat.numel()),), device=dev, generator=g)
            flat[pick] = special[torch.randint(0, special.numel(), pick.shape, device=dev, generator=g)]
            for dtype in (torch.int64, torch.int32):
                pos = torch.randint(0, freqs.shape[0], (rows,), device=dev, generator=g).to(dtype)
                want = torch.empty(q.shape, dtype=FP8, device=dev)
                _FN["fqnr"](q, want, None, freqs, pos)
                got = torch.full(q.shape, 0x5A, dtype=U8, device=dev)
                a = _args(q, pos, got.view(FP8), freqs)
                if a is None:
                    return f"self-test call at {rows} rows not eligible"
                _launch(a, device_index=dev.index if dev.index is not None else torch.cuda.current_device())
                if not torch.equal(got, want.view(U8)):
                    n = int((got != want.view(U8)).sum())
                    return f"self-test: {n} q_out bytes differ from stock at {rows} rows, positions {dtype}"
                coarse, fine = _key(a)
                _OK[coarse] = True
                _OK[fine] = True
    _STATE["checks"] += 10
    return ""


def summary() -> str:
    return f"{_STATE['fast']} fast calls, {_STATE['checks']} checks, {'on' if _STATE['ok'] else 'off'}"
