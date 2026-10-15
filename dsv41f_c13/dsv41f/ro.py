"""Lever RO (CUDA unit ro_unit): the inverse RoPE of OUR attention replicas' output -- stock's
fused_rope_inplace(o[..., -64:], None, freqs_cis, positions, inverse=True) in the lifted MQALayer.forward
(dsv41f/fork_attn.py), the 32 local heads' 128-byte rope tails rotated in place every layer -- as ONE launch of a kernel
with stock's per-pair arithmetic and far more memory-level parallelism, the same bytes.

Stock (fork 5e72ce5): sglang.kernels.ops.attention.dsv4.elementwise.fused_rope_inplace -> the JIT
sglang::deepseek_rope_kernel<is_arch_support_pdl(), true, int32 / int64> (deepseek_v4/rope.cuh): one warp per
(token, head), one bf16x2 pair per lane, position -> freq -> data -> store, so each SM has ~8 KB of loads in flight
(12096-row prefill pass: 78.8 us a call, 99 MB moved, 1.26 TB/s, x 40 layers). The unit (ro_unit.cu header): 16-byte
loads, a thread's four freq pairs loaded once for its heads (levers.RO_TILES by row count: 8 heads a thread = a warp a
token at prefill sizes, fewer at small ones), stock's FMUL / FFMA order and bf16 RNE pack per pair, stock's PDL
placement (position + freq loads before griddepcontrol.wait, launch_dependents after the stores; programmatic stream
serialization when stock's kUsePDL = is_arch_support_pdl()).

Binding: none. gen_ro.py edits only the bundle's own fork_attn.py -- the lifted MQALayer.forward that model_node binds
as the forward of OUR private attention replicas (copies; the served MQALayer instances and the class keep stock's
forward) -- so that one call site calls inv_rope() below with stock's arguments. No served module, class or sglang
function is patched or rebound (cacheon's node seal); the draft model (its own MQALayer.forward) never reaches it.
inv_rope() runs stock's fused_rope_inplace for every call it does not take, installed or not.

Scope: every forward of our replicas (eager EXTEND / MIXED, decode / verify eager or under CUDA-graph capture): a call
with k None, inverse True, a bf16 [rows >= 1, 16 / 32 / 64 heads, 64] tail (last dim contiguous, 16-byte aligned rows /
heads, non-overlapping heads), a contiguous unconjugated complex64 [*, 32] CUDA table and contiguous int32 / int64
positions of the same length on the same device, outside torch.compile. Everything else -> stock.

Exactness: tests/ro_test.py (B300, fork image) compares stock's and the unit's SASS per pair, and the bytes of the whole
attention output (tails rotated, every other byte untouched) over every row count 1..640, the CTA edges to 16384,
random rows, int32 / int64 positions to the model's 1M context, the model's two freqs tables, random / adversarial bf16
bit patterns (NaN payloads and signs, +-Inf, huge, subnormals, -0), adversarial fp32 tables, padded / strided outputs,
in place as stock. In the engine the first eager call of every check key (table, heads, position dtype, strides, tile,
rows mod tokens per CTA) runs stock on a copy of the tail and the unit on the tail itself and compares every byte; a
difference restores stock's bytes and turns the lever off for the process. A CUDA-graph capture uses the unit only for
keys verified eagerly (stock otherwise). install() pins the source of the stock functions the unit mirrors or relies on
(sha256 of the AST segment); any difference keeps stock.
"""
import ast
import hashlib
import inspect
import sys
import textwrap

import torch

from dsv41f import levers

ROPE = 64
HEADS = (16, 32, 64)
BF16 = torch.bfloat16
I16 = torch.int16
I32 = torch.int32
I64 = torch.int64
C64 = torch.complex64
_STATE = {"detail": "not installed", "ok": False, "fast": 0, "checks": 0, "replicas": 0}
_OK: dict = {}            # check key -> True (bytes == stock) / False
_TAB: dict = {}           # id(freqs_cis) -> (freqs_cis, data_ptr or None, device index)
_FN: dict = {}            # run, stock, raw, pdl, tiles

# sha256[:16] of the source segment of every stock function the unit mirrors or this module relies on (fork 5e72ce5);
# any difference: lever RO off
PINNED = {
    ("sglang.kernels.ops.attention.dsv4.elementwise", "fused_rope_inplace"): "f69647bfa5ea0c8e",
    ("sglang.kernels.ops.attention.dsv4.elementwise", "_jit_fused_rope_module"): "7aec66c6d104d84d",
    ("sglang.kernels.jit.utils.arch", "is_arch_support_pdl"): "d6dbc4ab73e8b3cb",
    ("sglang.srt.models.deepseek_v4", "MQALayer.forward"): "b1423a431a7aaca7",
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
            import ro_unit
        except ImportError as err:
            levers.announce("ro-missing", f"bundle CUDA unit ro_unit not importable ({err}): lever RO off")
            return None
        _FN["run"] = ro_unit.run
    return _FN["run"]


def _tiles():
    """levers.RO_TILES as ((max rows, heads per thread, tokens per CTA, fixed stride), ...), ascending max rows; raises
    on a malformed table."""
    tiles = tuple((int(m), int(i), int(t), int(f)) for m, i, t, f in levers.RO_TILES)
    tops = [m for m, _, _, _ in tiles]
    if (not tiles or tops != sorted(tops) or any(i not in (1, 2, 4, 8) or t < 1 or f not in (0, 1)
                                                for _, i, t, f in tiles)):
        raise RuntimeError(f"levers.RO_TILES {levers.RO_TILES!r}")
    return tiles


def _calls_ro(fn) -> bool:
    """Whether the lifted attention forward calls inv_rope (as `ro.inv_rope(`) and no longer calls stock's
    fused_rope_inplace with inverse=True (gen_ro.py's edit)."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001 - no source: reported as not edited
        return False
    ours = stock = 0
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        inv = any(kw.arg == "inverse" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                  for kw in n.keywords)
        if (isinstance(n.func, ast.Attribute) and n.func.attr == "inv_rope" and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "ro"):
            ours += 1
        elif isinstance(n.func, ast.Name) and n.func.id == "fused_rope_inplace" and inv:
            stock += 1
    return ours == 1 and stock == 0


def install(model=None, layers=None) -> str:
    """Pin the stock internals, check the lifted attention forward's edit and load the unit (model_node.prepare,
    before any capture). `layers` (the private replicas) is only counted for the announce line."""
    if _STATE["ok"] or _STATE["detail"].startswith("off"):
        return _STATE["detail"]
    if not getattr(levers, "RO", False):
        _STATE["detail"] = "off (levers.RO)"
        return _STATE["detail"]
    try:
        mods = _modules()
        cache: dict = {}
        bad = [f"{mod}.{name} {got}" for (mod, name), want in PINNED.items()
               if (got := source_hash(mods[mod], name, cache)) != want]
        if bad:
            raise RuntimeError("source differs from the mirror: " + ", ".join(bad))
        ew = mods["sglang.kernels.ops.attention.dsv4.elementwise"]
        dsv4 = mods["sglang.srt.models.deepseek_v4"]
        from dsv41f import fork_attn

        stock = ew.fused_rope_inplace
        if dsv4.fused_rope_inplace is not stock or fork_attn.fused_rope_inplace is not stock:
            raise RuntimeError("deepseek_v4 / fork_attn do not bind elementwise.fused_rope_inplace")
        if ew._is_hip or ew._is_xpu or not dsv4._is_cuda or dsv4._is_hip or dsv4._is_npu:
            raise RuntimeError("not the CUDA path of fused_rope_inplace")
        if not _calls_ro(fork_attn.attn_forward):
            raise RuntimeError("fork_attn.attn_forward does not call ro.inv_rope in place of fused_rope_inplace")
        tiles = _tiles()
        if unit() is None:
            raise RuntimeError("unit ro_unit missing")
        pdl = 1 if ew.is_arch_support_pdl() else 0     # stock's kUsePDL (make_cpp_args(is_arch_support_pdl()))
        raw = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        _FN.update(stock=stock, pdl=pdl, tiles=tiles,
                   raw=raw if raw is not None else (lambda idx: torch.cuda.current_stream(idx).cuda_stream))
        replicas = 0
        for r in layers or ():
            attn = r._modules.get("self_attn") if hasattr(r, "_modules") else None
            replicas += getattr(getattr(attn, "forward", None), "__func__", None) is fork_attn.attn_forward
    except Exception as e:  # noqa: BLE001 - any missing / changed internal keeps stock
        _STATE["detail"] = f"off ({type(e).__name__}: {str(e)[:400]})"
        return _STATE["detail"]
    _STATE["ok"] = True
    _STATE["replicas"] = replicas
    _STATE["detail"] = (f"on: inverse RoPE of the attention output on {replicas} attention replicas (fork_attn), PDL "
                        f"{'on' if pdl else 'off'} (stock's kUsePDL), tiles (max rows, heads a thread, tokens a CTA, "
                        f"fixed stride) {tiles}; first eager call per key checked vs stock")
    return _STATE["detail"]


def _stock():
    stock = _FN.get("stock")
    if stock is None:
        from sglang.srt.models.deepseek_v4 import fused_rope_inplace as stock
    return stock


def inv_rope(q, k, freqs_cis, positions, inverse=False):
    """fork_attn's inverse RoPE call (stock's fused_rope_inplace arguments): the unit for an eligible call, stock's
    fused_rope_inplace otherwise; q is rotated in place either way and None is returned, as stock does."""
    if (k is None and inverse is True and _STATE["ok"] and not torch.compiler.is_compiling()
            and _fast(q, freqs_cis, positions)):
        return None
    return _stock()(q, k, freqs_cis, positions=positions, inverse=inverse)


def _table(freqs_cis):
    """(freqs_cis, data pointer, device index) of a table the unit reads as stock's view_as_real(...).flatten(-2), or
    (freqs_cis, None, -1)."""
    ent = (freqs_cis, None, -1)
    try:
        f = freqs_cis
        if (type(f) is torch.Tensor and f.dtype is C64 and f.dim() == 2 and f.shape[1] == ROPE // 2 and f.is_cuda
                and f.is_contiguous() and not f.is_conj() and not f.data_ptr() & 15):
            ent = (f, f.data_ptr(), f.get_device())
    except Exception:  # noqa: BLE001 - an unexpected table keeps stock
        pass
    _TAB[id(freqs_cis)] = ent
    return ent


def _tile(rows, heads):
    """(heads per thread, tokens per CTA, fixed stride) of the first RO_TILES row band holding `rows`; the CTA is capped
    at 512 threads (a token = 8 * heads / heads-per-thread threads)."""
    tile = _FN["tiles"][-1]
    for t in _FN["tiles"]:
        if rows <= t[0]:
            tile = t
            break
    _, it, tok, fx = tile
    return it, max(1, min(tok, 512 // (8 * heads // it))), fx


def _fast(q, freqs_cis, positions):
    if type(q) is not torch.Tensor or type(positions) is not torch.Tensor or q.dtype is not BF16 or q.dim() != 3:
        return False
    rows, heads, d = q.shape
    sb, sh, sd = q.stride()
    if (d != ROPE or sd != 1 or heads not in HEADS or rows < 1 or sb & 7 or sh & 7 or sh < ROPE
            or (rows > 1 and sb < heads * sh)):
        return False
    tab = _TAB.get(id(freqs_cis))
    if tab is None or tab[0] is not freqs_cis:
        tab = _table(freqs_cis)
    fptr = tab[1]
    if fptr is None or freqs_cis.data_ptr() != fptr:
        return False
    pdt = positions.dtype
    if pdt is I64:
        p64 = 1
    elif pdt is I32:
        p64 = 0
    else:
        return False
    if positions.dim() != 1 or positions.shape[0] != rows or (rows > 1 and positions.stride(0) != 1):
        return False
    dev = q.get_device()
    if dev < 0 or dev != tab[2] or positions.get_device() != dev:
        return False
    qp = q.data_ptr()
    if qp & 15:
        return False
    it, tok, fx = _tile(rows, heads)
    key = (fptr, heads, p64, sb, sh, it, tok, fx, rows % tok)
    ok = _OK.get(key)
    if ok:
        _FN["run"](qp, sb, sh, heads, fptr, positions.data_ptr(), p64, rows, it, tok, fx, _FN["pdl"], _FN["raw"](dev))
        _STATE["fast"] += 1
        return True
    if ok is None and not torch.cuda.is_current_stream_capturing():
        return _check(key, q, freqs_cis, positions, rows, heads, sb, sh, fptr, p64, (it, tok, fx), dev)
    return False


def _check(key, q, freqs_cis, positions, rows, heads, sb, sh, fptr, p64, tile, dev):
    """Stock on a copy of the tail, the unit on the tail itself; the unit's bytes if every byte equals stock's (key
    on), stock's bytes restored otherwise (lever off for the process). Exceptions of stock's own call propagate as
    from stock."""
    ref = q.clone()
    _stock()(ref, None, freqs_cis, positions=positions, inverse=True)
    why = ""
    try:
        _FN["run"](q.data_ptr(), sb, sh, heads, fptr, positions.data_ptr(), p64, rows, *tile, _FN["pdl"],
                   _FN["raw"](dev))
        if not torch.equal(q.view(I16), ref.view(I16)):
            why = "bytes"
    except Exception as e:  # noqa: BLE001 - any failure keeps stock
        why = f"{type(e).__name__}: {str(e)[:200]}"
    _STATE["checks"] += 1
    _OK[key] = not why
    if why:
        q.copy_(ref)
        _STATE["ok"] = False
        print(f"CACHEON_DSV4F lever RO: key {key[1:]} at {rows} rows: {why} differs from stock: lever RO OFF",
              file=sys.stderr, flush=True)
        return True
    table = _TABLE_NO.setdefault(fptr, len(_TABLE_NO) + 1)
    levers.announce(f"ro-{key}", f"lever RO: key {key[1:]} (freqs table #{table}) first at {rows} rows == stock: on")
    return True


_TABLE_NO: dict = {}      # data pointer -> 1, 2, ... in check order (announce detail only)


def summary() -> str:
    return f"{_STATE['fast']} fast calls, {_STATE['checks']} checks, {'on' if _STATE['ok'] else 'off'}"
