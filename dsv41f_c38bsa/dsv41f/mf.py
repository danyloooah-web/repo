"""Lever MF: our eager EXTEND attention's sparse-MLA launch (FlashInfer 0.7.1 trtllm_batch_decode_sparse_mla_dsv4, called by
the lifted _forward_trtllm_prefill in ct_pf.py, backend "trtllm-gen", varlen query) goes straight to the C++ launcher stock's
wrapper ends in (get_trtllm_gen_fmha_module().trtllm_paged_attention_decode_sparse_mla_dsv4) with the arguments that wrapper
derives, skipping its ~40 argument checks and normalizations (pure Python: views, dtype / shape checks, backend resolution).

Exactness: the same launcher with the same arguments -> the same kernel and bytes. The derivation below is stock's for this call
pattern (HND caches, varlen q with max_q_len, metadata rows == query rows, no RopeQuant, no index remap): every step is a view or
a check that does not depend on the row count. The first call of every key (dtypes, head count, strides, batch size, out / sinks
presence, scale types) runs stock with its launcher call recorded through a temporary proxy of the flashinfer module function
get_trtllm_gen_fmha_module (no served module touched), derives our arguments for the same inputs and compares them one by one
(tensors: same storage pointer, shape, strides, dtype, device; fresh allocations -- out when stock allocates it, the multi-CTA
counter buffer -- by shape, strides, dtype, device; other values by equality), then launches ours on the same inputs and compares
the output bytes with stock's. Any difference: that key stays stock for the process. Stock's sources are pinned (sha256 of their
inspect.getsource text) at install; any difference leaves the lever off.
"""
import hashlib
import inspect
import sys

import torch

from dsv41f import levers

_ST = {"detail": "not installed", "fast": 0, "stock": 0, "learn": 0, "bad": 0}
_H: dict = {}
_KEYS: dict = {}          # key -> True (verified) / False (stock for good)

# (the multi-CTA counter resolver is not pinned: lever CT wraps it, and both paths call it through the module attribute)
PIN_NAMES = ("trtllm_batch_decode_sparse_mla_dsv4", "_check_dsv4_sparse_mla_inputs", "_normalize_dsv4_sparse_mla_kv_cache",
             "_normalize_dsv4_topk_lens", "_resolve_dsv4_sparse_mla_backend", "_validate_dsv4_sync_checks")


def _src_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()[:16]


def install() -> str:
    """Resolve and pin the FlashInfer internals (model_node.prepare). Off on any surprise."""
    if not levers.MF:
        _ST["detail"] = "off (levers)"
        return _ST["detail"]
    try:
        import flashinfer
        from flashinfer.mla import _core as fc

        if getattr(flashinfer, "__version__", "") != "0.7.1":
            raise RuntimeError(f"flashinfer {getattr(flashinfer, '__version__', '?')} != 0.7.1")
        fns = (fc.trtllm_batch_decode_sparse_mla_dsv4, fc._check_dsv4_sparse_mla_inputs,
               fc._normalize_dsv4_sparse_mla_kv_cache, fc._normalize_dsv4_topk_lens, fc._resolve_dsv4_sparse_mla_backend,
               fc._validate_dsv4_sync_checks)
        pins = {n: _src_hash(f) for n, f in zip(PIN_NAMES, fns)}
        bad = [n for n, h in pins.items() if levers.MF_PINS.get(n) != h]
        if bad:
            _ST["detail"] = f"off (stock changed: {bad} {[pins[n] for n in bad]})"
            return _ST["detail"]
        op = fc.get_trtllm_gen_fmha_module()
        run = getattr(op, "trtllm_paged_attention_decode_sparse_mla_dsv4", None)
        if run is None:
            raise RuntimeError("no trtllm_paged_attention_decode_sparse_mla_dsv4 launcher")
        from flashinfer.utils import get_device_sm_count

        _H.update(fc=fc, stock=fc.trtllm_batch_decode_sparse_mla_dsv4, getmod=fc.get_trtllm_gen_fmha_module, run=run,
                  sm=get_device_sm_count, pdl=fc.device_support_pdl)
    except Exception as e:  # noqa: BLE001 - another stack keeps stock's wrapper
        _ST["detail"] = f"off ({type(e).__name__}: {str(e)[:200]})"
        return _ST["detail"]
    _ST["detail"] = ("on: eager EXTEND sparse-MLA launches go straight to stock's trtllm-gen launcher with stock's derived "
                     "arguments; first call per key recorded + compared (arguments and output bytes)")
    return _ST["detail"]


def _cache4(kv):
    """stock _normalize_dsv4_sparse_mla_kv_cache for kv_layout HND (validated at the key's first call)."""
    return kv.unsqueeze(1) if kv.ndim == 3 else kv


def _strided(kv) -> bool:
    return kv.stride(-2) != kv.size(-1) or kv.stride(0) != kv.size(-2) * kv.stride(-2)


def _derive(query, swa_kv_cache, workspace_buffer, sparse_indices, compressed_kv_cache, sparse_topk_lens, seq_lens, out,
            bmm1_scale, bmm2_scale, sinks, cum_seq_lens_q, max_q_len):
    """The launcher's positional arguments as stock's wrapper derives them for this call pattern."""
    swa = _cache4(swa_kv_cache)
    comp = _cache4(compressed_kv_cache)
    batch_size = cum_seq_lens_q.numel() - 1
    num_heads = query.shape[1]
    if out is None:
        out = torch.empty((query.shape[0], num_heads, 512), dtype=torch.bfloat16, device=query.device)
    rows = sparse_indices.shape[0]
    si = sparse_indices.reshape(rows, -1).contiguous()
    lens = sparse_topk_lens.contiguous()
    sm = _H["sm"](query.device)
    counter = _H["fc"]._resolve_trtllm_gen_multi_ctas_kv_counter_buffer(None, batch_size, num_heads, sm, query.device)
    # enable_pdl: stock resolves None to device_support_pdl(device) (functools.cache); scales are host floats here
    return (out, query, comp, swa, workspace_buffer, counter, si, si, False, seq_lens.contiguous(), lens, bmm1_scale,
            bmm2_scale, batch_size, max_q_len, sm, _H["pdl"](query.device),
            workspace_buffer.numel() * workspace_buffer.element_size(), sinks, cum_seq_lens_q.contiguous(), None, None)


def _key(query, swa_kv_cache, sparse_indices, compressed_kv_cache, sparse_topk_lens, seq_lens, out, bmm1_scale,
         bmm2_scale, sinks, cum_seq_lens_q):
    return (query.dtype, query.shape[1:], query.stride(), sparse_indices.dtype, sparse_indices.shape[1],
            sparse_indices.is_contiguous(), swa_kv_cache.dtype, swa_kv_cache.shape[1:], swa_kv_cache.stride(),
            compressed_kv_cache.dtype, compressed_kv_cache.shape[1:], compressed_kv_cache.stride(),
            sparse_topk_lens.dtype, sparse_topk_lens.is_contiguous(), seq_lens.dtype, seq_lens.is_contiguous(),
            cum_seq_lens_q.dtype, cum_seq_lens_q.numel(), out is None, sinks is None, type(bmm1_scale), type(bmm2_scale))


def _same_arg(a, b, fresh: bool) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return False
        if (a.shape != b.shape or a.stride() != b.stride() or a.dtype != b.dtype or a.device != b.device):
            return False
        return fresh or (a.data_ptr() == b.data_ptr() and a.storage_offset() == b.storage_offset())
    return type(a) is type(b) and a == b


def sparse_mla(*, query, swa_kv_cache, workspace_buffer, sparse_indices, compressed_kv_cache, sparse_topk_lens, seq_lens,
               out, bmm1_scale, bmm2_scale, sinks, kv_layout, backend, cum_seq_lens_q, max_q_len):
    """ct_pf's call site: stock's keyword arguments; the fast launch where verified, stock otherwise."""
    stock = _H.get("stock")
    kw = dict(query=query, swa_kv_cache=swa_kv_cache, workspace_buffer=workspace_buffer, sparse_indices=sparse_indices,
              compressed_kv_cache=compressed_kv_cache, sparse_topk_lens=sparse_topk_lens, seq_lens=seq_lens, out=out,
              bmm1_scale=bmm1_scale, bmm2_scale=bmm2_scale, sinks=sinks, kv_layout=kv_layout, backend=backend,
              cum_seq_lens_q=cum_seq_lens_q, max_q_len=max_q_len)
    if stock is None:
        from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4 as fallback

        return fallback(**kw)
    if (backend != "trtllm-gen" or kv_layout != "HND" or cum_seq_lens_q is None or max_q_len is None
            or isinstance(bmm1_scale, torch.Tensor) or isinstance(bmm2_scale, torch.Tensor)
            or query.ndim != 3 or sparse_indices.ndim != 2 or sparse_indices.shape[0] != query.shape[0]
            or swa_kv_cache.ndim not in (3, 4) or compressed_kv_cache.ndim not in (3, 4)
            or _H["fc"].trtllm_batch_decode_sparse_mla_dsv4 is not stock or _H["fc"].get_trtllm_gen_fmha_module is not _H["getmod"]
            or torch.cuda.is_current_stream_capturing()):
        _ST["stock"] += 1
        return stock(**kw)
    key = _key(query, swa_kv_cache, sparse_indices, compressed_kv_cache, sparse_topk_lens, seq_lens, out, bmm1_scale,
               bmm2_scale, sinks, cum_seq_lens_q)
    ok = _KEYS.get(key)
    if ok is None:
        return _learn(key, kw)
    if not ok or _strided(_cache4(compressed_kv_cache)) or _strided(_cache4(swa_kv_cache)):
        _ST["stock"] += 1
        return stock(**kw)
    try:
        args = _derive(query, swa_kv_cache, workspace_buffer, sparse_indices, compressed_kv_cache, sparse_topk_lens,
                       seq_lens, out, bmm1_scale, bmm2_scale, sinks, cum_seq_lens_q, max_q_len)
        _H["run"](*args)
    except Exception as e:  # noqa: BLE001 - raised before the launch (rank-local, no collective): stock for this key
        _KEYS[key] = False
        _ST["bad"] += 1
        levers.announce(f"mf-err-{type(e).__name__}", f"lever MF: fast launch failed ({type(e).__name__}: "
                        f"{str(e)[:120]}): stock for that key")
        return stock(**kw)
    _ST["fast"] += 1
    return args[0]


def _learn(key, kw):
    """Key's first call: stock with its launcher call recorded, ours derived and compared; then ours launched and its
    output bytes compared with stock's. Returns stock's output."""
    fc, getmod, run = _H["fc"], _H["getmod"], _H["run"]
    rec = []

    def recorder(*a):
        rec.append(a)
        return run(*a)

    class _Proxy:
        """stock's wrapper reads exactly one attribute of the module object: its sparse-MLA launcher."""

    proxy = _Proxy()
    proxy.trtllm_paged_attention_decode_sparse_mla_dsv4 = recorder
    fc.get_trtllm_gen_fmha_module = lambda: proxy
    try:
        ref = _H["stock"](**kw)
    except AttributeError:
        ref = None                      # stock wanted another attribute: plain stock below, key off
    finally:
        fc.get_trtllm_gen_fmha_module = getmod
    _ST["learn"] += 1
    if ref is None:
        _KEYS[key] = False
        _ST["bad"] += 1
        levers.announce("mf-proxy", "lever MF: stock's wrapper read another launcher attribute: OFF for that key")
        return _H["stock"](**kw)
    why = ""
    try:
        if len(rec) != 1 or not isinstance(ref, torch.Tensor):
            why = f"stock made {len(rec)} launcher calls"
        elif _strided(_cache4(kw["compressed_kv_cache"])) or _strided(_cache4(kw["swa_kv_cache"])):
            why = "strided KV pages"
        else:
            ours = _derive(kw["query"], kw["swa_kv_cache"], kw["workspace_buffer"], kw["sparse_indices"],
                           kw["compressed_kv_cache"], kw["sparse_topk_lens"], kw["seq_lens"], kw["out"], kw["bmm1_scale"],
                           kw["bmm2_scale"], kw["sinks"], kw["cum_seq_lens_q"], kw["max_q_len"])
            got = rec[0]
            fresh = {5} | ({0} if kw["out"] is None else set())
            if len(got) != len(ours):
                why = f"launcher arity {len(got)} != {len(ours)}"
            else:
                for i, (a, b) in enumerate(zip(got, ours)):
                    if not _same_arg(a, b, i in fresh):
                        why = f"launcher argument {i} differs"
                        break
            if not why:
                if kw["out"] is not None:
                    ours = (torch.empty_like(kw["out"]),) + ours[1:]
                run(*ours)
                if not torch.equal(ours[0].view(torch.int16), ref.view(torch.int16)):
                    why = "output bytes differ"
    except Exception as e:  # noqa: BLE001 - a derivation that fails keeps stock for this key
        why = f"{type(e).__name__}: {str(e)[:120]}"
    _KEYS[key] = not why
    if why:
        _ST["bad"] += 1
    levers.announce(f"mf-key{len(_KEYS)}", f"lever MF: key #{len(_KEYS)} (batch {kw['cum_seq_lens_q'].numel() - 1}, first at "
                    f"{kw['query'].shape[0]} rows): {'arguments + output bytes == stock: on' if not why else 'OFF (' + why + ')'}")
    return ref


def stats() -> dict:
    return dict(_ST)
