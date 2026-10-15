"""Lever FMT on the fork (levers.FMT_TACTICS): extend (prefill) MoE calls of a measured row band run FlashInfer's routed
NVFP4 expert op (custom op flashinfer::trtllm_fp4_block_scale_moe) at a measured launch tactic instead of the
autotuner's warmup choice (lev_MT's idea on the arena line; the fork's 4bd18f1 pins decode batches the same way).

Why: FlashInfer picks the routed op's tactic at boot by timing warmup forwards that route every row to the same top-k
experts; on real routing the expert load is spread and other tiles stream the weights faster (arena line: -35 % op time
at 4096 rows, bit-identical; fork shape: fork/tests/fork_mt.py).

Mechanism: a class-level AutoTuner.choose_one wrapper, installed once in prepare() (outside any capture), forces the
tactic only while this module's ContextVar holds one and only for that op; every other call -- other ops, other
threads, every call outside the wrapped MoE forward -- reaches the choose_one it wrapped (stock's, or the fork's own
decode pin, which delegates the same way). The ContextVar is set by the MoE replicas' forward for extend calls whose row
count falls in a FMT_TACTICS band and the tactic is valid for that shape.

Exactness: tactics differ in launch shape only; the probe found every tactic bit-identical at the fork's shape. The
first eager call of each band runs the stock forward and the forced forward on the same input and compares every
output byte; a difference turns that band off for the process (stock's choice from then on).
"""
import contextvars

import torch

from dsv41f import levers

OP = "flashinfer::trtllm_fp4_block_scale_moe"
_PIN = contextvars.ContextVar("dsv41f_fmt_pin", default=None)
_INSTALLED: list = []
_BAND_OK: dict = {}      # band index -> True (bits checked equal) / False (off for the process)
_VALID: dict = {}        # (rows, tactic) -> the op's own valid-tactic entry for it at that shape, or False


def _pair(t):
    """A valid-tactic entry as a tuple of ints. FlashInfer 0.7.1's trtllm_get_valid_moe_configs returns TVM-FFI sequence
    objects, not Python lists / tuples: the old isinstance(list, tuple) test never matched, so every band fell back to
    the autotuner's choice (in-engine probe fdiag, 10-14)."""
    try:
        return tuple(int(x) for x in t)
    except (TypeError, ValueError):
        return None


def _install() -> None:
    if _INSTALLED:
        return
    from flashinfer.autotuner import AutoTuner

    wrapped = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        pin = _PIN.get()
        if pin is not None and custom_op == OP:
            tactic, rows = pin
            key = (rows, tuple(tactic))
            ok = _VALID.get(key)
            if ok is None:
                want = tuple(int(x) for x in tactic)
                try:
                    valid = runners[0].get_valid_tactics(inputs, None)
                    ok = next((t for t in valid if _pair(t) == want), False)
                except Exception:  # noqa: BLE001
                    ok = False
                _VALID[key] = ok
                if ok is False:
                    levers.announce(f"fmt-invalid{want}", f"lever FMT: tactic {list(want)} not valid at {rows} rows "
                                    f"(stock's choice)")
            if ok is not False:
                return runners[0], [int(x) for x in ok]   # a plain list, as the fork's own pin returns (lever RR /
                                                          # MO learn hooks read the tactic as a list / tuple)
        return wrapped(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = choose_one
    _INSTALLED.append(True)


def armed() -> bool:
    return bool(_INSTALLED)


def _decode_bands():
    return tuple(getattr(levers, "FMT_DECODE_TACTICS", ()))


def prepare() -> str:
    if not levers.FMT_TACTICS and not _decode_bands():
        return "off (FMT_TACTICS / FMT_DECODE_TACTICS empty)"
    # the fork's own decode pin (flashinfer_trtllm._install_expert_tactic_pin, installed lazily at its first decode
    # call) wraps whatever choose_one is installed then and answers its pinned calls without asking the hooks inside it;
    # installed first here -- always, with or without decode bands -- so it stays innermost and the hooks installed
    # after it (ours, lever RR's tactic learning, lever MO's) see its answers. Installed lazily after RR, it would hide
    # every decode tactic from RR (bench n51f: RR never armed, the routing kernel back, k4 +4 %).
    try:
        from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as _ft

        _ft._install_expert_tactic_pin()
    except (ImportError, AttributeError):   # a stack without the pin: ours wraps stock's choose_one directly
        pass
    _install()
    return (f"armed: extend MoE calls pinned by band {levers.FMT_TACTICS}; decode / verify bands "
            f"{_decode_bands()}")


def _band(rows: int, decode: bool = False):
    bands = _decode_bands() if decode else levers.FMT_TACTICS
    for i, (lo, hi, tactic) in enumerate(bands):
        key = ("d", i) if decode else i
        if lo <= rows <= hi and _BAND_OK.get(key, True):
            return key, tactic
    return None, None


def _flat(value, out):
    if torch.is_tensor(value):
        out.append(value)
    elif isinstance(value, (tuple, list)):
        for v in value:
            _flat(v, out)
    elif hasattr(value, "_fields"):
        for v in value:
            _flat(v, out)
    return out


def _same(a, b) -> bool:
    ta, tb = _flat(a, []), _flat(b, [])
    if len(ta) != len(tb) or not ta:
        return False
    for x, y in zip(ta, tb):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if x.dtype.is_floating_point and x.element_size() == 2:
            if not torch.equal(x.view(torch.int16), y.view(torch.int16)):
                return False
        elif not torch.equal(x, y):
            return False
    return True


def bind(moe) -> bool:
    """The MoE replica's forward: stock's, with the tactic pinned for extend calls in a band."""
    if not (levers.FMT_TACTICS or _decode_bands()) or not _INSTALLED:
        return False
    stock_forward = moe.forward

    def forward(hidden_states, forward_batch=None, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return stock_forward(hidden_states, forward_batch, *args, **kwargs)
        if mode.is_extend_without_speculative():
            band, tactic = _band(int(hidden_states.shape[0]))
        elif mode.is_decode() or mode.is_target_verify():
            band, tactic = _band(int(hidden_states.shape[0]), decode=True)
        else:
            band, tactic = None, None
        if band is None:
            return stock_forward(hidden_states, forward_batch, *args, **kwargs)
        if band not in _BAND_OK:
            if torch.cuda.is_current_stream_capturing():
                return stock_forward(hidden_states, forward_batch, *args, **kwargs)
            ref = stock_forward(hidden_states, forward_batch, *args, **kwargs)
            token = _PIN.set((tactic, int(hidden_states.shape[0])))
            try:
                got = stock_forward(hidden_states, forward_batch, *args, **kwargs)
            finally:
                _PIN.reset(token)
            ok = _same(ref, got)
            _BAND_OK[band] = ok
            bands = _decode_bands() if isinstance(band, tuple) else levers.FMT_TACTICS
            label = band[1] if isinstance(band, tuple) else band
            levers.announce(f"fmt-band{band}", f"lever FMT band {bands[label]} first at "
                            f"{hidden_states.shape[0]} rows: forced bits {'== stock: on' if ok else '!= stock: OFF'}")
            return got if ok else ref
        token = _PIN.set((tactic, int(hidden_states.shape[0])))
        try:
            return stock_forward(hidden_states, forward_batch, *args, **kwargs)
        finally:
            _PIN.reset(token)

    moe.forward = forward
    return True
