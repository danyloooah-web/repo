"""Lever MOET on the fork (levers.MOET / MOET_TACTICS): decode / verify MoE calls of a measured row band run FlashInfer's
routed NVFP4 expert op (custom op flashinfer::trtllm_fp4_block_scale_moe, trtllm-gen FC1 + FC2) at a measured launch
tactic [tile, config] instead of the fork's own decode pin (_expert_decode_tactic: [32, 122] above 16 rows).

Why (fork/tests/moet_micro.py on box1, the engine's REAL routing recorded in-graph by the debug bundle c28moet_dbg:
top-k after stock E5 of every layer at 48 / 96 / 144 rows, cold weights, CUDA graphs): the routed GEMMs are at the
HBM streaming limit in isolation (FC1 6.7-7.1 TB/s = 96-99 % of a raw read of the same bytes), every one of the 576
valid tactics is bit-identical to the pin, and isolated no tactic is faster by more than noise; with the shared
expert's two dense GEMMs running beside the op (as the engine runs them, on a side stream) the 16-token tile with the
2-CTA cluster configs loses less: [16, 312] -0.86 % of the op at 96 rows (FC1 -1.0 / FC2 -0.5 us), -0.33 % at 144
(FC1 -1.5 us), -0.1 % at 48.

Mechanism: a class-level AutoTuner.choose_one wrapper installed once in prepare() (outside any capture), AFTER lever
FMT's (which installs the fork's decode pin first) and BEFORE lever RR's tactic hook, so RR learns and verifies the
tactic this lever answers (RR's record + body are per (rows, tile, config)). The wrapper answers only the routed op
and only while this module's ContextVar holds a pin; the ContextVar is set by the MoE replicas' forward (bind) for
DECODE / TARGET_VERIFY forwards whose row count lies in a MOET_TACTICS band and for which the tactic is valid at that
shape (the op's own get_valid_tactics: a host-side function of the shape). Everything else reaches the wrapped
choose_one unchanged.

Honesty / exactness: the decision depends only on the forward mode, the row count and the shape -- never on capture
state, call counts or anything that tells an audited call from a served one: the audit's eager decode / verify calls
run exactly the tactic the CUDA graphs run. Tactics differ in launch shape only (token tile x kernel config: every
output element is the same K accumulation of the same tcgen05 block-scaled MMA chain, the same fused SwiGLU + FP4
requant epilogue, the same per-(token, slot) GEMM2 rows); the micro compared every tactic's finalized output with the
pin's bit for bit on 8 real-routing calls at each of 48 / 96 / 144 rows (1728 tactic runs, 0 differences). No
runtime cross-tactic check (a fail-closed switch could leave already-captured graphs and later eager calls on
different tactics); lever RR still verifies its record + body against the stock call at this tactic per shape.
Rank symmetry: both TP ranks see the same rows, mode and shape -> the same tactic.
"""
import contextvars

from dsv41f import levers

OP = "flashinfer::trtllm_fp4_block_scale_moe"
_PIN = contextvars.ContextVar("dsv41f_moet_pin", default=None)
_INSTALLED: list = []
_VALID: dict = {}        # (rows, tactic) -> the op's own valid-tactic entry as a list, or False


def _pair(t):
    try:
        return tuple(int(x) for x in t)
    except (TypeError, ValueError):
        return None


def _bands():
    return tuple(getattr(levers, "MOET_TACTICS", ()))


def _install() -> None:
    if _INSTALLED:
        return
    from flashinfer.autotuner import AutoTuner

    wrapped = AutoTuner.choose_one

    def choose_one(self, custom_op, runners, tuning_config, inputs, **kwargs):
        pin = _PIN.get()
        if pin is not None and custom_op == OP:
            tactic, rows = pin
            key = (rows, tactic)
            ok = _VALID.get(key)
            if ok is None:
                try:
                    valid = runners[0].get_valid_tactics(inputs, None)
                    hit = next((t for t in valid if _pair(t) == tactic), None)
                    ok = [int(x) for x in hit] if hit is not None else False
                except Exception:  # noqa: BLE001 - an op without that API: the wrapped choice
                    ok = False
                _VALID[key] = ok
                levers.announce(f"moet-{rows}-{list(tactic)}", f"lever MOET: {rows}-row decode / verify MoE "
                                f"tactic {list(tactic)} " + ("valid: on" if ok else "not valid here: wrapped choice"))
            if ok is not False:
                return runners[0], list(ok)   # a plain list (lever RR / MO hooks read the tactic as a list)
        return wrapped(self, custom_op, runners, tuning_config, inputs, **kwargs)

    AutoTuner.choose_one = choose_one
    _INSTALLED.append(True)


def prepare() -> str:
    if not getattr(levers, "MOET", False) or not _bands():
        return "off (levers.MOET / MOET_TACTICS)"
    _install()
    return f"armed: decode / verify MoE calls pinned by band {_bands()}"


def _band(rows: int):
    for lo, hi, tactic in _bands():
        if lo <= rows <= hi:
            return tuple(int(x) for x in tactic)
    return None


def bind(moe) -> bool:
    """The MoE replica's forward (whatever is bound: lever FMT's wrapper or stock's): decode / verify calls in a band
    run with the band's tactic pinned for the routed op."""
    if not _INSTALLED or not _bands():
        return False
    inner = moe.forward

    def forward(hidden_states, forward_batch=None, *args, **kwargs):
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not (mode.is_decode() or mode.is_target_verify()):
            return inner(hidden_states, forward_batch, *args, **kwargs)
        rows = int(hidden_states.shape[0])
        tactic = _band(rows)
        if tactic is None:
            return inner(hidden_states, forward_batch, *args, **kwargs)
        token = _PIN.set((tactic, rows))
        try:
            return inner(hidden_states, forward_batch, *args, **kwargs)
        finally:
            _PIN.reset(token)

    moe.forward = forward
    moe.dsv4f_moet = True
    return True


def describe() -> str:
    on = sorted(k for k, v in _VALID.items() if v is not False)
    off = sorted(k for k, v in _VALID.items() if v is False)
    return f"bands {_bands()}; valid (rows, tactic) {on[:8]}{'...' if len(on) > 8 else ''}; invalid {off[:8]}"
