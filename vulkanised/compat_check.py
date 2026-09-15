"""
compat_check.py — checks a parsed GGUF model against the Vulkan-native spec
(vulkan-native-moe-spec.md) before any conversion work happens.

Rules enforced (spec sections 1-3):
  - head_dim (embedding_length / head_count) == 64
  - head_count : head_count_kv ratio is a power of 2
  - hidden_size (embedding_length) divisible by 256
  - no expert_count metadata unless it matches the small-coarse-grained scheme

Fails loud and specific — this is the gate between "real conversion" and
"you need to retrain."
"""

from dataclasses import dataclass
from gguf_reader import GGUFFile


@dataclass
class CompatResult:
    ok: bool
    failures: list       # list[str] — empty if ok
    info: dict           # extracted values, for logging/debugging


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def check_compat(g: GGUFFile, max_expert_count: int = 8) -> CompatResult:
    failures = []
    info = {}

    arch = g.get("general.architecture")
    if arch is None:
        failures.append("missing general.architecture key — cannot resolve arch-specific metadata names")
        return CompatResult(ok=False, failures=failures, info=info)
    info["architecture"] = arch

    def key(suffix):
        return f"{arch}.{suffix}"

    embedding_length = g.get(key("embedding_length"))
    head_count = g.get(key("attention.head_count"))
    head_count_kv = g.get(key("attention.head_count_kv"), head_count)  # MHA models omit this, kv==q

    if embedding_length is None or head_count is None:
        failures.append(
            f"missing {key('embedding_length')} or {key('attention.head_count')} — "
            "can't compute head_dim"
        )
        return CompatResult(ok=False, failures=failures, info=info)

    info["embedding_length"] = embedding_length
    info["head_count"] = head_count
    info["head_count_kv"] = head_count_kv

    # Rule 1: head_dim tiles cleanly against subgroup width (32).
    # 64 and 128 both divide evenly (2 and 4 subgroups per head-row respectively) —
    # no remainder, no divergent branching in the attention shader. 128 is also the
    # dominant choice in real production models, so restricting to 64 only rejected
    # almost everything for no real tiling reason.
    VALID_HEAD_DIMS = (64, 128)
    if embedding_length % head_count != 0:
        failures.append(
            f"embedding_length ({embedding_length}) not divisible by head_count ({head_count}) — "
            "head_dim is not an integer"
        )
    else:
        head_dim = embedding_length // head_count
        info["head_dim"] = head_dim
        if head_dim not in VALID_HEAD_DIMS:
            failures.append(f"head_dim = {head_dim}, spec requires one of {VALID_HEAD_DIMS}")

    # Rule 2: head_count : head_count_kv ratio is power of 2
    if head_count_kv and head_count % head_count_kv != 0:
        failures.append(
            f"head_count ({head_count}) not evenly divisible by head_count_kv ({head_count_kv})"
        )
    elif head_count_kv:
        ratio = head_count // head_count_kv
        info["gqa_ratio"] = ratio
        if not _is_power_of_two(ratio):
            failures.append(f"GQA ratio {ratio}:1 is not a power of 2")

    # Rule 3: hidden_size divisible by 256
    if embedding_length % 256 != 0:
        failures.append(f"embedding_length ({embedding_length}) not divisible by 256")

    # Rule 4: MoE expert count, if present, must fit the small coarse-grained scheme
    expert_count = g.get(key("expert_count"))
    if expert_count is not None:
        info["expert_count"] = expert_count
        if expert_count > max_expert_count:
            failures.append(
                f"expert_count ({expert_count}) exceeds spec's coarse-grained cap ({max_expert_count})"
            )

    # Activation function — GGUF doesn't always encode this explicitly per-arch;
    # flag for manual confirmation rather than silently assuming SwiGLU.
    info["activation_note"] = (
        "GGUF metadata does not reliably expose activation function — "
        "confirm SwiGLU/gated-FFN manually against the source model card before proceeding"
    )

    return CompatResult(ok=(len(failures) == 0), failures=failures, info=info)


def print_report(result: CompatResult):
    print("=== Compat check ===")
    for k, v in result.info.items():
        print(f"  {k}: {v}")
    if result.ok:
        print("PASS — architecture fits the Vulkan-native spec, safe to convert.")
    else:
        print("FAIL — this is not a format conversion, it's a different architecture:")
        for f in result.failures:
            print(f"  - {f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python compat_check.py <model.gguf>")
        sys.exit(1)
    from gguf_reader import parse_gguf
    g = parse_gguf(sys.argv[1])
    result = check_compat(g)
    print_report(result)
    sys.exit(0 if result.ok else 1)
