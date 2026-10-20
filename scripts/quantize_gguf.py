#!/usr/bin/env python3
"""Quantize a vibevoice gguf.

Walks the source gguf and rewrites it with a new dtype for the heavy LM
matmul tensors (attention projs + FFN + lm_head) and, on request, the
token embedding. Everything else (conv1d kernels, norms, biases,
layer-scale gammas, scalars) is preserved as-is so the rest of the
orchestrator keeps working without code changes.

Why selective: ggml_mul_mat handles quantized weights natively (the
Q*_K, Q8_0 row formats are designed for it). But our conv1d wrapper
casts kernels to fp16 inline (`ggml_cast(kernel, GGML_TYPE_F16)`) — it
doesn't dequantize on the fly — so quantizing those would produce
wrong results.

Tensor-name allowlist for quantization (regex):

  (lm|tlm).blk.*.attn_(q|k|v|o).weight
  (lm|tlm).blk.*.ffn_(gate|up|down).weight
  lm_head.weight
  lm.tok_embd.weight          (only with --embed-type q8_0, realtime-0.5b)

K-quants and row width
----------------------
Q4_K / Q5_K / Q6_K pack 256 elements per super-block, so a tensor can
only take a K-quant when its row width (ne[0], the input dimension of
the matmul) is a multiple of 256. The Qwen2.5-0.5B stack behind the
realtime model has hidden size 896 = 3.5 x 256, so every attention
q/k/v/o projection, ffn_gate, ffn_up and the token embedding are NOT
K-quantizable; only ffn_down (4864 wide) is. Rather than silently
dropping those rows to Q4_0 (which costs ~6 points of rare-word
pronunciation accuracy on the realtime model), such tensors take the
block-32 type given by --fallback (default q8_0) and the per-group
summary printed at the end says so.

Pure-Python gguf-py can encode Q4_0 / Q5_0 / Q8_0 but not the K-quants.
Asking for a K-quant here fails before anything is written and points
you at build/bin/vibevoice-quantize, which links libggml and takes the
same flags (--type / --fallback-type / --embed-type ...).

Example (realtime TTS, LM matmuls at Q8_0, embedding kept at source dtype):

  scripts/quantize_gguf.py \\
      --src models/vibevoice-realtime-0.5B.gguf \\
      --out models/vibevoice-realtime-0.5B-q8_0.gguf \\
      --type q8_0

After quantizing, you should still pass the closed-loop test:

  build/tests/test_closed_loop  # see tests/test_closed_loop.cpp
"""

import argparse
import re
import sys
from pathlib import Path

import gguf
import numpy as np
from gguf import GGMLQuantizationType as QT
from gguf.constants import GGML_QUANT_SIZES

QUANT_MAP = {
    "q4_0": QT.Q4_0,
    "q5_0": QT.Q5_0,
    "q8_0": QT.Q8_0,
    "q4_k": QT.Q4_K,
    "q5_k": QT.Q5_K,
    "q6_k": QT.Q6_K,
    "f16":  QT.F16,
}
K_QUANTS = {QT.Q4_K, QT.Q5_K, QT.Q6_K}
FALLBACK_CHOICES = ("q8_0", "q5_0", "q4_0")

# Tensor families, in match order. Everything that matches none of these
# passes through at its source dtype. The patterns mirror our gguf naming
# convention (see scripts/convert_vibevoice_to_gguf.py).
GROUPS = [
    ("lm attn q/k/v/o", re.compile(r"^(lm|tlm)\.blk\.\d+\.attn_[qkvo]\.weight$")),
    ("lm ffn_gate/up",  re.compile(r"^(lm|tlm)\.blk\.\d+\.ffn_(gate|up)\.weight$")),
    ("lm ffn_down",     re.compile(r"^(lm|tlm)\.blk\.\d+\.ffn_down\.weight$")),
    ("lm_head",         re.compile(r"^lm_head\.weight$")),
    ("lm.tok_embd",     re.compile(r"^lm\.tok_embd\.weight$")),
]
EMBED_GROUP = "lm.tok_embd"
# Only the realtime TTS loop dequantizes embedding rows on lookup; the 1.5B
# and ASR paths read the table as F32/F16 and would misread a quantized one.
EMBED_QUANT_VARIANTS = {"realtime-0.5b"}


def group_of(name: str):
    for label, pat in GROUPS:
        if pat.match(name):
            return label
    return None


def block_size(qt: QT) -> int:
    return GGML_QUANT_SIZES[qt][0]


def can_encode(qt: QT) -> bool:
    """gguf-py implements the encoder for some types only; probe it."""
    if qt in (QT.F16, QT.F32):
        return True
    try:
        gguf.quants.quantize(np.zeros((1, block_size(qt)), dtype=np.float32), qt)
        return True
    except NotImplementedError:
        return False


def plan_tensor(name: str, ne0: int, src_type: QT, args, variant: str):
    """Return (target type or None for pass-through, note)."""
    group = group_of(name)
    if group is None:
        return None, ""
    if group == EMBED_GROUP:
        if args.keep_embed or args.embed_type == "keep":
            return None, "kept at source dtype (--embed-type keep)"
        qt = QUANT_MAP[args.embed_type]
        if ne0 % block_size(qt) != 0:
            return None, f"kept: row {ne0} not a multiple of {block_size(qt)}"
        return qt, "--embed-type"
    if group == "lm_head" and args.keep_embed:
        return None, "kept at source dtype (--keep-embed)"
    target = QUANT_MAP[args.type]
    if target in (QT.F16, QT.F32):
        return target, ""
    if ne0 % block_size(target) == 0:
        return target, ""
    if target in K_QUANTS:
        fb = QUANT_MAP[args.fallback]
        if ne0 % block_size(fb) == 0:
            return fb, f"fallback from {target.name}: row {ne0} % {block_size(target)} != 0"
        return None, f"kept: row {ne0} fits neither {target.name} nor {fb.name}"
    return None, f"kept: row {ne0} not a multiple of {block_size(target)}"


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    gv = gguf.GGUFValueType
    adders = {
        gv.STRING:  writer.add_string,
        gv.UINT8:   lambda k, v: writer.add_uint8(k, int(v)),
        gv.INT8:    lambda k, v: writer.add_int8(k, int(v)),
        gv.UINT16:  lambda k, v: writer.add_uint16(k, int(v)),
        gv.INT16:   lambda k, v: writer.add_int16(k, int(v)),
        gv.UINT32:  lambda k, v: writer.add_uint32(k, int(v)),
        gv.INT32:   lambda k, v: writer.add_int32(k, int(v)),
        gv.UINT64:  lambda k, v: writer.add_uint64(k, int(v)),
        gv.INT64:   lambda k, v: writer.add_int64(k, int(v)),
        gv.FLOAT32: lambda k, v: writer.add_float32(k, float(v)),
        gv.FLOAT64: lambda k, v: writer.add_float64(k, float(v)),
        gv.BOOL:    lambda k, v: writer.add_bool(k, bool(v)),
        gv.ARRAY:   lambda k, v: writer.add_array(k, list(v)),
    }
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue  # written by GGUFWriter
        try:
            value = f.contents()
        except Exception:
            continue
        if value is None or not f.types:
            continue
        add = adders.get(f.types[0])
        if add is None:
            continue
        try:
            add(f.name, value)
        except Exception as e:
            print(f"warning: could not copy KV {f.name} ({f.types[0]}): {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src",  required=True, type=Path,
                    help="source gguf (any dtype)")
    ap.add_argument("--out",  required=True, type=Path,
                    help="destination gguf (will be overwritten)")
    ap.add_argument("--type", default="q8_0", choices=sorted(QUANT_MAP),
                    help="quantization for LM matmul weights (default: q8_0). "
                         "K-quants need build/bin/vibevoice-quantize.")
    ap.add_argument("--fallback", default="q8_0", choices=FALLBACK_CHOICES,
                    help="block-32 type for tensors whose row width is not a "
                         "multiple of 256 when --type is a K-quant (default: "
                         "q8_0; q4_0 costs ~6 points of rare-word accuracy)")
    ap.add_argument("--embed-type", default="keep", choices=("keep", "q8_0"),
                    help="lm.tok_embd.weight: keep the source dtype (default) or "
                         "store it as q8_0 (realtime-0.5b only). Nothing below "
                         "q8_0 is offered: rare-token rows are exactly what "
                         "suffers from a coarse embedding.")
    ap.add_argument("--keep-embed", action="store_true",
                    help="keep lm.tok_embd and lm_head at the source dtype "
                         "(implies --embed-type keep; some ASR setups prefer this)")
    ap.add_argument("--verbose", action="store_true",
                    help="list every rewritten tensor, not just the per-group summary")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr); return 1

    reader = gguf.GGUFReader(str(args.src))
    arch = "vibevoice"
    variant = ""
    for f in reader.fields.values():
        try:
            if f.name == "general.architecture": arch = f.contents()
            elif f.name == "vibevoice.variant":  variant = f.contents()
        except Exception:
            pass

    if args.embed_type != "keep" and not args.keep_embed and variant not in EMBED_QUANT_VARIANTS:
        print(f"error: --embed-type {args.embed_type} is only supported for variant "
              f"{sorted(EMBED_QUANT_VARIANTS)} (this file: {variant or 'unknown'}); "
              "the 1.5B and ASR runtimes read the embedding as F32/F16", file=sys.stderr)
        return 1

    # ---- plan, and refuse before writing if gguf-py cannot encode a type ---
    plans = []
    for t in reader.tensors:
        ne0 = int(t.shape[0])
        target, note = plan_tensor(t.name, ne0, t.tensor_type, args, variant)
        plans.append((t, target, note))
    needed = {target for _, target, _ in plans if target is not None}
    missing = sorted(qt.name for qt in needed if not can_encode(qt))
    if missing:
        print(f"error: gguf-py cannot encode {', '.join(missing)} in pure Python. "
              "Use build/bin/vibevoice-quantize (links libggml, same tensor "
              "selection, --fallback-type for the 896-wide rows) for K-quants.",
              file=sys.stderr)
        return 1

    writer = gguf.GGUFWriter(str(args.out), arch=arch)
    copy_metadata(reader, writer)

    # ---- rewrite tensors ----------------------------------------------------
    rows = []  # (name, group, src type, out type, bytes in, bytes out, note)
    for t, target, note in plans:
        src_type = t.tensor_type
        group = group_of(t.name) or "other"
        if target is None or target == src_type:
            # Pass through as-is. Source stored as bytes (raw_dtype set);
            # writer recovers shape from byte-shape via quant_shape_from_byte_shape.
            writer.add_tensor(t.name, np.asarray(t.data), raw_dtype=src_type)
            rows.append((t.name, group, src_type, src_type, t.data.nbytes, t.data.nbytes, note))
            continue
        # gguf-py stores tensor data row-major in the file; t.data is a 1-D
        # ndarray view. Reshape to logical shape (note: t.shape is gguf
        # convention, fastest-varying first → reverse for numpy).
        np_shape = list(reversed([int(d) for d in t.shape]))
        arr = gguf.quants.dequantize(t.data, src_type).astype(np.float32).reshape(np_shape)
        qbytes = gguf.quants.quantize(arr, target)
        writer.add_tensor(t.name, qbytes, raw_dtype=target)
        rows.append((t.name, group, src_type, target, t.data.nbytes, qbytes.nbytes, note))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # ---- summary ------------------------------------------------------------
    if args.verbose:
        for name, group, st, ot, bi, bo, note in rows:
            if st != ot or note:
                print(f"  {name:48s} {st.name:>5s} -> {ot.name:<5s} {bo/1e6:8.1f} MB  {note}")
    agg = {}
    for name, group, st, ot, bi, bo, note in rows:
        key = (group, ot.name, note.split(":")[0])
        c, b = agg.get(key, (0, 0))
        agg[key] = (c + 1, b + bo)
    order = [g for g, _ in GROUPS] + ["other"]
    print(f"{'tensor group':18s} {'type':6s} {'tensors':>7s} {'MB':>8s}  note")
    for group in order:
        for (g, ty, note), (c, b) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            if g == group:
                print(f"{g:18s} {ty:6s} {c:7d} {b/1e6:8.1f}  {note}")
    bytes_in  = sum(r[4] for r in rows)
    bytes_out = sum(r[5] for r in rows)
    n_q = sum(1 for r in rows if r[2] != r[3])
    n_fb = sum(1 for r in rows if r[6].startswith("fallback"))
    print(f"wrote {args.out}: rewrote {n_q} tensors ({n_fb} via --fallback {args.fallback}), "
          f"{bytes_in/1e6:.0f} MB -> {bytes_out/1e6:.0f} MB "
          f"({100 * (1 - bytes_out / bytes_in):.1f}% smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
