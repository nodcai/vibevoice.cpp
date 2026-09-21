#!/usr/bin/env python3
"""Merge a DeepFilterNet3 gguf and/or the tokenizer into a VibeVoice gguf.

The realtime TTS model occasionally puts a musical artefact in front of an
utterance that starts with certain openers. The runtime removes it with a
DFN3 post-filter, and the filter's weights ride along inside the model file
so a deployment ships one gguf:

    python scripts/merge_dfn_gguf.py \\
        --model     models/vibevoice-realtime-0.5B-q8_0.gguf \\
        --dfn       models/deepfilternet3.gguf \\
        --tokenizer models/tokenizer.gguf \\
        --out       models/vibevoice-realtime-0.5b-dfn-q8_0.gguf

Every tensor and metadata key of --model is copied byte for byte (quantized
tensors included), then the `dfn.*` tensors and `dfn.*` keys of --dfn are
appended and `vibevoice.postfilter` is set to "deepfilternet3". The DFN
gguf is written by `scripts/convert_deepfilternet_to_gguf.py` (arch
"deepfilternet3", ~4 MB).

With --tokenizer, the `tokenizer.*` keys of a tokenizer gguf (written by
convert_tokenizer.py; it has no tensors) are copied in as well, and the
runtime then needs no separate tokenizer file.

The runtime detects the filter by the presence of `dfn.window` and the
tokenizer by `tokenizer.tokens`; a model without them behaves as before.
"""

from __future__ import annotations

import argparse
import sys

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except ImportError:
    sys.exit("pip install gguf")


def copy_kv(writer: GGUFWriter, reader: GGUFReader, keep) -> int:
    n = 0
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue                      # virtual header fields
        if not keep(name):
            continue
        vtype = field.types[0]
        value = field.contents()
        if vtype == GGUFValueType.ARRAY:
            writer.add_array(name, value)
        else:
            writer.add_key_value(name, value, vtype)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="vibevoice gguf (any quantization)")
    ap.add_argument("--dfn", default=None, help="deepfilternet3 gguf")
    ap.add_argument("--tokenizer", default=None, help="tokenizer gguf (convert_tokenizer.py output)")
    ap.add_argument("--out", required=True, help="merged gguf to write")
    args = ap.parse_args()

    if not args.dfn and not args.tokenizer:
        sys.exit("error: nothing to merge; give --dfn and/or --tokenizer")
    model = GGUFReader(args.model)
    dfn = GGUFReader(args.dfn) if args.dfn else None
    tok = GGUFReader(args.tokenizer) if args.tokenizer else None

    arch_field = model.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field else "vibevoice"
    if dfn:
        dfn_arch = dfn.fields.get("general.architecture")
        if dfn_arch and dfn_arch.contents() != "deepfilternet3":
            sys.exit(f"error: --dfn architecture is {dfn_arch.contents()!r}, expected 'deepfilternet3'")
        if any(t.name.startswith("dfn.") for t in model.tensors):
            sys.exit("error: --model already carries dfn.* tensors")
    if tok:
        if "tokenizer.tokens" not in tok.fields:
            sys.exit("error: --tokenizer has no tokenizer.tokens key")
        if "tokenizer.tokens" in model.fields:
            sys.exit("error: --model already carries a tokenizer")

    writer = GGUFWriter(args.out, arch)
    n_kv = copy_kv(writer, model, lambda k: k != "general.architecture")
    tensors = list(model.tensors)
    if dfn:
        n_kv += copy_kv(writer, dfn, lambda k: k.startswith("dfn."))
        writer.add_string("vibevoice.postfilter", "deepfilternet3")
        tensors += [t for t in dfn.tensors if t.name.startswith("dfn.")]
    if tok:
        n_kv += copy_kv(writer, tok, lambda k: k.startswith("tokenizer."))
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data)
    writer.close()

    n_dfn = sum(1 for t in dfn.tensors if t.name.startswith("dfn.")) if dfn else 0
    print(f"wrote {args.out}: {len(model.tensors)} model tensors + {n_dfn} dfn tensors, {n_kv} kv"
          f"{', tokenizer embedded' if tok else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
