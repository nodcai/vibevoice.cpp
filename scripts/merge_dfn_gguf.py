#!/usr/bin/env python3
"""Merge a DeepFilterNet3 gguf into a VibeVoice gguf.

The realtime TTS model occasionally puts a musical artefact in front of an
utterance that starts with certain openers. The runtime removes it with a
DFN3 post-filter, and the filter's weights ride along inside the model file
so a deployment ships one gguf:

    python scripts/merge_dfn_gguf.py \\
        --model models/vibevoice-realtime-0.5B-q8_0.gguf \\
        --dfn   models/deepfilternet3.gguf \\
        --out   models/vibevoice-realtime-0.5b-dfn-q8_0.gguf

Every tensor and metadata key of --model is copied byte for byte (quantized
tensors included), then the `dfn.*` tensors and `dfn.*` keys of --dfn are
appended and `vibevoice.postfilter` is set to "deepfilternet3". The DFN
gguf is written by `scripts/convert_deepfilternet_to_gguf.py` (arch
"deepfilternet3", ~4 MB).

The runtime detects the filter by the presence of `dfn.window`; a model
without it behaves as before.
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
    ap.add_argument("--dfn", required=True, help="deepfilternet3 gguf")
    ap.add_argument("--out", required=True, help="merged gguf to write")
    args = ap.parse_args()

    model = GGUFReader(args.model)
    dfn = GGUFReader(args.dfn)

    arch_field = model.fields.get("general.architecture")
    arch = arch_field.contents() if arch_field else "vibevoice"
    dfn_arch = dfn.fields.get("general.architecture")
    if dfn_arch and dfn_arch.contents() != "deepfilternet3":
        sys.exit(f"error: --dfn architecture is {dfn_arch.contents()!r}, expected 'deepfilternet3'")
    if any(t.name.startswith("dfn.") for t in model.tensors):
        sys.exit("error: --model already carries dfn.* tensors")

    writer = GGUFWriter(args.out, arch)
    n_kv = copy_kv(writer, model, lambda k: k != "general.architecture")
    n_kv += copy_kv(writer, dfn, lambda k: k.startswith("dfn."))
    writer.add_string("vibevoice.postfilter", "deepfilternet3")

    tensors = list(model.tensors) + [t for t in dfn.tensors if t.name.startswith("dfn.")]
    for t in tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in tensors:
        writer.write_tensor_data(t.data)
    writer.close()

    n_dfn = sum(1 for t in dfn.tensors if t.name.startswith("dfn."))
    print(f"wrote {args.out}: {len(model.tensors)} model tensors + {n_dfn} dfn tensors, {n_kv + 1} kv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
