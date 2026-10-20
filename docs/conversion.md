# Converting VibeVoice models to GGUF

`vibevoice.cpp` ships three Python tools that turn upstream Microsoft VibeVoice
artifacts into the GGUF format the C++ runtime consumes.

## Pipeline overview

```
  Hugging Face
  ┌─────────────────────────────┐
  │ Qwen/Qwen2.5-0.5B            │── tokenizer.json ─► convert_tokenizer.py ─► tokenizer.gguf
  │ microsoft/VibeVoice-…        │── safetensors    ─► convert_vibevoice_to_gguf.py ─► …gguf
  │ microsoft/VibeVoice          │── voices/*.pt    ─► convert_voice_to_gguf.py ─► voice.gguf
  └─────────────────────────────┘
                                                                 │
                                              (optional)          │
                                                                 ▼
                                                        quantize_gguf.py ─► …-q8_0.gguf
                                                        vibevoice-quantize ─► …-mixed.gguf (K-quants)
```

## `scripts/convert_tokenizer.py`

Wraps a Hugging Face tokenizer (`tokenizer.json`) into a GGUF that
`src/tokenizer.cpp` can load directly. The tokenizer is required by every
TTS / ASR run; you only need to convert it once and reuse it across model
checkpoints.

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir models/qwen2.5
python scripts/convert_tokenizer.py \
    --src models/qwen2.5/tokenizer.json \
    --out models/tokenizer.gguf
```

VibeVoice ASR re-uses three Qwen2.5 vision-token IDs to mark speech regions
in the prompt:

| ID     | Token                  | Role        |
| ------ | ---------------------- | ----------- |
| 151646 | `<|object_ref_start|>` | speech_start |
| 151647 | `<|object_ref_end|>`   | speech_end   |
| 151648 | `<|box_start|>`        | speech_pad   |

These are surfaced through the standard tokenizer (no patching required).

## `scripts/convert_vibevoice_to_gguf.py`

Walks a VibeVoice checkpoint directory (`config.json` + sharded
`model*.safetensors`), maps PyTorch tensor names to our GGUF naming
convention via an ordered set of regex rewrites, and writes a single GGUF
with all required metadata.

```bash
hf download microsoft/VibeVoice-Realtime-0.5B --local-dir models/vv-rt-0.5b
python scripts/convert_vibevoice_to_gguf.py \
    --src models/vv-rt-0.5b \
    --out models/vibevoice-realtime-0.5B.gguf
```

Variants are auto-detected from the `architectures` field in `config.json`:

| Variant         | Source repo                                                                       | Notes                                                                |
| --------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `realtime-0.5b` | `microsoft/VibeVoice-Realtime-0.5B`                                               | TTS, `vibevoice-cli tts --voice <voice.gguf>`                        |
| `1.5b`          | `microsoft/VibeVoice-1.5B`                                                        | TTS with runtime voice cloning, `vibevoice-cli tts --ref-audio <wav>`|
| `asr-7b`        | `microsoft/VibeVoice-ASR`                                                         | ASR, `vibevoice-cli asr`                                             |

### GGUF tensor naming

Both variants share a common naming scheme that mirrors the upstream
PyTorch module hierarchy. The notable patterns:

| HF / safetensors name (excerpt)                                                              | gguf tensor name                                |
| -------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `model.language_model.embed_tokens.weight`                                                   | `lm.tok_embd.weight`                            |
| `model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.{weight,bias}`                     | `lm.blk.{i}.attn_{q,k,v,o}.{weight,bias}`       |
| `model.language_model.layers.{i}.{input,post_attention}_layernorm.weight`                    | `lm.blk.{i}.{attn,ffn}_norm.weight`             |
| `model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight`                             | `lm.blk.{i}.ffn_{gate,up,down}.weight`          |
| `model.tts_language_model.layers.{j}.…`  (realtime only — split LM)                          | `tlm.blk.{j}.…`                                 |
| `model.tts_language_model.norm.weight`   (realtime only)                                     | `tlm.output_norm.weight`                        |
| `model.tts_input_types.weight`            (realtime only)                                    | `tts.input_types.weight`                        |
| `model.acoustic_tokenizer.encoder.…`                                                          | `at.enc.…`                                      |
| `model.acoustic_tokenizer.decoder.…`     (TTS variants — realtime + 1.5b)                    | `at.dec.…`                                      |
| `model.semantic_tokenizer.…`             (ASR + 1.5b)                                        | `st.…`                                          |
| `model.acoustic_connector.{linear1,norm,linear2}.…`                                          | `ac.{fc1,norm,fc2}.…`                           |
| `model.semantic_connector.…`             (ASR + 1.5b)                                        | `sc.…`                                          |
| `model.prediction_head.…`                (TTS variants — realtime + 1.5b)                    | `dh.…`                                          |
| `model.tts_eos_classifier.…`             (realtime only)                                     | `eos.…`                                         |
| `lm_head.weight`                          (ASR + 1.5b; for 1.5b synthesised from tied embed) | `lm_head.weight`                                |

Pass `--strict` to fail the conversion if any source tensor key is left
unmapped — useful when validating against a new upstream release.

### Encoder ratio quirk

The acoustic / semantic encoders apply downsample strides in **reversed**
order vs the decoder, mirroring upstream
`vibevoice/modular/modular_vibevoice_tokenizer.py:713`:

```python
self.ratios = list(reversed(config.ratios))   # encoder
self.ratios = config.ratios                   # decoder
```

That means `down_1` has stride `ratios[N-1]`, not `ratios[0]`. The C++
loader handles this in `src/acoustic_tokenizer.cpp::load_encoder`.

## `scripts/convert_voice_to_gguf.py`

VibeVoice TTS conditions on a precomputed KV cache for a known speaker
(the "voice prompt"). Upstream ships these as `.pt` files under
`microsoft/VibeVoice/demo/voices/streaming_model/`. The converter reads a
voice `.pt` and writes a small GGUF (~5–10 MB) containing per-layer K/V
tensors for both LM stacks plus the negative caches used by classifier-free
guidance.

```bash
curl -sLO https://github.com/microsoft/VibeVoice/raw/main/demo/voices/streaming_model/en-Carter_man.pt
python scripts/convert_voice_to_gguf.py \
    --src en-Carter_man.pt \
    --out models/voice-en-Carter_man.gguf
```

Voice GGUFs are model-agnostic in shape but specific to the realtime
TTS architecture (`realtime-0.5b`); they are not used by ASR.

## `scripts/quantize_gguf.py` and `vibevoice-quantize`

Optional. Both take any vibevoice GGUF and produce a quantized variant by
selectively quantizing only the LM matmul weights — attention q/k/v/o,
FFN gate/up/down, and `lm_head` — to a target dtype.  Everything else
(conv1d kernels, RMSNorm scales, biases, layer-scale gammas, small
scalars) passes through unchanged. The token embedding is kept at its
source dtype unless `--embed-type q8_0` is given (realtime-0.5b only: the
1.5B and ASR runtimes read the table as F32/F16).

Why selective: `ggml_mul_mat` handles quantized weights natively, but
the conv1d wrapper in `src/conv1d.cpp` casts kernels to fp16 inline
(`ggml_cast(kernel, GGML_TYPE_F16)`) — it does not dequantize on the
fly, so quantizing those would silently corrupt the convolution outputs.

```bash
python scripts/quantize_gguf.py \
    --src models/vibevoice-asr.gguf \
    --out models/vibevoice-asr-q8_0.gguf \
    --type q8_0
```

The Python script encodes what `gguf-py` can in pure Python: `q8_0`,
`q5_0`, `q4_0`, `f16`. For K-quants use `build/bin/vibevoice-quantize`
(built with the examples; links libggml), which takes the same tensor
selection plus per-family overrides (`--attn-type`, `--ffn-type`,
`--lm-head-type`, `--decoder-ffn-type`, `--head-type`, `--embed-type`).
Asking the Python script for a K-quant fails before anything is written
and says so.

### K-quants need a row width that is a multiple of 256

`Q4_K` / `Q5_K` / `Q6_K` pack 256 elements per super-block, so a tensor
only takes a K-quant when `ne[0]` (its input width) is divisible by 256.
The realtime model's LM is a Qwen2.5-0.5B with hidden size 896 = 3.5 x
256, so:

| tensor family (realtime-0.5b)       | `ne[0]` | K-quantizable? |
| ----------------------------------- | ------: | :------------: |
| `{lm,tlm}.blk.*.attn_{q,k,v,o}.weight` | 896   | no (96 tensors)  |
| `{lm,tlm}.blk.*.ffn_{gate,up}.weight`  | 896   | no (48 tensors)  |
| `{lm,tlm}.blk.*.ffn_down.weight`       | 4864  | **yes** (24 tensors) |
| `lm.tok_embd.weight`                | 896     | no             |
| `at.dec.*.ffn_linear{1,2}`          | 32–8192 | mostly; the 12 narrowest fall back |

Tensors that cannot take the requested K-quant get the block-32 type from
`--fallback` (Python) / `--fallback-type` (C++), **default `q8_0`**; the
older default followed llama.cpp's ladder (q4_k → q5_0) and the published
realtime q4_k file was built with an explicit `--fallback-type q4_0
--embed-type q4_0`. That file's LM is therefore 157 tensors of Q4_0 plus
24 of Q4_K, and Q4_0 LM weights cost about 6 points of rare-word
pronunciation accuracy (73.7 % vs 79.7 % rare-word slots hit with Q8_0;
F16 gives 78.3 %, so Q8_0 loses nothing). Everyday-word WER does not show
it. Both tools print a per-family table at the end so a fallback is
visible:

```
tensor group       type   tensors       MB  note
lm attn q/k/v/o    q8_0        96     46.8  fallback from q4_K
lm ffn_gate/up     q8_0        48    222.3  fallback from q4_K
lm ffn_down        q4_K        24     58.8
lm.tok_embd        q8_0         1    144.7
decoder ffn        q6_K        40    247.5
decoder ffn        q8_0        12      0.4  fallback from q6_K
diffusion head     q8_0        22     44.7
```

### Recommended recipe for the realtime TTS model

LM attention / gate / up and the token embedding at q8_0, `ffn_down` at
q4_k, decoder FFN at q6_k, diffusion head at q8_0:

```bash
./build/bin/vibevoice-quantize --src models/vibevoice-realtime-0.5B.gguf \
    --out models/vibevoice-realtime-0.5b-mixed.gguf \
    --type q4_k --fallback-type q8_0 --decoder-ffn-type q6_k \
    --embed-type q8_0 --head-type q8_0
```

Size by family, computed from the tensor shapes of the published file
(the same model reproduces the published q4_k file's 661 MB of tensor data
exactly; add ~6 MB for header, tokenizer and the DeepFilterNet3 weights):

| family                          | type  | MB    |
| ------------------------------- | ----- | ----: |
| LM attention q/k/v/o            | q8_0  | 47    |
| LM `ffn_gate` / `ffn_up`        | q8_0  | 222   |
| LM `ffn_down`                   | q4_k  | 59    |
| token embedding                 | q8_0  | 145   |
| decoder FFN                     | q6_k  | 248   |
| diffusion head                  | q8_0  | 45    |
| everything else (convs, norms)  | f16   | 91    |
| **total**                       |       | **~856 MB tensors, ~862 MB file** |

Against 667 MB for the published q4_k file and 1.1 GB for q8_0
throughout. Swapping the LM fallback for `q5_0` would give ~767 MB, but
Q5_0 has not been measured on the rare-word benchmark. The mixed file has
not been rebuilt yet: it needs an F16 conversion of the HF checkpoint via
`convert_vibevoice_to_gguf.py` as the source.

Available dtypes:

| `--type` | Bits/elem (matmul) | Python script | `vibevoice-quantize` |
| -------- | ------------------ | :-----------: | :------------------: |
| `q8_0`   | 8.5                | yes           | yes                  |
| `q5_0`   | 5.5                | yes           | yes                  |
| `q4_0`   | 4.5                | yes           | yes                  |
| `q6_k`   | 6.5 (rows % 256)   | no            | yes                  |
| `q5_k`   | 5.5 (rows % 256)   | no            | yes                  |
| `q4_k`   | 4.5 (rows % 256)   | no            | yes                  |
| `f16`    | 16                 | yes           | yes                  |

## Closed-loop verification

After conversion (and optional quantization), validate the bundle with:

```bash
cmake -B build -DVIBEVOICE_BUILD_TESTS=ON -DVIBEVOICE_TEST_LARGE=ON \
    && cmake --build build -j

VIBEVOICE_TTS_MODEL=$PWD/models/vibevoice-realtime-0.5B.gguf \
VIBEVOICE_VOICE=$PWD/models/voice-en-Carter_man.gguf \
VIBEVOICE_ASR_MODEL=$PWD/models/vibevoice-asr.gguf \
VIBEVOICE_TOKENIZER=$PWD/models/tokenizer.gguf \
VIBEVOICE_CLI=$PWD/build/bin/vibevoice-cli \
ctest --test-dir build -R test_closed_loop --output-on-failure
```

The test synthesizes a known phrase with TTS, transcribes it with ASR, and
asserts ≥80 % source-word recall in the recovered transcript. With our Q8_0
ggufs it consistently lands at 100 %.
