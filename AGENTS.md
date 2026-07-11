# AGENTS.md

Notes for running the "AICoverGen Mod" flow (`Hina_Mod_AICoverGen_colab.ipynb`) outside of
Google Colab, gathered while benchmarking on a real NVIDIA Tesla T4 (Turing, 16GB). These are
observations about existing behavior only — no code in this repo or in the underlying
`ardha27/AICoverGen-Mod` code repo needs to change for the notes below to be true.

## 1. The documented flow is Colab-only

`Hina_Mod_AICoverGen_colab.ipynb` is a Colab/Kaggle launcher notebook, not the inference code
itself. The notebook:

- Mounts Google Drive (`google.colab.drive.mount('/content/drive')`) and hardcodes
  `/content/drive/MyDrive/...` paths.
- Clones the actual inference code from a rot13-obfuscated URL
  (`codecs.decode(..., 'rot_13')` decodes to `https://github.com/ardha27/AICoverGen-Mod.git`
  for the default `Pitch_Change="12"` path, or `SociallyIneptWeeb/AICoverGen` for the
  alternate path).
- Assumes a Colab-preinstalled `torch` and strips version pins from `requirements.txt`
  before installing.

None of this runs unmodified on a non-Colab GPU box — "Run all" fails immediately outside
Colab. To run headlessly (e.g. on a bare GPU server), use the actual code repo's CLI directly:
`python src/main.py` (the same entry point `no_ui.py` wraps), after reconstructing the
environment by hand:

- `torch==2.0.1+cu118`
- `python3.10-dev` (needed to build `fairseq==0.12.2` from source — no cp310 wheel is
  published for it)
- `huggingface_hub==0.23.4` (newer `huggingface_hub` removed `HfFolder`, which `gradio==4.29`
  imports at top level)
- `setuptools<81` (`pyworld` still imports `pkg_resources`; still bundled in setuptools 81.x, removed starting `setuptools>=82`, so this pin stays one version ahead of the actual removal)
- `ffmpeg`, `sox`

## 2. The default run has no bundled RVC voice model

`download_models.py` only downloads the MDX vocal-separation model, `hubert_base.pt`, and
`rmvpe.pt`. It does **not** download an RVC voice-conversion model. `no_ui.py`'s default
`-dir Daemi` points at a folder that does not exist, so a default/first run fails with:

```
Exception: The folder .../rvc_models/Daemi does not exist.
```

An RVC voice model (`.pth` + `.index`) must be supplied manually under
`rvc_models/<name>/`, and `-dir <name>` must match that folder name.

## 3. MDX vocal separation silently falls back to CPU without extra CUDA libraries

MDX-Net vocal separation (`src/mdx.py`) requests `CUDAExecutionProvider` from `onnxruntime-gpu`,
but the `torch==2.0.1+cu118` wheel does not bundle `libcurand.so.10`, `libcufft.so.10`, or an
unversioned `libnvrtc.so`, which `onnxruntime-gpu`'s CUDA execution provider needs. Without
them, ONNX Runtime prints a warning and falls back to CPU for MDX separation only (the rest of
the pipeline — pitch extraction, RVC conversion — stays on GPU).

Measured on a Tesla T4:

- CPU fallback: ~28.7s per MDX separation pass, peak VRAM ~2.0 GB.
- After installing `nvidia-curand-cu11`, `nvidia-cufft-cu11`, `nvidia-cuda-nvrtc-cu11`, adding
  their `lib/` dirs (plus `torch/lib`) to `LD_LIBRARY_PATH`, and symlinking the versioned
  `libnvrtc.so.11.2` to an unversioned `libnvrtc.so` (cuDNN's `dlopen` looks for the
  unversioned name): CUDA execution provider loads, ~14.0s per pass — a **2.05x** speedup.
- Trade-off: peak VRAM rises from ~2.0 GB to ~13.8 GB (98% utilization on a 16GB T4) once MDX
  runs on GPU, leaving little headroom on 16GB cards.

## 4. fp16 is the correct dtype on T4 (no bf16 trap)

RVC/hubert/rmvpe run in fp16 (`is_half=True`) on T4, which is correct: `voice_change()` hardcodes
`Config('cuda:0', True)`, and `Config.device_config`'s fp32-downgrade branch checks for the
substring `"16"` in the GPU name string (targeting GTX 16xx cards). `torch.cuda.get_device_name(0)`
returns `"Tesla T4"`, which does not contain `"16"`, so the downgrade branch does not trigger and
fp16 is used as intended. `torch.cuda.is_bf16_supported()` is `False` on T4 (Turing), and the
code has no bf16 path, so there is no bf16 fallback trap here.

Note for GTX 16xx users specifically: that downgrade branch, when it does trigger, rewrites
`src/configs/{32k,40k,48k}.json` (`true` → `false`) directly on disk in the cloned repo, which
is a persistent side effect worth being aware of.
