# Malaysian SATS — Project Status

Adapting **MOSS-Transcribe-Diarize 0.9B** for Malaysian multilingual, code-switched
Speaker-Attributed Time-Stamped Transcription (SATS), via data reuse + synthetic simulation.

## Pipeline

| # | Stage | Script | Status |
|---|-------|--------|--------|
| 0 | Paper study + architecture recovered from released code | — | ✅ done |
| 1 | Project scaffold + isolated env (`envs/sats`, `datasets<4`) on DSW box | — | ✅ done |
| 2 | Dataset inspection (schema / duration / SR / speaker / transcript) | `scripts/inspect*.py` | ✅ done |
| 3 | Build single-speaker **pool** → normalized 16 kHz mono + manifest | `scripts/build_pool.py` | ✅ done |
| 4 | **Simulate** labeled multi-speaker SATS mixtures from the pool | `scripts/simulate.py` | 🔄 in progress |
| 5 | Fetch multi-speaker sets (parliament / podcast / diarization) for **Real + Eval** | `scripts/fetch_multispeaker.py` | ⬜ todo (special access) |
| 6 | Speaker-disjoint **train/val/test splits** | `scripts/make_splits.py` | ✅ done |
| 7 | **Scorer**: CER/WER, cpCER/Δcp + DER/timestamp | `src/.../metrics.py` | ✅ done |
| 8 | **Zero-shot baseline** of MOSS 0.9B on Malaysian eval | `scripts/run_baseline.py` | ✅ done |
| 9 | **LoRA fine-tune** (freeze Whisper encoder) on sim + real anchor | `scripts/train_lora.py` | 🔄 in progress |
| 10 | **Evaluate** fine-tuned vs zero-shot vs cascade baseline | `scripts/evaluate.py` | ⬜ todo |

## Data storage

- Code: `projects/malaysian-sats/` · Data: `datasets/` · Weights: `checkpoints/` · Logs: `logs/`
- Pool output: `datasets/pool/<lang>/<source>/*.wav` + `datasets/pool/manifest.jsonl` (the index all later stages read).

## Dataset inventory — inspection results

### A. Single-speaker POOL (usable now)

| Dataset | Lang | SR | Avg dur | Speakers | Transcript | Verdict |
|---|---|---|---|---|---|---|
| `mesolitica/malaya-speech-malay-stt` | Malay | 16 kHz | — | per-clip | ✅ (`Y`) | **Pool ✅** (audio in `filename`) |
| `google/fleurs` (`ms_my`) | Malay | 16 kHz | 13.0 s | none | ✅ | **Pool ✅** |
| `Scicom-intl/CommonVoice22-Sidon-HQ` (`ta`) | Tamil | 48 kHz | 8.1 s | 35 / 300 | ✅ | **Pool ✅** (+ DNSMOS quality scores) |
| `Scicom-intl/CommonVoice22-Sidon-HQ` (`yue`) | Cantonese | 48 kHz | 5.8 s | 54 / 300 | ✅ | **Pool ✅** |
| `Scicom-intl/CommonVoice22-Sidon-HQ` (`zh-CN`) | Mandarin | 48 kHz | 6.4 s | 128 / 300 | ✅ | **Pool ✅** |
| `Scicom-intl/TTS-Clean44k` (`tamil`,`aishell3`) | Tamil / Mandarin | 48 kHz | ~6 s | none | ❌ no text | Weak (needs pseudo-label) |

*(Speakers = unique in a 300-clip sample, not totals.)*

### B. Multi-speaker (Real / Eval) — exist, need special access

| Dataset | Intended role | Blocker |
|---|---|---|
| `mesolitica/unsupervised-malay-youtube-speaker-diarization` | Real anchor (diarization) | Stored as **WebDataset tar** — needs a WebDataset reader |
| `malaysia-ai/malaysia-parliament-youtube-chunk-24k` | Real + formal eval | **Multi-part zip** — needs manual download + extract |
| `malaysia-ai/malaysian-podcast-youtube` | Podcast-style eval | Multi-part zip |
| `malaysia-ai/malay-conversational-speech-corpus-whisper-format` | Conversational pool/anchor | Broken schema (declares `audio`, parquet has `filename/Y/null`) |
| `malaysia-ai/tamil-youtube` | Tamil pool | Audio only, no transcript/speaker |

### C. Not Malaysian-relevant (deprioritized)

| Dataset | Note |
|---|---|
| `Scicom-intl/MLS-Sidon-HQ` | European languages only |
| `Scicom-intl/clean-speech-raw-sources` | DAPS/RAVDESS/BibleTTS — English + African |

## Key findings

- **Malay is covered** by `malaya-speech-malay-stt` (16 kHz, transcribed) + FLEURS — the Scicom-intl CV subset has **no Malay**.
- **Pool side is solved**; **multi-speaker Real/Eval** exists but is packaged in formats HF streaming can't open — a dedicated fetch/extract step (stage 5).
- All CV/TTS sources are **48 kHz** → resampled to 16 kHz in `build_pool.py`; FLEURS and malaya-speech are already 16 kHz.
- No gold multi-speaker labels yet → **test set still needs a small human-verified slice** (auto/pseudo labels can't be the eval ground truth).

## Immediate next step

Run `build_pool.py` at full size, then write `simulate.py` (pool → labeled multi-speaker mixtures, §3.2 recipe: 2–12 speakers, ≤80 % overlap, SNR 0–15 dB).

## Dataset Pool

The dataset pool has been successfully built.

| Language | Clips | Hours |
|----------|------:|------:|
| Malay (`ms`) | 5,654 | 13.07 |
| Mandarin (`zh`) | 3,000 | 4.38 |
| Cantonese (`yue`) | 3,000 | 3.64 |
| Tamil (`ta`) | 772 | 1.43 |
| **Total** | **12,426** | **22.52** |

### Notes

- 🎯 **Malay** is the strongest language in the pool (**13.07 hours**), making it well suited for the Malay-first pilot.
- ⚠️ **Tamil** is comparatively limited (**772 clips, 1.43 hours**). This is primarily due to the `ovrl ≥ 3.0` quality filter and the available dataset size. Future improvements could include lowering the quality threshold or augmenting the pool with `tamil-youtube` pseudo-labeled data.

### Next Step

Ran `simulate.py` to convert this clean speech pool into labeled multi-speaker **SATS** mixtures following the paper's §3.2 simulation recipe:

- **2–12 speakers** per mixture
- **≤80% overlap ratio**
- **SNR:** 0–15 dB
- Gold transcript labels in the format:
  > [start][S01] text [end]
  > [start][S02] text [end]
  
## Milestone Recap

Everything needed to generate and evaluate the Malaysian **SATS** benchmark is now built and validated using only the data already available.

- **Speech pool:** 12,426 clips (22.52 hours)
- **Simulation pipeline:** `sim_ms` (2,000 mixtures), `sim_mixed` (1,000 mixtures), with gold labels
- **Evaluation toolkit:** `cpCER`, `cpWER`, and `Δcp` metrics, fully self-tested

### Remaining Work

The remaining tasks focus on the model evaluation pipeline:

1. Build the real evaluation set.
2. Establish zero-shot baselines.
3. Fine-tune the target models.


### Data splits (speaker/video-disjoint)

| Split | real/diar (videos) | pool (clips / speakers) |
|-------|-------------------:|------------------------:|
| train | 56                 | 11,197 / 5,679          |
| val   | —                  | 1,229 / 624             |
| test  | **24**             | —                       |
| **grouping key** | video `id`  | `speaker`               |

*Test set = 24 held-out real multi-speaker videos, disjoint from the anchor. Pool has no
test (the test is real audio, never synthetic); real has no val (validation comes from the
pool + optional real). Leakage check: **clean** (no group shared across splits).*

### Results (test = real/diar split_test, 24 videos, same set for all rows)

| System | CER↓ | cpCER↓ | Δcp↓ | WER↓ | cpWER↓ | Δwer↓ |
|--------|-----:|-------:|-----:|-----:|-------:|------:|
| MOSS 0.9B — zero-shot            | 61.62 | 86.64 | 25.02 | 91.78 | 113.96 | 22.18 |
| MOSS 0.9B — fine-tuned (LoRA)    | —     | —     | —     | —     | —      | —     |
| Cascade — Whisper-v3 + Pyannote  | —     | —     | —     | —     | —      | —     |

*Zero-shot = untuned MOSS. Labels are weak (auto ASR + re-clustered speakers) and n=24 is
small, so treat absolute values as a diagnostic floor; the comparison across rows on this
fixed set is what matters. A hand-verified gold slice of these 24 videos gives the
trustworthy number later.*
