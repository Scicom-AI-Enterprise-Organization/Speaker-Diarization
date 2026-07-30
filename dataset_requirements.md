# Dataset Requirements — Malaysian SATS for SciCom Deployment

Speaker-Attributed, Time-Stamped Transcription (SATS) — dataset spec to fine-tune
and evaluate **MOSS-Transcribe-Diarize 0.9B** for Malaysian multilingual speech.

> **Provenance:** `[Paper]` = MOSS technical report · `[Code]` = released config/weights · `[Proposed]` = our recommendation · `[Internal]` = SciCom/Mesolitica assets.
> **Hard constraint:** **reuse and modify existing datasets — no from-scratch collection** (no hiring speakers, no studio sessions). Where new audio is unavoidable it is a small, licensed/owned slice only.

---

## 0. Deployment context (drives every number below)

SciCom runs high-volume **contact-centre / customer-engagement** audio: mostly **2-party phone calls**, multilingual and code-switched, often **8 kHz narrowband telephony**. The dataset must therefore *match deployment acoustics*, not idealized studio speech. Key implications:
- **Channel:** include telephone-band audio; the model front-end wants **16 kHz** `[Code]`, so 8 kHz sources must be up-sampled (band-limited content stays band-limited — that realism is desirable).
- **Speaker count:** 2-party dominant (agent + customer); meetings are the multi-party stretch case.
- **Languages by business volume:** Malay, Malaysian English, Manglish, Mandarin first; Tamil/Cantonese/Hokkien second.

---

## 1. Data roles (from the MOSS pipeline)

| Role | What | Source strategy |
|---|---|---|
| **Pool** | single-speaker utterances | reuse existing corpora → feed simulator |
| **Sim** | synthetic multi-speaker mixtures | generate via §3.2 recipe (free gold labels) |
| **Real** | genuine multi-speaker audio | reuse existing *labeled* sets (Parliament/IMDA) + small owned call slice |
| **Eval** | held-out gold | small, double-annotated |

---

## 2. Minimum Viable Dataset (MVP)

Enough to fine-tune, beat the zero-shot baseline, and run a credible SciCom pilot. Built almost entirely from existing data.

| Requirement | MVP target | Reasoning |
|---|---|---|
| **Hours — Pool** | ~100 h single-speaker (existing corpora) | Feeds the simulator; 100 h across languages gives enough unique utterances for varied mixtures without new recording. |
| **Hours — Sim** | ~150–200 h generated | Cheap and free-labeled; the bulk of fine-tuning signal for speaker separation + timestamps. |
| **Hours — Real anchor** | ~20–30 h (Parliament/IMDA + small owned calls) | Minimum to pull the model off simulation artifacts (sim-to-real gap) without a collection program. |
| **Hours — Eval (gold)** | ~6–8 h, double-annotated | Small but *real* held-out set; enough for stable cpWER/DER on the priority languages. |
| **Speakers** | ~300–500 unique in Pool; ~50–100 in Real | Diarization generalization tracks **number of distinct voices**, not total hours — prioritize variety over duration. |
| **Languages** | Malay, Malaysian English, Manglish, Mandarin | Highest contact-centre volume → fastest deployment value. Tamil = stretch; Cantonese/Hokkien deferred to Ideal. |
| **Noise levels** | Clean + telephony + moderate noise; simulate **SNR ~ U(0–15 dB)** `[Paper]` and include a clean band | Must span real call conditions; 0 dB (noise = speech) is the hard tail that prevents overfitting to clean audio. |
| **Overlap ratio** | ~5–15% of speech time (2-party realistic); simulator cap ≤80% per turn `[Paper]` | Real calls have modest overlap (back-channels, interruptions); over-simulating overlap distorts the distribution vs deployment. |
| **Conversation types** | 2-party phone calls (primary); a little meeting/interview | Matches the deployment target; meetings add the multi-party stress case. |

**MVP explicitly excludes:** dialects (Cantonese/Hokkien), demographic balancing, and far-field meeting rigs — all deferred to Ideal to keep MVP fast and reuse-only.

---

## 3. Ideal Dataset

The production-grade version once the MVP proves value.

| Requirement | Ideal target | Reasoning |
|---|---|---|
| **Hours — Pool** | 500 h+ across all 7 languages | Broader coverage for robust simulation, incl. dialects. |
| **Hours — Real** | 150–200 h labeled multi-speaker, incl. owned calls | Shrinks the sim-to-real gap; covers real overlap the simulator can't fake. |
| **Hours — Eval** | 25–35 h, per-language + per-regime | Enough for per-language and per-accent breakdowns with confidence intervals. |
| **Diversity** | All 7 languages; regional accents (Kelantan/Sabah/Sarawak Malay; Penang vs Southern Hokkien); ~50/50 gender; ages 18–70 with 50+ over-sampled for dialects; telephony + mobile + far-field channels | Accent/channel/demographic mismatch is a top error source and a deployment risk; explicit coverage de-risks it. |
| **Annotation quality** | Double-annotated + adjudicated gold; overlap (`<ovl>`) + precise timestamps; inter-annotator agreement (cpCER/DER) reported; code-switch spans tagged | Enables trustworthy DER/timestamp metrics and a defensible benchmark; IAA sets a quality floor. |

---

## 4. Source datasets to reuse & how to modify each

| Resource | Role | Modification needed |
|---|---|---|
| **Mesolitica / Malaya-Speech** `[Internal]` | Pool (primary) | VAD-trim to clean utterances; resample 16 kHz mono; keep code-switch intact |
| **Common Voice** (`ms`,`ta`,`yue`,`zh`) | Pool | Filter to validated clips; trim silence; balance speakers |
| **FLEURS** (`ms`) | Pool | Small; use as-is after resample |
| **Malaysian Parliament / Hansard** `[Internal]` | Real anchor + eval | Align speaker turns → `[Sxx]`+timestamps; segment long sessions; light gold re-check |
| **IMDA NSC Parts 3–6** | Real anchor + eval | Map existing labels → SATS format; note Singaporean ≠ Malaysian |
| **SciCom owned calls** `[Internal]` | Real (deployment-matched) + eval | PDPA consent/redaction; small professional annotation; telephony channel = deployment-realistic |

---

## 5. Preprocessing — silence, resampling, channel

### 5.1 Should you remove long silence?

**Yes for the Pool, carefully for Real.**

- **Pool (single-speaker → simulator):** **trim aggressively.** The simulator re-places utterances on a new timeline and generates fresh labels, so tight, silence-free clips are strictly better (no wasted context, cleaner overlap construction). No label is harmed because labels are (re)generated downstream.
- **Real multi-speaker (train/eval with timestamps):** **do NOT blindly strip internal silence.** Every timestamp is ground truth; deleting a 2 s pause shifts everything after it and corrupts the diarization reference. Allowed:
  - Trim **leading/trailing** dead-air (adjust the first/last timestamps accordingly).
  - **Split** a session at very long internal gaps (e.g. >3–5 s hold/dead-air) into sub-sessions, preserving timing *within* each — instead of deleting.
  - If you must compact internal silence, re-derive **all** timestamps from the retained segments (VAD-based) and re-emit labels; treat it as a re-segmentation, not a trim.

### 5.2 Practical commands

**Resample to the model's format (always):**
```
ffmpeg -i in.wav -ar 16000 -ac 1 -sample_fmt s16 out.wav
```

**Trim silence for Pool clips (fast, ffmpeg):**
```
ffmpeg -i in.wav -af \
 silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB:\
stop_periods=-1:stop_silence=0.4:stop_threshold=-40dB \
 -ar 16000 -ac 1 pool_clip.wav
```

**VAD-based (preferred — also gives you timestamps for labels), Silero VAD:**
```python
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_ ) = utils
wav = read_audio('in.wav', sampling_rate=16000)
segs = get_speech_timestamps(wav, model, sampling_rate=16000)  # [{'start':.., 'end':..}]
# Pool: concatenate speech segs (silence dropped).
# Real: KEEP seg times → build [start][Sxx] labels; split at gaps > threshold instead of deleting.
```

> Rule of thumb: **VAD for the Pool is a cleaner; VAD for Real is a labeler.** Same tool, opposite intent — one removes silence, the other records where speech is so you can preserve timing.

### 5.3 Loudness / channel
- Normalize loudness (e.g. EBU R128 to −23 LUFS) so mixtures don't clip.
- Keep an **8 kHz→16 kHz telephony subset** so the model sees deployment-realistic band-limiting; don't "clean it up" out of existence.

---

## 6. Build procedure

1. **Assemble Pool** from existing single-speaker corpora → resample 16 kHz mono → VAD-trim silence → tag language/speaker.
2. **Generate Sim** with the §3.2 simulator (2–12 speakers, ≤80% overlap, SNR 0–15 dB); labels emitted automatically. Add a **2-party, low-overlap** preset to match contact-centre calls.
3. **Prepare Real anchor** from Parliament/IMDA (+ owned calls): convert existing labels to `[start][Sxx] text [end]`; split long sessions at long gaps.
4. **Carve Eval** — a small held-out, double-annotated gold set, **speaker-disjoint** from train.
5. **Verify** speaker-disjoint splits and Pool↔Eval non-overlap before any training.

---

## 7. Risks & caveats

| Risk | Mitigation |
|---|---|
| Base-language coverage unknown (no list) | Empirically probe zero-shot per language first (Phase 1) |
| No val split in MOSS to mirror | Define our own speaker-disjoint val from Real |
| Telephony 8 kHz vs 16 kHz front-end | Up-sample; keep a telephony subset for realism |
| Silence handling corrupts timestamps | Pool: trim freely · Real: preserve/split, never blind-strip |
| Sim overlap ≠ real call overlap | Add a 2-party low-overlap Sim preset; anchor with real calls |
| PDPA on owned calls | Consent + PII redaction before use; never ship raw |

---
*Reuse-only spec. MVP is built almost entirely from existing corpora + simulation; the only new audio is a small, licensed/owned, PDPA-cleared call slice for deployment-matched eval.*
