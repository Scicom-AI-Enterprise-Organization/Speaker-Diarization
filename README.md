# SATS — Speaker-Attributed, Time-Stamped Transcription
## Part I. Paper summary
> Link: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize

SATS aims to transcribe what is said and to precisely determine the timing of each speaker.

## 1. Motivation
| **Approach**           | **What it does**                                                                                                                           | **Limitation**                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DiarizationLM** [20] | LLM post-processes ASR+diarization outputs to fix speaker permutations | Non end-to-end — inherits cascade errors  |
| **Sortformer** [14]    | Joint model trained with a permutation-invariant **Sort Loss**                                                                          | Training is two-stage (train diarizer, freeze it, then train ASR) → not truly end-to-end    |
| **SpeakerLM** [21]     | Single MLLM with speaker-aware modeling                            | Limited to short audio (~50–90s) and ≤4 speakers; no explicit timestamps                      |
| **JEDIS-LLM** [17]     | Trains on ≤20s clips but streams over long audio via a **Speaker Prompt Cache** | Still chunk-wise; needs extra cache/alignment machinery for global consistency |

---

## 2. Architecture
MOSS Transcribe Diarize couples an audio encoder with a projection module that maps multi-speaker acoustic embeddings into the feature space of a pretrained text LLM.

| Component         | Specification                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Audio encoder** | **Whisper-Medium encoder** — 80 Mel bins, *d*<sub>model</sub> = 1024, 24 transformer layers, 16 attention heads (~307 M)                                                              |
| **Projector**     | **VQAdaptor 2-layer MLP** with **4× temporal merge** (~5 M)                                                                                                                                 |
| **Text backbone** | **Qwen3-0.6B** — hidden size = 1024, 28 transformer layers, 16 attention heads, 8 KV heads (GQA), vocabulary size = 151,936, max position embeddings = 131,072 (128K context) (~596 M) |

_Note:_ "0.9B" ≈ Whisper-Medium encoder (~0.3B) + VQAdaptor + Qwen3-0.6B.

```
log-mel input_features → WhisperEncoder
    → 4× time merge  (B, T, 1024) → (B, T/4, 4096)   # audio_merge_size = 4
    → VQAdaptor      (4096 → 1024)                    # Linear→SiLU→Linear→LayerNorm
    → Qwen3-0.6B     → [start][Sxx] text [end] ...
```

- **Timestamps as text** — emitted as literal tokens (plain BPE digits `[Code]`), so timestamping is ordinary next-token prediction; generalizes to hour-scale audio.
- **Serialized Output Training (SOT)** [10] — a conversation is one flat stream: `[start][Sxx] words [end] …`.
- **Speaker labels are relative** — `[S01]` = "voice #1 in this file," not a named identity.
- **Loss** — inferred next-token cross-entropy over the serialized target; **not stated**.

Main limitation is a **sim-to-real gap** and **limited real eval coverage**.

---

## 3. Data
- Real Data
  - AISHELL-4 [7] — Mandarin meeting-room recordings, with both far-field (overlapping) and near-field mics. They "use the averaged channel of the far-field signals." (Far-field = room mic, hard; near-field = close mic, clean.)
  - They curated Podcast and Movies sets (used as test sets here).
- Simulated Data

---

## 4. Evaluation
Table 1. Test sets
| **Dataset** | **Avg duration** | **# speakers** | **Character** |
|--------------|------------------|----------------|---------------|
| AISHELL-4 Test | ~2290s (≈38 min) | 5–7 | Long real meetings (Mandarin) |
| Podcast | ~2659s (≈44 min) | 2–11 | Long multi-guest YouTube interviews |
| Movies | ~11.5s | 1–6 | Short, overlap-rich film/TV clips; Chinese+English, also Korean/Japanese/Cantonese; professionally annotated |

Podcast and Movies will be open-sourced on HuggingFace.

---

## 5. Results

Table 2. Headline Results
| **Dataset** | **Metric** | **Best baseline** | **MOSS** |
|--------------|------------|-------------------|----------|
| AISHELL-4 | cpCER | 24.99 (VibeVoice) | 15.83 |
| AISHELL-4 | Δcp | 3.59 | 0.99 |
| Podcast | cpCER | 10.23 (Gemini 2.5) | 7.37 |
| Movies | cpCER | 14.73 (Gemini 3) | 12.76 |
| Alimeeting | cpCER | 29.33 (VibeVoice) | 22.17 |

> *Note:* Compared against Doubao, ElevenLabs Scribe v1, GPT-4o Transcribe, Gemini 2.5 Pro, Gemini 3 Pro, VibeVoice (MOSS). GPT-4o couldn't ingest the long files at all, and Gemini 3 Pro kept breaking the required output format on long audio — a "nominal capability vs. deployable capability" gap.

Long-context end-to-end modeling keeps speakers consistent where cascaded/short-context systems drift.

---

## Part 2. Malaysian adaptation
## 6. Adapting to Malaysian speech

### 6.1 Single-speaker pool from existing corpora `[Proposed]`
Assemble a Malaysian single-speaker pool from data can access:
- **Mesolitica / Malaya-Speech** — large Malaysian ASR incl. code-switch (mostly single-speaker → ideal pool).
- **Common Voice** (`ms`, `ta`, `yue`, `zh`), **FLEURS** (`ms`) — read speech.
Keep Malay, Manglish, Mandarin, Cantonese, Hokkien, Tamil, and regional accents in the pool so mixtures match the real distribution. **Do not language-filter** — Manglish switches language mid-sentence and the model must learn that.

### 6.2 Synthetic multi-speaker `[Paper]`
Run the paper's mixer **unchanged** on our pool to produce labeled multi-speaker conversations:
- Draw **2–12 speakers**, one utterance each; cut into word-runs (log-normal weights).
- One timeline, Gaussian gaps, speaker alternation, **overlap ≤80%** of shorter segment.
- Snap boundaries to low-energy points + 50 ms cross-fades.
- Augment with noise/reverb; **SNR ~ U(0,15) dB** (Malaysian room impulse responses if available).
Every mixture ships with ground-truth `[start][Sxx] text [end]`.

### 6.3 Existing *labeled* multi-speaker for real anchor + eval `[Proposed]`
- **Malaysian Parliament / Hansard** — multi-speaker, named turns, timing → real diarization anchor + a natural eval slice (formal register).
- **IMDA NSC Parts 3–6** — conversational, multi-speaker, code-switch (Singaporean ≈ but ≠ Malaysian).
Use these as (a) a small real fine-tuning anchor to reduce sim-to-real gap, and (b) the evaluation set. A **modest professional re-check** of a few hours for the gold eval is the only manual labeling — far cheaper than annotating a full corpus.

---

## 7. Datasets we can access

| Resource | Role here | Note |
| --- | --- | --- |
| **Mesolitica / Malaya-Speech** | Simulation pool (primary) | Large MY ASR + code-switch; mostly single-speaker |
| **Malaysian Parliament / Hansard** | Real anchor + eval | Multi-speaker, named turns, timed; formal register |
| **IMDA NSC Parts 3–6** | Real anchor + eval | Conversational, code-switch; Singaporean |
| **Common Voice** (`ms`,`ta`,`yue`,`zh`) | Simulation pool | Read speech |
| **FLEURS** (`ms`) | Simulation pool | Read speech, small |

---

## 8. Implementation plan (lightweight, no collection)

1. **Baseline.** Run released 0.9B zero-shot on a Parliament/IMDA slice; report cpCER/Δcp — tells us where fine-tuning is actually needed. `[Proposed]`
2. **Build the mixer + scorer.** Implement simulation and the A.2 normalizer; validate the scorer by reproducing an AISHELL-4 number.
3. **Generate SUARA-Sim** from the pool.
4. **Fine-tune (LoRA).** Freeze Whisper encoder, LoRA on Qwen3; short-context first, then RoPE-extend. Train on Sim + small real anchor.
5. **Evaluate** and iterate on the real:sim ratio.

**Ablations worth running `[Proposed]`:** real:sim ratio · encoder frozen vs tuned · 4× vs 2× time-merge · with/without hotwords.

---

## 9. Metrics & evaluation

- **CER** `[Paper]` — character edit distance, text only (ASR quality). *For Malay/English also report **WER** (space-delimited).*
- **cpCER** `[Paper]` — concatenated **minimum-permutation** CER: tries all speaker-label permutations (Hungarian matching), then scores. The whole-system number.
  - *Example:* Ref `[S01] hello there [S02] good morning`; Pred (labels swapped) `[S02] hello there [S01] good morning` → cpCER = 0 after matching.
- **Δcp = cpCER − CER** `[Paper]` — isolates diarization penalty. A **negative Δcp** (−2.69 on Alimeeting) is an artifact of how CER linearizes overlap: grouping by speaker aligns better than the time-ordered concatenation — attribution is effectively "free," not that diarization improved transcription.
- **Add (paper omits these) `[Proposed]`:** **DER** (collar 0.25), **boundary-F1**, **timestamp onset/offset error** — the "time-stamped" claim is otherwise unmeasured.
- **Normalizer:** replicate Appendix A.2 exactly (strip `\s*\(.*?\)`, `<.*?>`, non-speaker `\[(?!S\d+\]).*?\]`) or numbers aren't comparable.

### Evaluation coverage `[Proposed]`

We evaluate across **four regimes** so results generalize beyond any single speaking style. Two exist already; two mirror the paper's Podcast/Movies archetypes — sourced legally (not by scraping, unlike the paper's YouTube/film sets).

| Eval set | Regime it covers | Source | Annotation | Legal note |
| --- | --- | --- | --- | --- |
| **Parliament** | Long, **formal**, turn-clean | Malaysian Hansard debates | Existing named turns + light gold re-check | Public record — verify reuse terms |
| **IMDA-conv** | **Conversational**, code-switch | IMDA NSC Parts 3–4 | Existing labels | IMDA agreement; Singaporean ≠ MY |
| **Movies** | **Short, dense-overlap** — rapid turn-taking and genuine interruptions | Small slice of real overlapping Malaysian audio — licensed local drama/TV | Manual professional pass — small but required; must capture overlap (<ovl>) + gold timestamps. Auto-labels not acceptable for an overlap eval | Licensed or owned audio only — no scraping films/TV (copyrighted). If no licensed source, this regime stays a coverage gap, flagged as a limitation |
| **Podcast** | Long, **conversational**, Manglish | Small **CC-licensed / owned** MY podcast slice | Small professional pass | No YouTube scraping; licensed or owned only |
| **Simulated** | **controlled overlap** | **Simulated** from our pool (§6.2) | Auto gold (free) | Clean — our own audio |

Rationale: Parliament + IMDA alone miss dense overlap and conversational Manglish. **Movies-style is free** (the simulator already emits short overlap-rich clips with gold labels); only **Podcast-style** needs a small, targeted, legally-sourced + lightly-annotated slice. This closes the coverage gap without a collection program.

### Results scaffold (fill as we run)

Compact view (headline joint metrics; full CER/WER/cpCER/Δcp/DER reported per set separately).

| System | Parliament cpWER↓ / DER↓ | IMDA-conv cpWER↓ / DER↓ | Movies-style cpWER↓ / DER↓ | Podcast-style cpWER↓ / DER↓ |
| --- | --- | --- | --- | --- |
| Cascade: Whisper-large-v3 + Pyannote 3.1 | — | — | — | — |
| MOSS 0.9B (zero-shot) | — | — | — | — |
| **MOSS 0.9B (LoRA on Sim + anchor)** | — | — | — | — |

> Read **per column** (per regime) — cpWER/DER are not comparable across regimes (overlap density and register differ). Small dialect/Podcast slices → report variance / CIs.

### Reference — MOSS 0.9B reported `[Paper]` (Mandarin/English; not directly comparable)

| Dataset | CER↓ | cpCER↓ | Δcp↓ |
| --- | --- | --- | --- |
| AISHELL-4 | 14.84 | 15.83 | 0.99 |
| Alimeeting | 24.86 | 22.17 | −2.69 |
| Podcast | 5.97 | 7.37 | 1.40 |
| Movies | 6.36 | 12.76 | 6.40 |

---

## 10. I/O format `[Paper]`

- **Prompt (default, Chinese):** *"请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳。/ Transcribe the audio; each segment starts with a start timestamp and speaker ID, then the speech content, and ends with an end timestamp."*
- **Hotwords (free gains):** append `热词提示：Putrajaya, KWSP, Petronas, …` — biases toward Malaysian names/orgs, no retraining.
- **Output:** `[start_time][Sxx] text [end_time]`, plus optional `<emotion> <event> <ovl> (overlap) <ins>` tags (stripped at scoring).

---

## 11. Risks & limitations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| **Sim-to-real gap** | Synthetic mixtures lack real prosody/turn dynamics | Small real anchor (Parliament/IMDA); ablate real:sim ratio |
| **Eval coverage** | Four regimes help, but the **Podcast-style** Manglish slice is small and **Movies-style is synthetic** (not real overlap) | Report per-regime variance/CIs; treat Podcast-style as indicative until enlarged; keep the full-collection track as fallback |
| **Existing-corpus licensing** | Reuse/redistribution limits | Verify each license; keep training vs release separate |
| **Whisper-medium encoder ceiling** | Weaker on Tamil/Hokkien | Ablate large-v3 encoder swap |
| **Unreported training recipe** | Reproduction uncertainty | Reverse-engineer curriculum; log everything |
| **No timestamp metric in paper** | No baseline to compare against | We define DER + boundary-F1 + onset error |

**What this approach does *not* give us:** a professionally-annotated, demographically-controlled Malaysian benchmark covering all dialects and Manglish conversational speech. If evaluation coverage proves insufficient, the full data-collection track is the fallback.

---

# Shortlist table

| Paper                                                                                                              | Why it matters                                                             | Adaptation strategy to Malay                                                                                      | Data needs                                                        | Priority    |
| ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- | ----------- |
| [Speaker Diarization for Low-Resource Languages Through Wav2vec Fine-Tuning](https://arxiv.org/abs/2504.18582)                                   | Directly tests low-resource adaptation with limited labels and multilingual transfer. | Fine-tune a strong self-supervised encoder on Malay or Malay-adjacent speech, then diarize on your target domain. | Small labeled Malay set plus more unlabeled Malay audio.          | High        |
| [Toward Bridging Gaps in Low-Resourced Languages Using Speaker Diarization](https://doaj.org/article/e64f8ab000cc4216ba3445f4b35933d3)                                     | Explicitly frames Sarawak Malay as a low-resource diarization problem.                | Use it as a template for dataset design and evaluation framing for Malay / regional Malay variants.               | Conversational Malay speech with speaker turns; transcripts help. | High        |
| [Improving Speaker Diarization for Low-Resourced Languages Using Self-Training / Pseudo-Labeling](https://ieeexplore.ieee.org/iel7/10336886/10336974/10337314.pdf)    | Practical when labels are scarce, which is common in enterprise Malay data.           | Bootstrap with pseudo-labels from a baseline system, then iterate on cleaner retraining.                          | Unlabeled Malay audio plus a smaller clean seed set.              | High        |
| [From Modular to End-to-End Speaker Diarization](https://arxiv.org/abs/2407.08752)                                                               | Good overview of the shift from pipelines to EEND-style models.                       | Decide whether to keep a modular stack or to move toward end-to-end training.                             | No special data requirement; helps design choice.                 | High        |
| [End-to-End Neural Diarization](https://ui.adsabs.harvard.edu/abs/2020arXiv200302966F/abstract)                                                                       | Foundational for overlap-aware diarization.                                           | If your Malay audio has interruptions and crosstalk, this is the core modeling direction to study.                | Multi-speaker recordings with diarization labels.                 | High        |
| [End-to-End Speaker Diarization Conditioned on Speech Activity](https://arxiv.org/abs/2310.08696)                                      | Improves EEND behavior in real speech settings.                                       | Useful if you want stronger segmentation before diarization, especially in noisy enterprise audio.                | Labeled diarization data, preferably conversational.              | Medium-High |
| [End-to-End Online Speaker Diarization with Target Speaker Tracking and Overlap Detection](arxiv.org/abs/2106.04078)                                           | Best if you need streaming or near-real-time diarization.                         | Adapt for live contact-center or conferencing use cases.                                                          | Streaming-friendly diarization data.                              | Medium-High |
| [EEND-SS: Joint End-to-End Neural Speaker Diarization and Speech Separation](https://ieeexplore.ieee.org/document/10022924/)                         | Helps with overlapping speech and unknown speaker counts.                             | Consider if overlap is a major pain point in Malay call recordings.                                               | More demanding training data and compute.                         | Medium      |
| [Pushing the Limits of End-to-End Diarization](https://www.isca-archive.org/interspeech_2025/broughton25_interspeech.pdf)                                                          | Shows the current ceiling of modern EEND methods.                                     | Use as a benchmark for what “good” looks like on modern public datasets.                                          | Benchmark-style evaluation data.                                  | Medium      |
| [SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition](https://arxiv.org/abs/2508.06372)                                         | Relevant if you want speaker labels and transcripts together.                         | Useful direction if one system is needed for “who spoke when and what.”                                        | Large-scale data and stronger compute.                            | Medium      |
| [pyannote.audio: neural building blocks for speaker diarization](https://arxiv.org/abs/1911.01255)                                               | Very practical baseline and toolkit for rapid prototyping.                            | Strong starting point for Malay baseline experiments before more ambitious fine-tuning.                           | Can work with modest labeled data.                                | High        |
| [A Review of Speaker Diarization: Recent Advances with Deep Learning](https://arxiv.org/abs/2101.09624)                                          | Fast orientation to the field and its model families.                                 | Use to justify why you pick pipeline vs end-to-end vs hybrid.                                                     | None.                                                             | Medium      |
| [An Experimental Review of Speaker Diarization Methods…](https://dl.acm.org/doi/10.1016/j.csl.2023.101534)                                                       | Useful for engineering tradeoffs, compute, and robustness.                            | Helpful for deciding whether to favor EEND, VBx, or hybrid approaches in production.                              | Comparative benchmark data.                                       | Medium      |
| [Speaker Diarization: A Comprehensive Survey of Clustering-Based and End-to-End Approaches](https://www.internationaljournalssrg.org/IJECE/2026/Volume13-Issue1/IJECE-V13I1P117.pdf) | Broad survey of recent work.                                                          | Good secondary reference to keep your literature review current.                                                  | None.                                                             | Medium      |
