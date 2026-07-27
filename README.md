# SATS — Speaker-Attributed, Time-Stamped Transcription
Link: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
SATS aims to transcribe what is said and to precisely determine the timing of each speaker.

## Motivation
| **Approach**           | **What it does**                                                                                                                           | **Limitation**                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DiarizationLM** [20] | LLM post-processes ASR+diarization outputs to fix speaker permutations | Non end-to-end — inherits cascade errors  |
| **Sortformer** [14]    | Joint model trained with a permutation-invariant **Sort Loss**                                                                          | Training is two-stage (train diarizer, freeze it, then train ASR) → not truly end-to-end    |
| **SpeakerLM** [21]     | Single MLLM with speaker-aware modeling                            | Limited to short audio (~50–90s) and ≤4 speakers; no explicit timestamps                      |
| **JEDIS-LLM** [17]     | Trains on ≤20s clips but streams over long audio via a **Speaker Prompt Cache** | Still chunk-wise; needs extra cache/alignment machinery for global consistency |

## Architecture
MOSS Transcribe Diarize couples an audio encoder with a projection module that maps multi-speaker acoustic embeddings into the feature space of a pretrained text LLM.
Audio-encoder (log-Mel spectrogram front-end) → projection → Autoregressive SpeechLLM (0.9B parameters)
They write the timestamp as literal text tokens. This avoids binding temporal encoding to absolute positional indices, which become sparse and ineffective over long durations, and enables accurate timestamp generation over hour-scale audio.
Serialized Output Training (SOT). They represent a multi-speaker conversation as a single flat token stream with speaker-change tokens ([S01], [S02]) inline: [start][speaker] words [end] [start][speaker] words [end] ...
Loss function: most likely standard next-token cross-entropy over the serialized target sequence.
Speaker labels are relative, not global.

## Data
- Real Data
  - AISHELL-4 [7] — Mandarin meeting-room recordings, with both far-field (overlapping) and near-field mics. They "use the averaged channel of the far-field signals." (Far-field = room mic, hard; near-field = close mic, clean.)
  - They curated Podcast and Movies sets (used as test sets here).
- Simulated Data:
  - Draw 2–12 distinct speakers per synthetic dialogue; pick one utterance each from a single-speaker pool.
  - Cut each utterance into contiguous word runs: sample a segment count and log-normal weights (so segment lengths look natural — many short, few long).
  - Lay all segments on one timeline with Gaussian-distributed inter-segment gaps, enforcing speaker alternation but allowing overlaps capped at 80% of the shorter segment.
  - Snap segment boundaries to nearby low-energy points and apply 50 ms cross-fades (so cuts don't click/pop).
  - Augment with real-world noise and reverberation; sample SNR uniformly from 0–15 dB. (lower = noisier, 0 dB means noise as loud as speech)

## Evaluation
Table 1. Test sets
| **Dataset** | **Avg duration** | **# speakers** | **Character** |
|--------------|------------------|----------------|---------------|
| AISHELL-4 Test | ~2290s (≈38 min) | 5–7 | Long real meetings (Mandarin) |
| Podcast | ~2659s (≈44 min) | 2–11 | Long multi-guest YouTube interviews |
| Movies | ~11.5s | 1–6 | Short, overlap-rich film/TV clips; Chinese+English, also Korean/Japanese/Cantonese; professionally annotated |

Podcast and Movies are being open-sourced on HuggingFace.

## Metrics
CER (Character Error Rate) — edit distance between predicted and reference text only, ignoring speakers. Measures the ASR quality. (Character-level because Chinese has no word spaces; for Malay also report WER, word error rate, since Malay is space-delimited.) Lower is better.
cpCER (concatenated minimum-permutation CER) — evaluates ASR and diarization jointly. Because speaker labels are relative, it tries all permutations of predicted speaker labels and picks the one giving the lowest error (an optimal assignment / Hungarian-style matching), then computes CER on the speaker-attributed transcript. This is the headline "whole-system" number.
Δcp = cpCER − CER — the extra error introduced purely by getting speakers wrong. This is their cleanest measure of diarization quality: if Δcp is near zero, transcription is right and attributed to the right speaker. A negative Δcp (they get −2.69 on Alimeeting) is a quirk of the permutation matching — it essentially means speaker attribution added no penalty.

Model	AISHELL‑4	Alimeeting	Podcast	Movies
CER↓	cpCER↓	Δcp↓	CER↓	cpCER↓	Δcp↓	CER↓	cpCER↓	Δcp↓	CER↓	cpCER↓	Δcp↓
Doubao	18.18	27.86	9.68	25.25	37.57	12.31	7.93	10.54	2.61	9.94	30.88	20.94
ElevenLabs	19.58	37.95	18.36	25.70	36.69	10.99	8.50	11.34	2.85	11.49	17.85	6.37
GPT-4o	-	-	-	-	-	-	-	-	-	14.37	23.67	9.31
Gemini 2.5 Pro	42.70	53.42	10.72	27.43	41.64	14.21	7.38	10.23	2.85	15.46	24.15	8.69
Gemini 3 Pro	22.75	27.43	4.68	26.75	32.84	6.09	-	-	-	8.62	14.73	6.11
VIBEVOICE ASR	21.40	24.99	3.59	27.40	29.33	1.93	27.94	48.30	20.36	14.59	42.54	27.94
MOSS Transcribe Diarize 0.9B	14.84	15.83	0.99	24.86	22.17	-2.69	5.97	7.37	1.40	6.36	12.76	6.40
MOSS Transcribe Diarize Pro	13.78	14.02	0.24	18.22	13.94	-4.27	4.46	6.97	2.51	5.86	11.78	5.92

