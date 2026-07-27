# SATS — Speaker-Attributed, Time-Stamped Transcription
> Link: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize

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

| Component         | Specification                                                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Audio encoder** | **Whisper-Medium encoder** — 80 Mel bins, *d*<sub>model</sub> = 1024, 24 transformer layers, 16 attention heads                                                               |
| **Projector**     | **VQAdaptor MLP** with **4× temporal merge**                                                                                                                                  |
| **Text backbone** | **Qwen3-0.6B** — hidden size = 1024, 28 transformer layers, 16 attention heads, 8 KV heads (GQA), vocabulary size = 151,936, max position embeddings = 131,072 (128K context) |

_Note:_ "0.9B" ≈ Whisper-Medium encoder (~0.3B) + Qwen3-0.6B.
> Audio-encoder (log-Mel spectrogram front-end) → projection (VQAdaptor) → Autoregressive SpeechLLM (0.9B parameters)
> log-mel input_features -> HF WhisperEncoder
         -> 4x time merge  (B, T, 1024) -> (B, T/4, 4096)
         -> VQAdaptor       (4096 -> 1024)

Two parts:

(a) 4× time merge (time_merge): the Whisper encoder outputs one 1024-dim vector per audio frame. They concatenate every 4 consecutive frames into one 4096-dim vector, cutting the sequence length by 4×: (B, T, 1024) → (B, T/4, 4096). This is set by audio_merge_size = 4.

(b) VQAdaptor — the actual projector, a small 2-layer MLP:
> nn.Sequential(
    nn.Linear(4096, 1024, bias=True),   # adaptor_input_dim (4096) -> LM hidden (1024)
    nn.SiLU(),                          # smooth activation function
    nn.Linear(1024, 1024, bias=True),
    nn.LayerNorm(1024),
)

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

Example: 
Ref - [S01] hello there     [S02] good morning; 
Prediction (correct words, but labels swapped) - [S02] hello there     [S01] good morning

Δcp = cpCER − CER — the extra error introduced purely by getting speakers wrong. This is their cleanest measure of diarization quality: if Δcp is near zero, transcription is right and attributed to the right speaker. A negative Δcp (they get −2.69 on Alimeeting) is a quirk of the permutation matching — it essentially means speaker attribution added no penalty.

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

## I/O format
Prompt (default, Chinese):
>"请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，并在段末标注结束时间戳。"
>("Transcribe the audio; each segment starts with a start timestamp and speaker ID, then the speech content, and ends with an end timestamp.")

Hotword prompting — append a short hint to bias toward domain terms (names, jargon). Purely prompt-based, no retraining.
> 热词提示：hotword1, hotword2, hotword3

Output: 
> [start_time][Sxx] transcribed speech [end_time], and it can also emit <emotion>, <event>, <ovl> (overlap), <ins> tags.

Evaluation normalization (Appendix A.2) — before scoring they strip parentheticals \s*\(.*?\), angle-tags <.*?>, and non-speaker brackets \[(?!S\d+\]).*?\], keeping only [Sxx] tags and text. This normalizer must be replicated or CER numbers won't be comparable.

## Adapting to Malaysian speech
1. Assemble a single-speaker Malay/Malaysian pool. Sources: Mesolitica/Malaya-Speech Malaysian corpora — those single-speaker clips are the "utterance pool" the simulator needs. Include Malay, Manglish (English–Malay code-switch), Mandarin, Tamil, and dialects (Kelantanese, Sabah/Sarawak) to match the real distribution.
2. Run the §5.2 simulator unchanged to build multi-speaker conversations with ground-truth [Sxx] + timestamps. Keep 2–12 speakers, ≤80% overlap, 0–15 dB SNR, Malaysian room impulse responses if you have them.
3. Preserve code-switching inside utterances — don't language-filter the pool; Manglish turns often switch language mid-sentence, and the model should learn that.
4. Metrics: report WER in addition to CER (Malay is space-delimited; Tamil/Chinese portions justify CER), plus cpCER/Δcp using the paper's permutation matching and normalizer.
5. Tokenizer check: confirm the backbone LLM's tokenizer covers Malay diacritics and Tamil/Chinese scripts well; a poor tokenizer inflates CER on non-Latin segments.
6. Cheapest first experiment: before any training, run the released 0.9B checkpoint on Malaysian meeting audio and measure cpCER/Δcp. That baseline tells you how much its "50+ languages" already transfers, and where fine-tuning is actually needed.
7. Hotwords for free gains: Malaysian names, place names (e.g., Putrajaya, Kementerian), and org acronyms via the 热词提示 mechanism — no training required.

## Malay Datasets - options closest to AISHELL-4
| Resource | Match to AISHELL-4 | Notes |
|----------|--------------------|-------|
| **IMDA National Speech Corpus (Singapore), Parts 3–6** | **Best** | Conversational, multi-speaker, includes Malay and heavy code-switching (Manglish/Singlish). Parts 3–4 contain two-person conversations with timestamps, making them suitable for real diarization. |
| **Malaysian Parliament / Hansard recordings** | **Good** | Debate recordings with multiple speakers, named speaker turns, and timing. Mesolitica has released Malaysian Parliament speech datasets, making this a strong AISHELL-4 analogue for diarization training. |
| **Mesolitica / Malaya-Speech corpora** | **Good for the pool** | Large Malaysian ASR datasets, including code-switching. Mostly single-speaker recordings, making them ideal as the utterance pool for the synthetic simulator (§6), rather than as ready-made meeting data. |
| **FLEURS (ms), Common Voice Malay** | **Weak** | Single-speaker read speech. Useful as a simulation pool, but not as diarization evaluation data. |

Use IMDA Part 3 / parliament data as real multi-speaker train+test, and use Malaya-Speech/Common Voice/FLEURS as the single-speaker pool that you feed into the synthetic mixer. 
