# SATS — Speaker-Attributed, Time-Stamped Transcription
Link: https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize
SATS aims to transcribe what is said and to precisely determine the timing of each speaker.

## Motivation
| **Approach**           | **What it does**                                                                                                                           | **Limitation**                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DiarizationLM** [20] | LLM post-processes ASR+diarization outputs to fix speaker permutations
| Non end-to-end — inherits cascade errors                                                                                                                           |
| **Sortformer** [14]    | Joint model trained with a permutation-invariant **Sort Loss**                                                                          | Training is two-stage (train diarizer, freeze it, then train ASR) → not truly end-to-end    |
| **SpeakerLM** [21]     | Single MLLM with speaker-aware modeling                            | Limited to short audio (~50–90s) and ≤4 speakers; no explicit timestamps                      |
| **JEDIS-LLM** [17]     | Trains on ≤20s clips but streams over long audio via a **Speaker Prompt Cache** | Still chunk-wise; needs extra cache/alignment machinery for global consistency |

## Architecture

