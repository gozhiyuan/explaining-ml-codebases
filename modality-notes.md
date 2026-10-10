# Modality notes

ML pipelines handle very different data. Read the section for each modality
the repo touches, in the data, loading, and serving docs. Each section says
what the docs must show on top of the stage playbooks: the raw format, the
decode step, the tensor layout, the shapes at each step, and the traps.

For every modality, the running example uses one real item (one row, one
sentence, one image, one 2-second clip, one episode). The output examples
show that item at each step, as the rules in "Show the data, not only the
code" in [writing-modes.md](writing-modes.md) require.

## Contents

- Tabular and structured
- Text and tokens
- Images
- Video
- Audio and speech
- Time series and sensor streams
- Robotics and embodied episodes
- Point clouds, 3D, and geospatial
- Graphs
- Multimodal pairs
- Storage and dataset formats

---

## Tabular and structured

| Show | Detail |
|------|--------|
| Grain | what one row is (one entity-time bucket, one transaction), and the key columns |
| Schema at each step | columns added, dropped, retyped, with dtypes |
| Joins | keys, join type, row-count change, and duplicates the join can create |
| Time | event time versus processing time, the cut-off that prevents leakage |
| Encoding | categorical vocab and its reserved ids, numeric scaling statistics and where they are fitted, missing-value flags |
| Sample | 2-5 rows of the running example after each step |

Traps: target leakage through joins or aggregates, train/serve feature
skew, timezone mixing, row explosion after a join.

## Text and tokens

| Show | Detail |
|------|--------|
| Raw | encoding, normalization (Unicode, lowercasing), cleaning rules |
| Tokenizer | type (BPE, WordPiece, SentencePiece, byte-level), vocab size, special tokens and their ids, chat template |
| Ids | the running example's text, its token strings, and its ids |
| Length handling | truncation side, padding side and id, packing of several documents in one sequence, the attention mask |
| Batch | `input_ids [B, T]`, `attention_mask [B, T]`, labels with ignore index (`-100`) |

Traps: a different tokenizer at train and serve, special tokens added twice,
padding on the wrong side for a decoder, labels not masked on the prompt.

## Images

| Show | Detail |
|------|--------|
| Raw | file format, color mode (RGB, BGR, grayscale), bit depth, typical size |
| Decode | library (PIL, OpenCV, torchvision.io, DALI), where it runs (worker process, GPU) |
| Transforms | the list in order, which are random, the resize and crop sizes, normalization mean and std |
| Layout | `[C, H, W]` per image, `[B, C, H, W]` per batch (or NHWC), dtype and value range after each step |
| Labels | class ids, boxes (format: xyxy, xywh, normalized or pixels), masks, keypoints, and how transforms move them |
| Sample | one image's shape and value range after each transform |

Traps: BGR versus RGB, boxes not transformed with the image, a normalization
that differs between training and serving, EXIF rotation.

## Video

| Show | Detail |
|------|--------|
| Raw | container and codec (MP4/H.264, WebM/VP9), fps, resolution, duration |
| Decode | library (PyAV, decord, torchvision.io, ffmpeg, NVDEC), CPU or GPU, seek strategy |
| Sampling | frames per clip, stride, clip start policy, how timestamps map to frame indices |
| Layout | `[T, C, H, W]` per clip, `[B, T, C, H, W]` per batch (or `[B, C, T, H, W]`), and which one the model expects |
| Audio track | kept or dropped, and how it aligns with frames |
| Sample | one clip's frame indices, timestamps, and tensor shape |

Traps: variable frame rate, decoding cost dominating the pipeline,
off-by-one frame alignment with labels, a different sampling policy at
inference.

## Audio and speech

| Show | Detail |
|------|--------|
| Raw | format (WAV, FLAC, Opus), sample rate, channels, bit depth |
| Resample and mix | target sample rate, mono mixdown, gain normalization |
| Features | waveform `[B, samples]`, or STFT, mel spectrogram, or MFCC: window, hop, n_fft, n_mels, giving `[B, n_mels, frames]`. Frames = samples / hop |
| Chunking | segment length, overlap, voice activity detection (VAD), and how chunks are stitched back |
| Labels | transcripts and their tokenizer, alignments, speaker ids |
| Sample | 2 seconds at 16 kHz is 32,000 samples, which is about 200 frames at a 10 ms hop. Show the real numbers |

Traps: sample-rate mismatch, train on clean audio and serve on a codec-
compressed stream, chunk boundaries that cut words.

## Time series and sensor streams

| Show | Detail |
|------|--------|
| Index | timestamps, frequency, timezone, irregular or missing samples |
| Windows | lookback, horizon, stride, and how windows are cut from one series |
| Alignment | how several sensors with different rates are resampled or joined |
| Features | lags, rolling statistics, calendar features, scaling per series or global |
| Layout | `[B, T, F]` and which dims are time |

Traps: future data in a window, resampling that leaks, a backtest split
that does not match production. See archetype N in
[ml-archetypes.md](ml-archetypes.md).

## Robotics and embodied episodes

| Show | Detail |
|------|--------|
| Episode | what one episode is, its length, success flag, task or language instruction |
| Streams | each observation stream (camera images, depth, joint positions, gripper state, force/torque, proprioception) and the action stream, each with its shape, rate (Hz), and units |
| Time sync | the master clock, how streams with different rates are aligned (nearest, interpolation), the allowed tolerance, and dropped frames |
| Action space | joint positions, end-effector deltas, or velocities, the frame (base, world, tool), normalization statistics, and the action chunk size |
| Format | rosbag or MCAP, HDF5, LeRobot or RLDS datasets, Zarr. The on-disk layout of one episode |
| Policy input and output | observation history length, `obs` dict of tensors, `action [B, chunk, action_dim]` |
| Sim versus real | simulator (MuJoCo, Isaac, PyBullet), domain randomization, and what differs on the real robot |
| Sample | one episode, one timestep: every stream's value and its timestamp |

Traps: misaligned timestamps between cameras and joint states, action and
observation frames that differ, normalization statistics from a different
dataset, a control rate at deploy that differs from training.

## Point clouds, 3D, and geospatial

| Show | Detail |
|------|--------|
| Raw | format (PLY, LAS, PCD, meshes, GeoTIFF), coordinate frame, units |
| Layout | `[N, 3 + features]` per cloud, voxel or pillar grids, how variable N is batched |
| Transforms | sensor-to-world transforms, cropping, voxel size, augmentation |
| Geospatial | CRS and reprojection, tiling, resolution |

Traps: mixed coordinate frames, a different voxel size at inference, CRS
mismatches.

## Graphs

| Show | Detail |
|------|--------|
| Structure | nodes, edges, directed or not, node and edge features, edge index `[2, E]` |
| Batching | how several graphs are merged (block-diagonal adjacency, `batch` vector), or neighbor sampling for one large graph |
| Splits | node, edge, or graph splits, and leakage through edges |

## Multimodal pairs

| Show | Detail |
|------|--------|
| Pairing | how items from each modality are matched (caption to image, audio to transcript, instruction to episode) |
| Per-modality processing | point to the modality sections above |
| Fusion | where the modalities meet: early concat, cross-attention, a projector into the language model's embedding space, special placeholder tokens and how many tokens each image or clip becomes |
| Sample | one pair, both sides, and the fused input shape |

## Storage and dataset formats

Whatever the modality, show how the data sits on disk and how the loader
reads it.

| Format | Show |
|--------|------|
| Parquet, Delta, Iceberg | partition columns, file size, row groups, schema |
| WebDataset, TFRecord, Mosaic MDS | shard naming, samples per shard, the keys inside one sample |
| HF `datasets` | the dataset name, splits, features, streaming or cached |
| HDF5, Zarr | group layout, chunking, the arrays for one item |
| rosbag, MCAP | topics, message types, rates |
| Images or audio as loose files | the directory layout and the index or manifest file |

Say where each read happens (driver, worker process, GPU) and what is cached.
