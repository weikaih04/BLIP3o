# MosaicML Streaming (MDS) setup — blip3o_trellis env

⚠️ `pip install mosaicml-streaming` (normal) DOWNGRADES `transformers 5.2.0 → 4.57.6`
(hard pin `transformers<5`), which BREAKS Qwen3.5 loading. The MDS *core* (MDSWriter +
StreamingDataset for local/remote shards) does NOT use transformers at runtime.

Install WITHOUT touching transformers:
    pip install --no-deps mosaicml-streaming zstd crc32c python-snappy cramjam catalogue
Verify:
    python -c "from streaming import StreamingDataset, MDSWriter; import transformers; print(transformers.__version__)"  # 5.2.0
(cloud backends boto3/azure/gcs/oci NOT installed — we read MDS from /fsx + cache to local NVMe, no S3.)
