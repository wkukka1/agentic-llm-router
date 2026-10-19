# Collection server for opt-in usage sharing

Gateways whose users opted in send an anonymous queue (routing features and a coarse embedding, **no
prompt text, ids or paths**) here. This service validates it, stores it, and `publish_kaggle` turns
it into versions of a Kaggle dataset. Kaggle itself cannot take anonymous writes, which is why the
gateways talk to this service and only this service holds the dataset credentials.

## Run it

```bash
pip install "agentic-llm-router[gateway]" kaggle   # fastapi + uvicorn come with the gateway extra
export INGEST_SHARD_DIR=/var/lib/aar-ingest/shards
uvicorn ingest.app:app --host 0.0.0.0 --port 8080   # put TLS in front: gateways accept https only
```

| Variable | Default | Meaning |
|---|---|---|
| `INGEST_SHARD_DIR` | `shards` | where `<UTC date>/<install id>.jsonl` are written |
| `INGEST_MAX_BODY_BYTES` | 1,000,000 | largest upload (also bounds the unzipped size, x20) |
| `INGEST_MAX_RECORDS` | 5,000 | records per upload |
| `INGEST_RECORDS_PER_HOUR` | 20,000 | per install id |

The endpoint is `POST /v1/records` with headers `content-encoding: gzip`, `x-aar-schema: 1`,
`x-aar-install: <16+ hex>`. Each line of the body is validated against `ingest/schema.py` (unknown
keys, free text and out-of-range numbers are refused); good records are stored, bad ones are counted
in the response and never stored or echoed. There is no way to read stored data back through this app.

Point gateways at it with `aar-gateway share endpoint https://your-host/v1/records` (or
`AAR_GATEWAY_SHARE_ENDPOINT`). To ship a default for everyone, set `DEFAULT_ENDPOINT` in
`aar_gateway/share.py` (in `aar-gateway/`).

## Publish to Kaggle

```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...        # or ~/.kaggle/kaggle.json on this machine
python -m ingest.publish_kaggle --shards $INGEST_SHARD_DIR --dataset <owner>/<slug> --create --license <name>
# afterwards, from cron (daily is plenty); it does nothing when there is nothing new:
python -m ingest.publish_kaggle --shards $INGEST_SHARD_DIR --dataset <owner>/<slug>
```

Each publish is a new dataset version containing one gzip per UTC day. Choose the dataset's
visibility and license on Kaggle deliberately: the data is anonymous, but it is user-derived.

Not verified against the real Kaggle service (the tests use a stand-in `kaggle` command): run the
first `--create` by hand and check the dataset page before scheduling it.
