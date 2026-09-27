# Deployment

## One machine (laptop or server)

```bash
cp .env.example .env            # set POSTGRES_PASSWORD
docker compose up -d --build    # PostGIS, API, worker, dashboard
open http://localhost:3000
```

The API serves the datacube and twin outputs from `./data` (mounted at `/data`). Generate or copy a
cube and run the twin before expecting a map:

```bash
docker compose run --rm worker sailab synthetic generate --name demo
docker compose run --rm worker sailab twin replay 2025-08-10 2025-09-20 --members 0
```

The worker image uses CPU PyTorch; train on the laptop GPU or Kaggle and copy `runs/` into the data
volume.

## Free hosting plan (from the team guide)

| Piece | Where | Notes |
|---|---|---|
| Dashboard | Vercel (Next.js) | set `SAILAB_API_URL` to the API's public URL |
| API + tiles + worker | Oracle Always Free VM | `docker compose up -d api worker` (skip `db` if using Neon) |
| PostGIS | Neon (0.5 GB) | `SAILAB_DATABASE_URL=postgresql+psycopg://...neon.tech/sailab?sslmode=require` |
| Layer files | Cloudflare R2 (10 GB, no download fees) | sync `data/twin/` if the VM disk is small |

Keep the whole deployment private or behind a login while it shows real data: it is an
experimental research output for experts, never a public warning service.

## Environment variables

| Variable | Used by | Default |
|---|---|---|
| `SAILAB_DATA_DIR` | everything | `./data` |
| `SAILAB_RUNS_DIR` | training, twin | `./runs` |
| `SAILAB_RESULTS_DIR` | evaluation | `./results` |
| `SAILAB_DATABASE_URL` | API, twin, `db sync` | SQLite file in the data directory |
| `SAILAB_CUBE` | API | `demo` |
| `SAILAB_CORS_ORIGINS` | API | `http://localhost:3000` |
| `SAILAB_API_URL` | dashboard (rewrites) | `http://127.0.0.1:8000` |
| `SAILAB_VIIRS_URL`, `SAILAB_VIIRS_MATCH` | VIIRS scraper | unset (skipped) |
