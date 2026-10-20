# CardioAI API on Render

This repository has a Vite frontend and a FastAPI backend. The Blueprint in
`render.yaml` deploys only the Python API from the repository root; deploy the
frontend separately on Vercel.

## Render service settings

- Service type: **Web Service**
- Repository: `rishitha-1612/CardioAI`
- Root directory: `.` (repository root)
- Runtime: **Python**
- Plan: **Free** (suitable for a trial only; see limitations below)
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Python version: `.python-version` selects `3.11.11`

You can create the service from the Blueprint (`render.yaml`) or enter these
settings in the dashboard. The frontend is not built or served by this API.

## Environment variables

Set these in the Render service's Environment page. Never add secret values to
Git, `render.yaml`, Vercel, or the frontend's `VITE_*` variables.

| Name | Required | Value |
| --- | --- | --- |
| `DATABASE_URL` | Yes | Neon PostgreSQL connection string from Neon Console → project → **Connect**. Use the pooled connection string for a web app and preserve its `sslmode=require` query parameter. |
| `SECRET_KEY` | Yes | Generate a unique signing secret locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. Use at least 32 characters; do not reuse it elsewhere. |
| `CORS_ORIGINS` | Yes | `https://cardio-ai-roan.vercel.app` (no path or trailing slash). For a Vercel preview URL, list it explicitly after a comma. Do not use `*`. |
| `SQL_ECHO` | No | Set to `false`; SQL logging can expose sensitive patient data. |
| `CLEANUP_UPLOADS_ON_STARTUP` | No | Defaults to `false`. Set to `true` only if deleting uploads older than seven days at startup is intended. |
| `GROQ_API_KEY` | Optional | Groq Console → API Keys. The RAG/AI explanation generation feature needs this; without it, that feature returns a degraded response. |
| `SMTP_HOST` | Optional | SMTP provider host, if enabling email reports. |
| `SMTP_PORT` | Optional | SMTP provider port (defaults to `587`). Render Free blocks outbound ports 25, 465, and 587, so SMTP email will not work on the Free web service. |
| `SMTP_EMAIL` | Optional | Sender account configured with your mail provider. |
| `SMTP_PASSWORD` | Optional | App password or provider credential; enter as a Render secret. |
| `ALGORITHM` | No | JWT algorithm; defaults to `HS256`. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | No | JWT expiry in minutes; defaults to `60`. |

The `PORT` variable is assigned by Render; do not set it yourself. Never put a
Neon connection string or Groq/SMTP credential in Vercel. Vercel only needs the
public API URL in the frontend variable already used by the app, if configured.

## Neon PostgreSQL

1. Sign in at [Neon Console](https://console.neon.tech/), choose **Create
   project**, use the Free plan, choose a region close to the backend, and
   create a PostgreSQL project (the default database is `neondb`). Neon's
   current Free plan lists 1 GB storage and 100 compute-hours per project each
   month; check the console for current limits.
2. Open the project **Connect** panel, select the primary branch and database,
   enable **Pooled connection**, and copy the connection string. Keep it private;
   do not paste it into chat, GitHub, or Vercel.
3. Set that full string as Render's `DATABASE_URL`. Confirm it uses PostgreSQL
   and includes TLS (`sslmode=require`). The API normalizes `postgres://` to
   `postgresql://`, uses the installed PostgreSQL driver, and never falls back
   to SQLite.
4. On the Render service, set `CORS_ORIGINS` to
   `https://cardio-ai-roan.vercel.app`. On startup the app calls SQLAlchemy
   `create_all`: it creates missing tables
   without dropping or rewriting existing data. This is not a schema migration
   system; apply future schema changes deliberately with reviewed Alembic
   migrations. Back up important data before schema changes.

The models use PostgreSQL UUIDs and JSONB as well as PostgreSQL enum types, so
PostgreSQL is required. There is no extension setup in the current model
definitions.

## Files, models, and Free plan limits

ECG and echo uploads are saved under `uploads/ecg/` and `uploads/echo/`. The
optional startup cleanup can remove files older than seven days, but is disabled
by default to avoid deleting uploads unexpectedly. Render Free has an ephemeral
filesystem: uploads are lost on restart, redeploy, or spin-down.
Prediction history stores paths in PostgreSQL, not file contents. Do not use
this deployment for private medical uploads unless you accept that loss and
have reviewed the data-handling implications. The smallest no-new-service
option is to use only non-sensitive demo inputs; durable uploads require an
object storage product or paid persistent disk, neither is configured here.

Tracked runtime artifacts are present: the ECG checkpoint (~4 MB), echo model
(~34 MB), two clinical pickle model copies (~37 MB each), and FAISS index and
metadata (~7 MB combined). They remain part of the repository deployment.
Prediction models and the FAISS/sentence-transformers stack load only when a
prediction or RAG request needs them, keeping API startup and `/health` lighter.
Inference still imports PyTorch, TensorFlow, OpenCV, FAISS, sentence-transformers,
scikit-learn, XGBoost, and LightGBM. The combined inference workload is substantial.
Render Free currently provides
512 MB RAM and 0.1 CPU; full prediction, echo inference, RAG model loading, or
dependency builds may exceed that memory/compute budget. No memory-fit claim or
deployment result is implied by this configuration. If it exceeds the limit,
the safe resolution is a larger instance or a separately designed inference
service, not silently disabling clinical features.

Render Free services spin down after 15 minutes idle, take about a minute to
wake, and have ephemeral filesystems. Render Free also blocks SMTP ports 25,
465, and 587. Treat this as a demo/test deployment, not as a production service
for patient data.

## Local checks

From the repository root, after copying `.env.example` to `.env` and filling in
a working PostgreSQL URL plus a generated `SECRET_KEY`:

```powershell
python -m pip install -r requirements.txt
python -c "from api.main import app; print(app.title)"
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/health`. Startup also connects to the database
and initializes missing tables. A simple database connectivity check is:

```powershell
python -c "from sqlalchemy import text; from api.database.connection import engine; c=engine.connect(); c.execute(text('SELECT 1')); c.close(); print('database connection ok')"
```

## Deployment checklist

1. The current deployment preparation is pushed to the `main` branch of the
   GitHub repository. Future code changes should also be reviewed and pushed
   before deployment.
2. Create/verify the Neon database and copy its pooled, TLS-enabled connection
   string privately.
3. Connect the repository in Render and create the Blueprint service, or enter
   the settings above manually.
4. Add `DATABASE_URL`, a generated `SECRET_KEY`, and the exact deployed Vercel
   origin as `CORS_ORIGINS`. Set `SQL_ECHO=false`.
5. Add `GROQ_API_KEY` if RAG explanations/chat are needed. Add SMTP values only
   if using a provider/plan reachable from Render; SMTP ports are blocked on
   Free.
6. Deploy and inspect build/start logs. Confirm the service's `/health` check
   passes, then verify the Vercel app can reach the API with the configured
   origin.
7. Test a non-sensitive demo prediction and RAG request. Check memory metrics
   during model loading before relying on inference.
8. Keep uploads non-sensitive and disposable on Free. Decide on durable file
   storage and an appropriate compute plan before handling real patient files.
