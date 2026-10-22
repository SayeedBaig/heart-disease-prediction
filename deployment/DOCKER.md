# CardioAI backend with Docker

Docker runs the FastAPI backend in an isolated container. It does not containerize
or replace the Vercel frontend, and this configuration does not start a database
or apply database migrations. The backend connects to the PostgreSQL database
named by `DATABASE_URL`.

## Local run

1. Install Docker Desktop or Docker Engine with the Compose plugin.
2. Copy `.env.example` to `.env` and set `DATABASE_URL`, `SECRET_KEY`, and the
   exact `CORS_ORIGINS` value for the frontend origin you are using. Keep
   `.env` private. Use a disposable database for initial testing; never enable
   `AUTO_CREATE_SCHEMA` for a production database.
3. Build the API image and apply the database migrations before starting it:

   ```sh
   docker compose build api
   docker compose run --rm api alembic upgrade head
   docker compose up -d
   ```

   The defaults expose the API at host port 8000. Set `PORT` to change the
   container port and `HOST_PORT` to change the host port. Compose reads `.env`
   at runtime; use a disposable test database for initial testing.

   Migrations are intentionally not run automatically by the image. Back up
   the database and review the migration before applying it to a deployment.

4. Check the liveness endpoint at `http://localhost:8000/health` (replace `8000`
   with your configured `HOST_PORT`). The
   `/ready` endpoint also checks PostgreSQL, the tracked model files, and the
   pinned Sentence Transformers model; it returns `503` if a dependency is not
   available.
5. To use the local Vite frontend, set `VITE_API_URL=http://localhost:8000`
   in the ignored `frontend/.env.local` file, then run the frontend as usual.

Compose stores uploads in a named Docker volume so container recreation does
not remove them. A volume is local to this Docker host; it is not backed up or
shared automatically. Hosted containers may have ephemeral disks, so uploaded
ECG/echo files can disappear on restart unless persistent storage is configured.
The image contains the clinical, ECG, and echo artifacts, the
FAISS files, and the pinned embedding model. It does not contain local `.env`
files, SQLite databases, uploaded files, the frontend, `node_modules`, or the
tracked sample video dataset. No database migration runs at image build or
container startup. Docker installs the pinned CPU-only PyTorch and XGBoost
packages because application inference uses CPU execution. Package versions in
`requirements.txt` remain unchanged; the CPU XGBoost distribution provides the
same `xgboost` Python module. The application does not use GPU-only APIs.
Authenticated report context is kept in the PostgreSQL `chat_sessions` table,
keyed by the JWT session ID; logout removes it, and expired sessions are
purged periodically. The cleanup interval defaults to 900 seconds and can be
set with `CHAT_SESSION_CLEANUP_INTERVAL_SECONDS`. Apply the Alembic migration
before using the updated application.

To stop the container without removing uploads, run `docker compose down`. Do
not use `docker compose down --volumes` if you need to retain the named uploads
volume.

## Deploying the image

- Set `DATABASE_URL`, `SECRET_KEY`, and `CORS_ORIGINS` in the hosting
  provider's protected runtime settings. Add `GROQ_API_KEY` and SMTP settings
  only if those features are enabled. Never add these values to the Dockerfile,
  image build arguments, frontend variables, or Git.
- Set `CORS_ORIGINS` to the exact Vercel origin (scheme and host only). Set
  `VITE_API_URL` in Vercel to the actual HTTPS API URL after the backend is
  running. Do not use localhost or a guessed URL in production.
- Attach persistent storage at `/app/uploads`; uploaded ECG/echo files are not
  stored in PostgreSQL and are lost if the host's container filesystem is
  ephemeral.
- Keep `AUTO_CREATE_SCHEMA=false`, `CLEANUP_UPLOADS_ON_STARTUP=false`, and
  `SQL_ECHO=false`. Review the target database's current schema and patient
  rows, make a backup, and agree on a migration plan before running Alembic.
  In particular, the patient password-hash migration intentionally refuses to
  proceed when existing patient rows are present.
- The Docker image is Linux and follows the build host's CPU architecture.
  The existing dependency audit did not establish ARM64 compatibility for the
  full TensorFlow/PyTorch stack. Use a Linux x86_64 host for this image unless
  a separate ARM64 build and end-to-end inference test passes first. An
  x86_64 image is not a native image for an Ampere A1 VM.

The container's `/health` endpoint is a liveness check only. Use `/ready` to
verify database and model availability after configuring the service.
