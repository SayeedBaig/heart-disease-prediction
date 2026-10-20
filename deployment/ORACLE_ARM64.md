# CardioAI Oracle Cloud ARM64 deployment preparation

This runbook prepares a new Linux ARM64 VM. It does not provision OCI resources,
connect to Neon, migrate data, or configure production credentials.

## Current deployment blockers

- ARM64 dependency resolution was previously checked for CPython 3.11 on a
  glibc 2.28-compatible target, but a full install and inference run on ARM64
  has not been completed.
- The free Ampere A1 allocation is shared by the entire OCI tenancy. Account
  eligibility, home-region capacity, and actual VM creation must be confirmed
  in the OCI console.
- Clinical pickle artifacts record scikit-learn 1.6.1. The production pin is
  aligned to 1.6.1; LightGBM/XGBoost training versions remain unknown.
- Uploaded ECG/echo files are kept under uploads/. Mount that directory to
  persistent block storage before accepting uploads.
- The Hugging Face model is downloaded during the asset-preparation step and
  should be cached on persistent storage. Groq generation remains unavailable
  until a server-side GROQ_API_KEY is configured.

The pinned requirements resolve to ARM64 wheel metadata for CPython 3.11 using
manylinux_2_28_aarch64, manylinux_2_27_aarch64, and manylinux2014_aarch64 tags.
This was a metadata-only resolution; it is not a full ARM64 install or runtime
test and does not prove the native libraries load on the target VM.

## OCI account and instance

Oracle's current Always Free documentation lists Ampere A1 resources equivalent
to a total of 2 OCPUs and 12 GB RAM for an Always Free tenancy, with capacity
available only in the tenancy's home region. The VM creation page must show the
Always Free eligible label; an out-of-capacity message means the shape is not
currently available in that region. Current account limits and eligibility
must be checked in the console before creating an instance.

Use an ARM64 Oracle Linux 9 image (Python 3.11 is available from OL 9.2 onward)
or another ARM64 Linux image with glibc 2.28 or newer. Attach a persistent block
volume for application uploads and model caches. Assign a dedicated Linux user;
do not run the API as root.

After creating and signing in to the VM, verify the platform before installing:

~~~bash
uname -m
getconf GNU_LIBC_VERSION
sudo dnf install -y python3.11 libgomp git
python3.11 --version
~~~

Expected architecture is aarch64; verify the installed libc is at least 2.28.
If the selected image does not provide Python 3.11 or a prerequisite package,
stop and choose a supported image rather than silently using another Python.

## Checkout and persistent directories

After the deployment branch has been reviewed and approved for publication,
clone that branch. The current local work is not pushed yet.

~~~bash
sudo useradd --system --user-group --create-home --shell /bin/bash cardioai
sudo mkdir -p /opt/cardioai /var/lib/cardioai/uploads /var/lib/cardioai/cache/huggingface
sudo chown -R cardioai:cardioai /opt/cardioai /var/lib/cardioai
sudo -u cardioai git clone --branch codex/deployment-prep \
  https://github.com/rishitha-1612/CardioAI.git /opt/cardioai/app
sudo mount --bind /var/lib/cardioai/uploads /opt/cardioai/app/uploads
~~~

The checkout already contains an uploads directory. On this fresh VM, the bind
mount overlays its empty placeholder directories. Before doing this on any VM
that has accepted uploads, inspect and back them up first. Add the bind mount to
/etc/fstab only after verifying the persistent block volume is mounted at
/var/lib/cardioai. Do not copy real patient data to this test environment.

## Install, configure caches, and smoke-test

~~~bash
sudo -u cardioai python3.11 -m venv /opt/cardioai/venv
sudo -u cardioai /opt/cardioai/venv/bin/python -m pip install --upgrade pip
sudo -u cardioai /opt/cardioai/venv/bin/python -m pip install \
  --only-binary=:all: -r /opt/cardioai/app/requirements.txt
~~~

If the wheel-only install fails, record the exact package and platform tag. Do
not build native packages on the VM as an undocumented workaround. Set
HF_HOME=/var/lib/cardioai/cache/huggingface before running Python so the cache
persists across application checkouts.

Run the asset preparation command from the repository root, with the cache
environment configured. It downloads the pinned Sentence Transformers revision,
loads all saved models, runs synthetic CPU inference, and tests a RAG lookup:

~~~bash
cd /opt/cardioai/app
sudo -u cardioai env HF_HOME=/var/lib/cardioai/cache/huggingface \
  /opt/cardioai/venv/bin/python -m deployment.prepare_assets
~~~

It must finish with JSON reporting clinical, ecg, echo, and rag as pass. This
uses synthetic inputs and does not call Groq or touch the database.

## Server environment and database migration

Store variables in a root-owned environment file outside the Git checkout,
readable only by the service user. Never commit it or print it. Required names:

~~~bash
sudo install -d -o root -g cardioai -m 0750 /etc/cardioai
sudo install -o root -g cardioai -m 0640 /dev/null /etc/cardioai/cardioai.env
sudoedit /etc/cardioai/cardioai.env
~~~

- DATABASE_URL: Neon console > project > Connect dialog > pooled connection
  string for normal application traffic. Keep it server-side.
- SECRET_KEY: unique random value of at least 32 characters; generate on the VM
  with python3.11 -c 'import secrets; print(secrets.token_urlsafe(48))'.
- CORS_ORIGINS: exact deployed Vercel origin, with no path or wildcard.
- HF_HOME: persistent cache path shown above.
- Optional GROQ_API_KEY: server-only key from the Groq console for RAG generation.
- Optional SMTP_HOST, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD for email reports.
- SQL_ECHO=false, AUTO_CREATE_SCHEMA=false, and
  CLEANUP_UPLOADS_ON_STARTUP=false in production.

Before the first API start, inspect the target Neon branch and migration version
using the Neon console or a read-only query. Do not assume it is empty. For a
new/approved schema, use a direct non-pooled connection for Alembic, then start
the app with the pooled URL:

~~~bash
cd /opt/cardioai/app
DATABASE_URL='<direct, non-pooled Neon URL>' \
  /opt/cardioai/venv/bin/alembic upgrade head
~~~

Do not use the above placeholder literally. Supply the value through the
protected service environment, never through shell history or chat. If the
database has tables but no Alembic version row, stop and plan a schema baseline;
do not rerun table-creation migrations against it.

Important existing-data blocker: migration
85178fb28e85_add_patient_auth_fields.py now checks for existing patient rows
before any schema changes and refuses to run when it finds them. The current
repository has no patient password-reset flow. For an existing database, stop
and create an explicitly reviewed recovery plan and tested migration before
changing that guard or applying this migration. The migration is still an
explicit operator action; it is not run automatically by API startup.

## Run as a managed service

Create /etc/systemd/system/cardioai.service as root, with a protected
EnvironmentFile path and these service settings:

~~~ini
[Unit]
Description=CardioAI FastAPI service
After=network-online.target
Wants=network-online.target

[Service]
User=cardioai
Group=cardioai
WorkingDirectory=/opt/cardioai/app
EnvironmentFile=/etc/cardioai/cardioai.env
ExecStart=/opt/cardioai/venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
~~~

Use a reverse proxy and valid TLS certificate for public HTTPS. Expose only SSH
restricted to your IP and HTTPS publicly; do not expose PostgreSQL or the
Uvicorn port. Enable/start the unit only after the migration and smoke tests.

Verify liveness and readiness locally on the VM:

~~~bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
~~~

/ready checks database reachability, required tracked model/index files, and
the locally cached pinned embedding-model snapshot without downloading it. The
asset preparation command is still required to verify that the ML models load
and run inference.

## Vercel connection

The frontend already reads VITE_API_URL. After the backend has a verified HTTPS
hostname, set that variable in Vercel for Production and redeploy. Set backend
CORS_ORIGINS to the known production origin
https://cardio-ai-roan.vercel.app, and separately add preview origins only if they are
intentionally supported. Never set DATABASE_URL, SECRET_KEY, GROQ_API_KEY, or
SMTP credentials in Vercel's Vite client variables.

Test the actual deployed GET /health, GET /ready, login, upload, prediction,
and RAG paths before describing frontend-to-backend integration as working.

## Backups, rollback, and troubleshooting

- Keep database recovery in Neon; do not treat the VM as the only database
  backup. Back up persistent uploads and test restoring them.
- Keep the previous service checkout/release until the new release passes
  health and integration checks. Roll back by switching the service to the
  previous release and restarting it. Do not automatically downgrade a schema.
- For 503 /ready, inspect service logs locally, verify database access and
  required tracked assets, and rerun deployment.prepare_assets; never paste
  connection strings or logs containing credentials into chat.
- For ModuleNotFoundError or wheel resolution errors, record Python version,
  uname -m, libc version, package name, and wheel tag. Do not claim ARM64
  support until the exact install and inference smoke tests pass on ARM64.
- For missing uploads after restart, check that /opt/cardioai/app/uploads is
  mounted to the persistent block volume and that cardioai owns it.
