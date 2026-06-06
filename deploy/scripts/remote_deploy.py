"""One-shot deploy to cloud server via SSH/SFTP. Password via OER_DEPLOY_PASS env."""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import paramiko

HOST = os.getenv("OER_DEPLOY_HOST", "62.234.27.253")
USER = os.getenv("OER_DEPLOY_USER", "ubuntu")
PASS = os.getenv("OER_DEPLOY_PASS", "")
KEY_PATH = os.getenv(
    "OER_DEPLOY_KEY",
    str(Path.home() / ".ssh" / "id_ed25519_redox"),
)
APP_DIR = os.getenv("OER_DEPLOY_APP_DIR", "/home/ubuntu/redox-llm")
ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(r"E:\Desktop\Nature药物发现\OER\OER_md")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "chroma_db",
    "output",
    ".pytest_cache",
    "data",
}
SKIP_FILES = {".env"}


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
    print(f"$ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out.strip()[:2000])
    if err.strip() and code != 0:
        print(err.strip()[:2000], file=sys.stderr)
    return code, out, err


def mkdir_p(sftp: paramiko.SFTPClient, remote: str) -> None:
    parts = remote.replace("\\", "/").split("/")
    cur = ""
    for p in parts:
        if not p:
            cur = "/"
            continue
        cur = f"{cur.rstrip('/')}/{p}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def upload_dir(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    mkdir_p(sftp, remote)
    n = 0
    for item in local.rglob("*"):
        rel = item.relative_to(local)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if item.name in SKIP_FILES:
            continue
        rpath = f"{remote}/{rel.as_posix()}"
        if item.is_dir():
            mkdir_p(sftp, rpath)
            continue
        mkdir_p(sftp, str(Path(rpath).parent.as_posix()))
        sftp.put(str(item), rpath)
        n += 1
        if n % 50 == 0:
            print(f"  uploaded {n} files...")
    return n


def upload_tree(sftp: paramiko.SFTPClient, local: Path, remote: str) -> int:
    mkdir_p(sftp, remote)
    n = 0
    for item in local.rglob("*"):
        if item.is_dir():
            continue
        rel = item.relative_to(local)
        rpath = f"{remote}/{rel.as_posix()}"
        mkdir_p(sftp, str(Path(rpath).parent.as_posix()))
        sftp.put(str(item), rpath)
        n += 1
        if n % 100 == 0:
            print(f"  uploaded {n} files...")
    return n


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_file = Path(KEY_PATH)
    print(f"Connecting {USER}@{HOST}...")
    if key_file.is_file():
        key = paramiko.Ed25519Key.from_private_key_file(str(key_file))
        client.connect(
            HOST,
            username=USER,
            pkey=key,
            timeout=30,
            banner_timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
    elif PASS:
        client.connect(
            HOST,
            username=USER,
            password=PASS,
            timeout=30,
            banner_timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
    else:
        raise SystemExit("Set OER_DEPLOY_KEY or OER_DEPLOY_PASS")
    print("Connected.")
    return client


def main() -> int:
    client = connect()

    run(
        client,
        "command -v docker >/dev/null || (curl -fsSL https://get.docker.com | sh && systemctl enable --now docker)",
        timeout=900,
    )
    run(client, f"mkdir -p {APP_DIR}/data/md {APP_DIR}/data/chroma_db {APP_DIR}/data/output")

    sftp = client.open_sftp()
    print("Uploading app code...")
    n_code = upload_dir(sftp, ROOT, APP_DIR)
    print(f"  code files: {n_code}")

    env_local = ROOT / ".env"
    if env_local.is_file():
        print("Uploading .env...")
        sftp.put(str(env_local), f"{APP_DIR}/.env")
    else:
        print("WARN: no local .env")

    chroma = ROOT / "chroma_db"
    if chroma.is_dir():
        print("Uploading chroma_db (~340MB)...")
        t0 = time.time()
        n_chroma = upload_tree(sftp, chroma, f"{APP_DIR}/data/chroma_db")
        print(f"  chroma files: {n_chroma} in {time.time()-t0:.0f}s")
    else:
        print("WARN: chroma_db missing")

    if CORPUS.is_dir():
        print("Uploading corpus...")
        n_md = upload_tree(sftp, CORPUS, f"{APP_DIR}/data/md")
        print(f"  md files: {n_md}")
    else:
        print("WARN: corpus missing")

    sftp.close()

    run(client, f"chmod +x {APP_DIR}/deploy/scripts/*.sh")
    env_exports = (
        f"export MD_DATA_DIR={APP_DIR}/data/md "
        f"CHROMA_DATA_DIR={APP_DIR}/data/chroma_db "
        f"CHAT_OUTPUT_DIR={APP_DIR}/data/output"
    )
    cmd = f"cd {APP_DIR} && {env_exports} && sudo -E docker compose up -d --build"
    code, _, _ = run(client, cmd, timeout=1800)
    if code != 0:
        return code

    run(client, f"sudo docker compose -f {APP_DIR}/docker-compose.yml ps")
    run(client, "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8501/_stcore/health || true")
    client.close()
    print(f"\nTeam URL: http://{HOST}:8501")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
