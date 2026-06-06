"""Add local SSH public key to server authorized_keys (one-time, uses password)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = "62.234.27.253"
USER = "ubuntu"
PASS = os.environ.get("OER_DEPLOY_PASS", "")
KEY = Path.home() / ".ssh" / "id_ed25519_redox"
PUB = Path(str(KEY) + ".pub")


def main() -> int:
    if not PASS:
        print("Set OER_DEPLOY_PASS", file=sys.stderr)
        return 1
    publine = PUB.read_text(encoding="utf-8").strip()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=20, allow_agent=False, look_for_keys=False)
    cmd = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        f"grep -qxF '{publine}' ~/.ssh/authorized_keys 2>/dev/null || "
        f"echo '{publine}' >> ~/.ssh/authorized_keys && "
        "chmod 600 ~/.ssh/authorized_keys && echo KEY_INSTALLED"
    )
    _, out, err = c.exec_command(cmd)
    print(out.read().decode())
    if err.read().decode().strip():
        print(err.read().decode(), file=sys.stderr)
    c.close()

    key = paramiko.Ed25519Key.from_private_key_file(str(KEY))
    c2 = paramiko.SSHClient()
    c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c2.connect(HOST, username=USER, pkey=key, timeout=20, allow_agent=False, look_for_keys=False)
    _, o, _ = c2.exec_command("whoami")
    print("Key login OK:", o.read().decode().strip())
    c2.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
