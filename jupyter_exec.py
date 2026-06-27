from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import requests
import websocket


def execute(base_url: str, token: str, code: str, timeout: float) -> int:
    base_url = base_url.rstrip("/")
    session = requests.Session()
    kernel_resp = session.post(f"{base_url}/api/kernels", params={"token": token}, timeout=timeout)
    kernel_resp.raise_for_status()
    kernel_id = kernel_resp.json()["id"]
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws = websocket.create_connection(
        f"{ws_url}/api/kernels/{kernel_id}/channels?token={token}",
        timeout=timeout,
    )

    msg_id = uuid.uuid4().hex
    msg = {
        "header": {
            "msg_id": msg_id,
            "username": "codex",
            "session": uuid.uuid4().hex,
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": False,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "buffers": [],
        "channel": "shell",
    }
    ws.send(json.dumps(msg))

    exit_code = 0
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            raw = ws.recv()
            event = json.loads(raw)
            parent = event.get("parent_header", {})
            if parent.get("msg_id") != msg_id:
                continue
            msg_type = event.get("msg_type") or event.get("header", {}).get("msg_type")
            content = event.get("content", {})
            if msg_type == "stream":
                print(content.get("text", ""), end="")
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                if "text/plain" in data:
                    print(data["text/plain"])
            elif msg_type == "error":
                exit_code = 1
                print("\n".join(content.get("traceback", [])))
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        else:
            raise TimeoutError(f"Execution timed out after {timeout} seconds")
    finally:
        ws.close()
        session.delete(f"{base_url}/api/kernels/{kernel_id}", params={"token": token}, timeout=timeout)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("JUPYTER_URL", "http://134.199.198.104"))
    parser.add_argument("--token", default=os.environ.get("JUPYTER_TOKEN"), required=False)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("code")
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("Missing --token or JUPYTER_TOKEN")
    raise SystemExit(execute(args.url, args.token, args.code, args.timeout))


if __name__ == "__main__":
    main()
