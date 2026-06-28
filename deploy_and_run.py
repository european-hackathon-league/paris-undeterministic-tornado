#!/usr/bin/env python
"""Deploy a local .py to the server and run it in background (nohup).
Usage: deploy_and_run.py <local.py> <remote_name> [run|write]
"""
import sys, base64
from jupyter_exec import execute

URL = "http://134.199.198.104"
TOKEN = "tP1Kw7bI4y0kM0qNhesV3OezEd1Ii1YDHTCyFfuUhgyRsKXzd"

local = sys.argv[1]
remote = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "run"
b64 = base64.b64encode(open(local, "rb").read()).decode()

remote_code = f'''
import base64, subprocess, os
os.makedirs("/root/work", exist_ok=True)
path = "/root/work/{remote}"
open(path, "wb").write(base64.b64decode("{b64}"))
print("wrote", path, len(open(path,'rb').read()), "bytes")
'''
if mode == "run":
    log = remote.replace(".py", ".log")
    remote_code += f'''
subprocess.run("cd /root/work && nohup python {remote} > {log} 2>&1 &", shell=True)
print("launched {remote} -> {log}")
'''
sys.exit(execute(URL, TOKEN, remote_code, 60))
