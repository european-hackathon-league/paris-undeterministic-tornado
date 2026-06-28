#!/usr/bin/env python
"""Run a python file on the remote Jupyter server. Usage: run_remote.py <file.py> [timeout]"""
import sys
from jupyter_exec import execute

URL = "http://134.199.198.104"
TOKEN = "tP1Kw7bI4y0kM0qNhesV3OezEd1Ii1YDHTCyFfuUhgyRsKXzd"

code = open(sys.argv[1]).read()
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
sys.exit(execute(URL, TOKEN, code, timeout))
