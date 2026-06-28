import subprocess, os
os.makedirs("/root/work/brats_full", exist_ok=True)
os.makedirs("/root/work/brats2024", exist_ok=True)
script = r'''#!/bin/bash
export KAGGLE_KEY=KGAT_293cb3c384a9537de7bd3877e5285023
kaggle datasets download pramada/2023-brats-glioma-full -p /root/work/brats_full --unzip > /root/work/dl_full.log 2>&1 &
A=$!
kaggle datasets download nguyenthanhkhanh/brats2024-small-dataset -p /root/work/brats2024 --unzip > /root/work/dl_2024.log 2>&1 &
B=$!
wait $A $B
echo DONE > /root/work/dl_more_done.flag
'''
open("/root/work/dl_more.sh","w").write(script)
subprocess.run("chmod +x /root/work/dl_more.sh; nohup /root/work/dl_more.sh > /root/work/dl_more_master.log 2>&1 &", shell=True)
print("launched extra BraTS downloads")
import time; time.sleep(6)
print(subprocess.run("tr '\\r' '\\n' < /root/work/dl_full.log 2>/dev/null | tail -1; tr '\\r' '\\n' < /root/work/dl_2024.log 2>/dev/null | tail -1", shell=True, capture_output=True, text=True).stdout)
