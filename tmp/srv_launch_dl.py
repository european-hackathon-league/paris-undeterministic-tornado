import subprocess, os, json, time
os.makedirs("/root/work/brats", exist_ok=True)
os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump({"username": "kgat", "key": "KGAT_293cb3c384a9537de7bd3877e5285023"}, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)

script = r'''#!/bin/bash
export KAGGLE_KEY=KGAT_293cb3c384a9537de7bd3877e5285023
cd /root/work/brats
kaggle datasets download aiocta/brats2023-part-1 -p /root/work/brats --unzip > /root/work/dl_part1.log 2>&1 &
P1=$!
kaggle datasets download aiocta/brats2023-part-2zip -p /root/work/brats --unzip > /root/work/dl_part2.log 2>&1 &
P2=$!
cd /root/work
kaggle competitions download -c ehl-paris-medical-image-retrieval -p /root/work > /root/work/dl_comp.log 2>&1 &
P3=$!
wait $P1 $P2 $P3
echo DONE > /root/work/dl_all_done.flag
'''
with open("/root/work/dl.sh", "w") as f:
    f.write(script)
subprocess.run("chmod +x /root/work/dl.sh; nohup /root/work/dl.sh > /root/work/dl_master.log 2>&1 &", shell=True)
print("launched")
time.sleep(8)
print(subprocess.run("ls -la /root/work; echo '--- logs ---'; tail -3 /root/work/dl_part1.log /root/work/dl_part2.log /root/work/dl_comp.log 2>/dev/null",
                     shell=True, capture_output=True, text=True).stdout)
