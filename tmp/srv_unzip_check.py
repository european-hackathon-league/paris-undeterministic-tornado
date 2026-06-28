import subprocess, os
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace").stdout

print("=== brats status ==="); print(sh("tr '\\r' '\\n' < /root/work/dl_part1.log | tail -1; tr '\\r' '\\n' < /root/work/dl_part2.log | tail -1; echo flag:; cat /root/work/dl_all_done.flag 2>/dev/null || echo NOT-DONE"))
print("=== brats dir ==="); print(sh("ls /root/work/brats; echo; echo 'GLI subject dirs:'; ls -d /root/work/brats/BraTS-GLI-* 2>/dev/null | wc -l"))
# unzip competition data if not yet
if not os.path.exists("/root/work/comp/dataset1"):
    print("=== unzipping comp.zip ==="); print(sh("cd /root/work && mkdir -p comp && unzip -q -o comp.zip -d comp 2>&1 | tail -3; echo done"))
print("=== comp contents ==="); print(sh("ls /root/work/comp 2>/dev/null; echo; find /root/work/comp -name '*.nii' 2>/dev/null | wc -l; find /root/work/comp -name 'train_pairs.csv' 2>/dev/null"))
print("=== disk ==="); print(sh("df -h / | tail -1; du -sh /root/work/* 2>/dev/null"))
