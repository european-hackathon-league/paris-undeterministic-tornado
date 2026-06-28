import subprocess
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout
print("=== comp.log ==="); print(sh("cat /root/work/dl_comp.log"))
print("=== part1 tail (strip CR) ==="); print(sh("tr '\\r' '\\n' < /root/work/dl_part1.log | tail -3"))
print("=== part2 tail ==="); print(sh("tr '\\r' '\\n' < /root/work/dl_part2.log | tail -3"))
print("=== disk used in /root/work ==="); print(sh("du -sh /root/work/* 2>/dev/null; echo; ls /root/work/brats | head; echo count:; ls /root/work/brats 2>/dev/null | wc -l"))
print("=== done flag? ==="); print(sh("cat /root/work/dl_all_done.flag 2>/dev/null || echo not-done"))
