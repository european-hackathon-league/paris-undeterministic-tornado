import subprocess
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout

# Try competition download via Bearer token (KGAT) directly against Kaggle API
cmd = (
    'cd /root/work && '
    'curl -sL -H "Authorization: Bearer KGAT_293cb3c384a9537de7bd3877e5285023" '
    '"https://www.kaggle.com/api/v1/competitions/data/download-all/ehl-paris-medical-image-retrieval" '
    '-o comp.zip -w "HTTP %{http_code} size %{size_download}\\n"'
)
print("=== curl bearer attempt ==="); print(sh(cmd))
print(sh("cd /root/work && ls -la comp.zip; file comp.zip; head -c 200 comp.zip | tr -d '\\000'"))
print("=== brats progress ==="); print(sh("tr '\\r' '\\n' < /root/work/dl_part1.log | tail -1; tr '\\r' '\\n' < /root/work/dl_part2.log | tail -1; echo flag:; cat /root/work/dl_all_done.flag 2>/dev/null || echo not-done; du -sh /root/work/brats"))
