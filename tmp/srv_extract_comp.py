import subprocess, textwrap
# write an extractor and run it in background (python zipfile, since unzip is absent)
extractor = textwrap.dedent('''
    import zipfile, os, time
    os.makedirs("/root/work/comp", exist_ok=True)
    t=time.time()
    with zipfile.ZipFile("/root/work/comp.zip") as z:
        names=z.namelist()
        print("entries:", len(names), flush=True)
        z.extractall("/root/work/comp")
    open("/root/work/comp_done.flag","w").write("done %.0fs"%(time.time()-t))
    print("extracted in %.0fs"%(time.time()-t))
''')
open("/root/work/extract_comp.py","w").write(extractor)
subprocess.run("cd /root/work && nohup python extract_comp.py > extract_comp.log 2>&1 &", shell=True)
print("extraction launched in background")
import time; time.sleep(3)
print(subprocess.run("cat /root/work/extract_comp.log 2>/dev/null; echo; ls /root/work/comp 2>/dev/null",shell=True,capture_output=True,text=True).stdout)
