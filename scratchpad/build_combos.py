import os, shutil, subprocess, re
BASE="scratchpad/stage_v58"
PACKS={
 "model_s48":"scratchpad/stage_v58/model_s48",
 "model_kf768_79":"scratchpad/stage_v58/model_kf768_79",
 "model_kf768_81":"scratchpad/stage_v58/model_kf768_81",
 "model_kf768_m2b79":"scratchpad/pack_m2b79/model_sub",
 "model_kf768_m2b81":"scratchpad/pack_m2b81/model_sub",
 "model_dv3ko_s79":"scratchpad/pack_dv3ko_s79/model_sub",
 "model_dv3ko_s81":"scratchpad/pack_dv3ko_s81/model_sub",
}
# classic+s48+2멤버 (전부 신규, 미제출). envelope: s48(512)+2×768.
COMBOS={
 "cA":["model_s48","model_kf768_81","model_kf768_m2b79"],       # kf81+mint2b_s79 (mint2b_s79 k79슬롯)
 "cB":["model_s48","model_kf768_m2b79","model_kf768_m2b81"],    # 2 mint2b
 "cC":["model_s48","model_kf768_81","model_kf768_m2b81"],       # kf81+mint2b_s81
 "cD":["model_s48","model_kf768_m2b79","model_dv3ko_s79"],      # mint2b_s79+dv3ko_s79
 "cE":["model_s48","model_kf768_81","model_dv3ko_s79"],         # kf81+dv3ko_s79
 "cF":["model_s48","model_kf768_m2b79","model_dv3ko_s81"],      # mint2b_s79+dv3ko_s81
}
for name,members in COMBOS.items():
    st=f"scratchpad/stage_{name}"
    if os.path.exists(st): shutil.rmtree(st)
    subprocess.run(["cp","-al",BASE,st],check=True)
    for d in list(os.listdir(st)):
        if d.startswith("model_") and d!="model": shutil.rmtree(os.path.join(st,d))
    for m in members:
        subprocess.run(["cp","-al",PACKS[m],os.path.join(st,m)],check=True)
    sp=os.path.join(st,"script.py"); txt=open(sp).read()
    new="SEED_DIRS = ["+", ".join(f"'./{m}'" for m in members)+"]"
    txt=re.sub(r"SEED_DIRS = \[[^\]]*\]",new,txt,count=1)
    # ★하드링크 분리(07-14 수정): cp -al 클론의 script.py는 BASE·타 staging과 inode 공유 —
    # open('w') in-place 쓰기는 공유 inode를 덮어써 전 staging을 오염시킴(cA~cF 사고).
    # 반드시 unlink 후 새 파일로 생성해 이 staging만의 inode를 갖게 한다.
    os.unlink(sp)
    with open(sp,"w") as f: f.write(txt)
    assert os.stat(sp).st_ino != os.stat(os.path.join(BASE,"script.py")).st_ino, "script.py가 여전히 BASE와 inode 공유 — 오염 위험"
    for junk in subprocess.run(["find",st,"-name",".DS_Store"],capture_output=True,text=True).stdout.split():
        os.remove(junk)
    z=f"submits/submit_{name}.zip"
    if os.path.exists(z): os.remove(z)
    subprocess.run(f"cd {st} && zip -rq1 -X ../../submits/submit_{name}.zip . -x '*.DS_Store' '*__pycache__*'",shell=True,check=True)
    sz=os.path.getsize(z)
    print(f"{name}: {members[1]}+{members[2]} → {z} {sz:,}B {'✓<1GB' if sz<1e9 else '✗OVER'} | SEED_DIRS={new.split('[')[1]}")
