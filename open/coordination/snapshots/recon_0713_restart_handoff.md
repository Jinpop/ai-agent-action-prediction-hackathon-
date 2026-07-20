# RECON 스냅샷 — 07-13 재시작 인수인계 (read-only 그라운딩)

> 수집: 2026-07-13 00:15~00:35 KST. 목적: 세션 재시작 후 새 세션이 `.claude/agents/` 서브에이전트(train-launcher/zip-verifier/log-analyst)와 함께 최신 상태를 이어받게 함. 진실소스는 HANDOFF.md — 이 파일은 recon 증거 보존용. **mint2 섹션은 재정찰 진행 중(완료 시 갱신).**

## 0. 세션 컨텍스트 (왜 이 스냅샷이 있나)
- 사용자가 오케스트레이션 도입 → `.claude/agents/`에 3역할 에이전트 생성(train-launcher/zip-verifier/log-analyst). **단 세션 시작 시에만 로드돼서 현 세션 레지스트리엔 없음** → 재시작 후 사용 가능. (워크플로우가 `agent type 'train-launcher' not found`로 확인.)
- 07-13 슬롯배분(사용자 지시)·전략원칙(keep-시드 비주력, 큰 카드 우선)은 CLAUDE.md 핵심규칙 + HANDOFF §9에 이미 반영됨.

## 1. 치타 원격 실측 (SSH read-only, 00:30 KST)
- **GPU: A5000 0/1 둘 다 idle (0%, ~0MiB/24564MiB). 실행 중 colab_train_base2 프로세스 없음.** → 학습 2팔 즉시 발사 가능 상태(gate·사용자 승인 후).
- **oof_ids 둘 다 존재:** `run_oof74/oof_ids_seed74.json`(2.6MB, 07-12 06:44), `run_oof79/oof_ids_seed79.json`(2.6MB, 07-12 11:40). → 증류 강한 kf teacher id-정렬 복구 가능.
- **pt74c 배포자산 원격 존재:** `run_kf768pt74c/`에 `model_sub`(배포팩) + `holdout_probs3.npy` + `holdout_idx3.npy` + colab/feat/data/requirements/env_manifest. → C콤보 fetch 가능·밴드체크 재료.
- **★백본 캐시 발견:** `~/.cache/huggingface/hub/models--team-lucid--deberta-v3-base-korean` 치타에 **이미 캐시됨**(+ mdeberta-v3-base). → deberta-v3-korean 백본은 오프라인 다운로드 불필요.
- 멤버 단독 홀드아웃(참고, 블렌드 아님): k79=0.7617 · k78=0.7590 · k81=0.7567 · pt74c=0.7551 · k77=0.7541 · kf768(base)=0.7579 · s48mf(mint-fine, 배포 s48 아님)=0.7398. h83은 치타에 run 없음(H100/elice 산출 → 로컬 scratchpad/pack_h83).

## 2. ★★ A/B/C 콤보 = 배포 envelope 위반 (headline blocker)
staging이 submits/submit_v58.zip 실제 해부:
- v58 = classic(model/artifacts.pkl 86.3MB) + s48(model_s48, **RoBERTa 512-len, 압축 203.6MB**) + kf768_79 + kf768_81(각 압축 341.2MB). 총 978,557,947B. 블렌드=argmax(0.4×classic + 0.6×mean(s48,k79,k81)). **script.py 라인63 SEED_DIRS 하드코딩** → 멤버 교체 시 스크립트 편집 필수(파일 드롭인 아님).
- **문제:** 3콤보 모두 s48(512, 작고 빠름)을 768-len kf-DeBERTa(압축 341.2MB)로 교체 → ①zip ≈1,116MB로 **1GB decimal 한도 +116MB 초과** ②768-len 트랜스포머 3개 → v58의 9:46/10:00(마진 16초) **시간마진 붕괴**(거의 확실 timeout). INT8 양자화 툴링은 레포에 없음.
- 아티팩트 존재: A(k78=`scratchpad/v49/model_kf768_78`)·B(h83=`scratchpad/pack_h83/model_sub`)는 로컬 완비, C(pt74c)는 로컬 배포팩 없음(치타 fetch 필요).
- **결론: A/B/C는 순수 s48→kf768 드롭인 스왑으로는 배포 불가.** 슬롯 태우기 전에 **로컬 홀드아웃 블렌드 밴드체크로 무료 스크리닝**(홀드아웃=자격필터, §3.4 — 우열판정 아님) 후, in-band인 것만 배포재설계(INT8 등) 검토.

### A/B/C 밴드체크 셋업 (post-restart log-analyst 첫 작업)
- 계약(§14): 각 멤버 홀드아웃 **raw logits → 멤버별 softmax 후** 블렌드. logit/prob 혼용 금지. 동일 hidx(동일 split) 교차확인 필수.
- 재료: pt74c=`scratchpad/hp_kf768pt74c.npy`+`scratchpad/hidx_pt74c.npy`(로컬). 나머지(s48,k79,k81,k78,classic)는 각 치타 run의 `holdout_probs3.npy`+`holdout_idx3.npy` fetch 필요(동일 split인지 idx 대조). 
- 계산: v58기준(classic+s48+k79+k81) vs A(+k78)/B(+h83)/C(+pt74c) 블렌드 홀드아웃 F1 → 밴드 = ≥(v58홀드 − 0.0058). in-band Y/N만 보고.

## 3. 증류 (③) — 메커니즘 OK, 블로커 2개 중 1개 SSH로 해소
- colab_train_base2.py 플래그: `SOFT_TARGET=<npz(probs N×14 + ids)>`, `DISTILL_T`(기본2.0), `DISTILL_ALPHA`(기본0.5=hard-CE 가중), `OOF=1`(5-fold GroupKFold→oof_logits/ids 방출). hard CE는 항상 계산 후 블렌드 → **'hard CE 유지' 네이티브 충족**(0<alpha<1). class order = feat.ACTIONS 일치 확인됨.
- **블로커A(코드픽스 필요): temperature mismatch.** `make_soft_target.py`가 teacher를 **T=1 plain softmax**로 굽는데 student는 T=2 → v26 버그(체크리스트20 item5). "temperature matched"엔 make_soft_target.py에 T 인자 추가(softmax(logits/T)) 또는 npz에 raw logits 저장+forward에서 teacher도 T 적용 필요.
- **블로커B(SSH로 해소): 강한 kf teacher.** 로컬 `scratchpad/oof_logits_seed74.npy`(73180행, mint 포함)는 ids sidecar 없었으나 **치타 run_oof74/oof_ids_seed74.json·run_oof79/oof_ids_seed79.json 둘 다 존재** → fetch해서 id-정렬 가능. classic OOF는 `open/artifacts/oof/oof_classic_probs.npy` 로컬 완비.
- 커버리지: soft.npz/classic/seed80·81은 real 70000 id만 → 챔피언레시피(EXTRA_DATA=train_mint) student는 KeyError. **real-only student로 돌리거나** mint 커버 teacher(seed74 73180행) 사용.
- 제약(사용자): OOF teacher만·split/class order/temp matched·hard CE 유지·s48 1슬롯 교체·**밴드통과해도 '증류 성공' 호칭 금지·probe 1장 한정.** v26 전면교체 증류는 서버확정 죽은축(§5)이나 이 스코프는 다름.

## 4. 신규 백본 (④) — Qwen 폐기, deberta-v3-korean으로 피벗
- **Qwen 0.5B BLOCKED:** 오프라인 캐시 없음 + fp16 단독~1GB(예산초과) + **디코더(CLS 없음)→HybridNet 풀링 코드변경** + INT8 툴링 없음 + T4시간 서버probe만. 최고위험.
- **★추천 대체: `team-lucid/deberta-v3-base-korean`** — 치타 캐시 이미 있음(다운로드 불요) + **인코더(CLS 풀링 그대로 OK, 코드변경 최소)** + 한국어+DeBERTa-v3(유일 생존계열) + ~base급이라 fp16 <1GB 여유 + fast tokenizer(sentencepiece 배포리스크 낮음). "오류집합을 바꾸는 큰 카드" 취지에 부합(kf-deberta와 다른 사전학습).
- 게이트(§6 card7 순서): zip<1GB + net.half() + fast tokenizer + T4 단독추론 확인(A5000라 로컬 측정불가 → 서버 probe 또는 보수추정) → 통과 후 2seed. 새 백본은 **ADD 불가(예산포화)→단독 probe 또는 슬롯 교체**. 재사용 툴: conv_fp16.py·convert_base_st.py·verify_zip.py·build_deploy.py.
- 이미 기각: electra(열세)·mdeberta(kf보다 열등, run_mdeb 홀드 0.7294).

## 5. mint2 balanced (②, 최상위 카드) — 재정찰 완료 (code-grounded)
모든 수치 `open/.venv/bin/python scratchpad/audit_mint_recoverability.py --json`로 실측 재현.
- **현 상태:** train_mint.jsonl=3,180행(mint1 longest/target), **가중치 컬럼 없음**. EXTRA_DATA는 real 70k에 **단순 동등가중 concat**(colab_train_base2.py:204). per-sample WT는 SESSION_EQUAL/AU_BOOST로만 채워지고 **데이터 필드는 안 읽음**(:214-221,266-267) → target별 가중 부재.
- **이미 존재하는 산출물:** `train_mint_exact.jsonl`(2,863 strict∩exact)·`train_mint_exact_gate.jsonl`(2,314 holdout제외)·`train_mint_pre.jsonl`(2,542)·`train_mint2.jsonl`(6,859 롤링윈도우 소스, max 6 variants/target). 그룹핑 로직 audit:165-176, emit 골격 `scratchpad/build_mint_exact.py`.
- **구현 필요(현 코드로 불가):** 신규 빌더 `build_mint_balanced.py`(그룹핑 audit:168-176 재사용) + colab에 **신규 가중경로**.
  - ③ target별 총가중=1: **경로A(권장·최소침습)** = 신규 env `TARGET_BALANCE=1`, SESSION_EQUAL(:214-217)이 세션 count로 나누는 로직을 **canonical target key 기준 count로 치환**(target key는 id에서 파싱). per-epoch resampling은 커스텀 sampler 필요 → 경로A가 안전.
  - ⑥ **budget/elapsed·zero-sentinel 원천차단 = PRETEXT 경로.** PRETEXT면 119d 전체 0화(:238-240) → 차별 sentinel 없음. exact 필드는 텍스트 `[META]`로만(budget/elapsed는 텍스트에 애초에 없음). **∴ `PRETEXT=1 PRETEXT_META=keep`(stage A mint-only)로 태우면 v40 sentinel 함정 회피.** non-PRETEXT 직접주입(:204)이 v40이 밟은 함정.
- **★★누수 landmine (반드시 먼저 결정):** 코드 leak-guard(:244-248)는 **GroupKFold(5) fold-0** holdout을 쓰는데 gate 파일·audit는 **scratchpad/hidx.npy** holdout을 씀 → **두 split이 다름(세션 1,885 중 385만 겹침).** balanced gate는 학습·평가·밴드체크의 holdout을 **hidx 또는 GroupKFold-fold0 중 하나로 통일·고정**해야 누수 0 성립. (refit_eval*.py는 전부 GroupKFold fold-0.) → **plan_review Codex 게이트의 핵심 점검항목.**
- **stage 구조:** A(`PRETEXT=1 PRETEXT_META=keep EXTRA_DATA=balanced.jsonl`)→`./pretext_backbone` 저장 후 exit → B(`INIT_BACKBONE=./pretext_backbone`, EXTRA_DATA 없이 real-only, 홀드+refit). PRETEXT⊥INIT_BACKBONE(:67) → A→B 순차 필수. GRAD_CKPT=0. seed79=GPU0 / seed81=GPU1. (발사 커맨드 초안은 재정찰 원문 §4 참조.)
- **서버 맥락:** text-only→real-only 축은 seed74 v52 +0.00083(vs v47)·seed79 v54 −0.00009(vs v49) — seed-독립 가산 아니고 강한 draw서 체감(§6.3). balanced-mint는 그 **stage A 데이터 품질(과표집 제거·window 다양성)**을 바꾸는 변형.
- 발사 전 강제: checklist `10_pre_training`(strict 3,162/exact 2,863/gate 2,314 assert·gate 누수0·budget/elapsed 0건·stage B real-only 확인).

## 6. Post-restart 작업 큐 (권장 순서)
1. 이 스냅샷 + HANDOFF §9(07-13 큐) 읽기. `/agents`로 서브에이전트 등록 확인.
2. **log-analyst**: A/B/C 홀드아웃 블렌드 밴드체크(§2 셋업) — 무료 스크리닝. + 배포 envelope 재확인.
3. mint2 재정찰 결과로 balanced jsonl 생성코드 작성 → **data-auditor plan_review PASS**(누수 landmine·가중=1·budget/elapsed금지) → 10_pre_training → pre_launch → seed79/81 발사(train-launcher, 워처1).
4. ④ 백본: deberta-v3-base-korean으로 zip/시간 게이트 카나리(runtime_status 보고) → 통과 후 2seed.
5. ③ 증류: make_soft_target.py temp-match 코드픽스 + oof74/79 teacher fetch → real-only student 1장(probe 한정).
- 불변 경계: 제출=사용자직접 / H100=사전허락 / 원격pkill=승인 / 파일수정=지휘자만 / **위험단계 data-auditor PASS 필수**(Codex는 선택 병행).
- **★07-13 하네스 교정(재시작 후 유효):** 필수게이트=data-auditor(Codex 대체) / 훅은 안내(강제X) / verify_zip.py=v49전용(비-v49는 unverified) / 5행E2E=스모크·T4시간은 runtime_status 별도. 상세 CLAUDE.md §검증규약·§독립감사게이트.
