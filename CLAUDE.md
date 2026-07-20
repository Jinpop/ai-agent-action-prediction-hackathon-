# CLAUDE.md — Dacon 236694 프로젝트 하네스

이 프로젝트(Dacon 236694 AI Agent 행동예측)에서 작업하는 모든 에이전트는 다음을 **반드시** 따른다.

## 🔑 시작 시 (필수, 예외 없음)
작업 전에 이 순서로 읽는다:
1. **`open/docs/HANDOFF.md`** — 단일 진실 소스(현재 챔피언·확립된 법칙·죽은축/살아있는축·인프라 접속·제출 규칙·진행 스냅샷). **이게 최우선.**
2. `open/docs/실험로그.md` — 전 제출 버전별 LB 스코어보드 + 시간순 서사.
3. `open/docs/대회규칙_채점환경.md` — 채점 환경/제약.

메모리(`dacon-236694-agent-action.md`)가 있으면 참고하되, **HANDOFF.md가 우선**이다(메모리 소실 대비 설계).

## ⚙️ 메타규칙 (하네스 자기갱신 — 최우선)
- **★사용자가 규칙·선호·지시를 말할 때마다 즉시 이 CLAUDE.md에 추가한다.** (이 메타규칙 포함) 말로만 받고 넘어가지 말 것. 규칙 확정 즉시 하네스 반영 → 메모리(feedback)도 병행.
- **★★(07-13 사용자 지시) 입출력에서 새 정보가 나올 때마다 즉시 해당 문서를 작성·갱신한다(말로만 받고 넘기지 말 것).** 정보유형 → 문서 매핑: 사용자 규칙/선호 → **CLAUDE.md**(+메모리 feedback); 축 우선순위·판정·법칙·챔피언·인프라·계획/스냅샷 → **HANDOFF**(§1/3/5/6/7/9); 제출·실측 결과·오류발견·정정·서사 → **실험로그(append-only)**; 감사 판정 → **open/coordination/results/**; 학습/zip 절차 규약 → **checklists**; 재사용 배포 사실 → **model_notes**. **변경 지시를 받으면 채팅으로만 답하지 말고 반드시 대상 문서에 반영 후 어디를 고쳤는지 보고한다.** (근거: 07-13 축맵 변경을 문서에 안 써 "미반영" 지적.)
- **★실험/작업(학습·refit·평가 등)을 돌릴 때마다 예상 완료 시각(ETA)을 함께 명시한다.** "언제 끝나냐" 되묻게 하지 말 것. 예: "refit 3변형 ~12분, 완료 ≈ HH:MM경."
- **★ETA를 말했으면 그 시각에 스스로 이어서 작업한다.** 백그라운드 작업은 **반드시 완료감시를 붙일 것**: 로컬은 `run_in_background`로 하네스가 추적하게 하거나(nohup 단독 발사 금지 — 완료 알림 안 옴), 원격(치타)은 블로킹 워처 커맨드를 `run_in_background`로 병행. 완료 알림 수신 즉시 후속 단계(수확→포장→메모) 자동 진행.
- **★백그라운드 작업 위생(07-11 사용자 지시, 07-12 하네스 강제화): 새 백그라운드 작업을 추가할 때마다 기존 백그라운드 작업 중 불필요해진 것(목적 소멸·대체된 워처·잔류 ssh 세션)이 없는지 확인하고 즉시 TaskStop으로 정리한다.** ①워처를 교체하면 구 워처 중지 ②원격 발사용 ssh 세션은 발사 확인 후 잔류 시 정리 ③**동시 한도: 워처 1 + 연산 1** ④검증은 sleep 없는 즉답형 foreground 1회(대기자 중첩 금지) ⑤원격 체인은 서버측 순차 스크립트로 묶어 로컬 대기자 최소화. **강제 장치: `.claude/hooks/bg_hygiene.py`(PreToolUse 훅)가 run_in_background 호출마다 이 체크리스트를 컨텍스트에 자동 주입**(.claude/settings.local.json). 리마인더로 부족하면 permissionDecision=deny 차단형으로 격상 가능.

## 📌 작업 중 (지속 갱신 — 하네스의 핵심)
- **★`실험로그.md`는 append-only 시간순 원장이다(07-10 Codex 감사 후 확립, AGENTS.md와 동일 규칙).** 과거 본문·표·행을 소급 수정·삭제·재서술 금지. 새 제출·오류발견·정정·결론변경은 발생 시각의 새 항목으로 **문서 끝에만 추가**.
- **★신뢰성 규칙(감사 §14):** 축 종결 선언은 matched control(동일 시드·동일 split·1변수 변경)이 있을 때만. 홀드아웃 예측은 그 행을 학습하지 않은 모델로만(full-refit 모델의 train행 예측 ≠ 홀드아웃). 블렌드 합성 전 멤버별 softmax 여부 확인(logit/prob 혼용 금지). 실험 코드·산출물 보존(임시코드 삭제 후 기억으로 서술 금지).
- **매 제출 후:** `실험로그.md` 문서 끝에 결과 항목 추가(버전·구성·홀드아웃·LB·메모) + `HANDOFF.md` §1(챔피언)·§9(스냅샷) 갱신.
- **새 법칙/축 판정 시:** `HANDOFF.md` §3(법칙)·§5(죽은축)·§6(살아있는축) 갱신.
- **인프라 변경 시:** `HANDOFF.md` §7 갱신.
- **★zip 만들 때마다:** 별도 제출메모 파일 두지 말 것 — 제출메모는 **`실험로그.md`로 통합**(스코어보드 행 + 필요시 서사/교정 절에 ①목적②구성③디테일④결과). 그와 **별개로** 사이트 붙여넣기용 짧은 메모는 채팅으로 사용자에게 제공(사용자가 Dacon 사이트에 직접 붙여넣음). zip 내 memo.txt 불필요.
- 메모리도 병행 갱신하되 HANDOFF.md를 진실 소스로 유지.

## ⚖️ 핵심 규칙 요약 (상세는 HANDOFF.md)
- **챔피언은 자동 최종선택**(최고 LB 자동 채택). 라디오 수동조작 없음.
- **★홀드아웃 = 제출 자격 필터일 뿐, 우열 판단자 아님.** 홀드아웃 점수로 "대박/최고"라 부르지 말 것. 밴드(챔피언홀드−0.0058) 통과 = "제출해볼 자격"일 뿐. **우열·챔피언·어느 걸 제출할지는 LB와 전략으로만 판단.** 서버LB ≈ 홀드아웃 + 0.0131 ± 0.0039(참고). 정보축은 홀드아웃으로 죽이지 말 것. 정규화/specialist류는 서버증거로 닫힘(재시도 금지).
- **★★(07-11 사용자 재교정) 학습 중단·후보 폐기·GPU 배분도 홀드아웃 서열로 결정 금지.** 절차: ①프레임 스왑 코어홀드 계산 ②밴드 체크 ③통과면 기본값 = refit 완주·보존·서버 실측(특히 시드복권은 홀드아웃이 서버 승자를 못 맞춘다는 전제의 축). 중단·폐기는 **비홀드아웃 근거**(서버 실측 증거·배포 시간/용량 제약·완전 중복)가 있을 때만. 이 규칙은 m74 refit 중단·s80 폐기 시도(둘 다 홀드아웃 서열 판단)의 재발 방지용.
- **슬롯: 매일 자정(KST) 10개 리셋.** 매일 소진.
- **★★(07-13 사용자 전략지시) keep-시드 증산을 주력으로 삼지 말 것.** 지금 필요한 건 +0.001 복권 여러 장이 아니라 기존 모델의 **오류 집합을 실제로 바꾸는 큰 카드**(balanced-mint augmentation·신규 백본). 시드-only/keep 교체는 값싸게 병행하되 **GPU·슬롯 주력은 큰 카드에 배정**. (근거: 남은 목표차 +0.004~+0.007은 시드복권 분산으로 메우기 어려움 — 79 +0.0012 / 80계열 −0.0015 양방향 실측.)
- **정보축(전달됨)**: 더 긴 컨텍스트·더 많은/좋은 데이터·refit·강한 멤버. **정규화(반납)**: FGM·증류·smooth·specialist·스태킹·CA.
- **배포:** net.half() 필수, zip 1GB, T4 600초, 결합기 없는 순수 블렌드(0.6 seed + 0.4 고전).
- **제출은 사용자 직접**(포장·브리핑만 Claude). **엘리스 H100(과금)은 사전허락.** epoch별 알림 금지·완료시 소요보고.
- **★★(07-14 사용자 지시 — 최우선) 치타 무료 GPU 포함 모든 신규 학습 발사는 매번 사용자의 명시 승인("발사해"급 직접 문구)을 받는다.** 감사 PASS·계획 문서·간접 시사("권장 순서" 텍스트 붙여넣기, "GPU 더 돌릴 거 없어?" 질문)로는 발사 권한이 생기지 않음. 사전 위임이 있어도 그 위임 범위(예: "내일 확인 전까지")가 끝나면 소멸. 근거: 07-14 mint2v2 gate를 외부감사 권장문구 해석만으로 발사 → 사용자 "내가 그런 명령을 했었나?" → 중단 지시. 실행 중인 학습의 중단·kill도 동일하게 사용자 지시로만(이번 kill은 사용자 "멈춰" 직접 지시). **★★(2026-07-14 오후 사용자 선호 변경 — 이 per-command 규칙을 캠페인 범위로 완화) 사용자가 campaign manifest를 1회 명시 승인하면 그 캠페인 범위 내 발사·gate·refit·포장 등 정상 단계전이는 재승인 없이 ZIP_READY까지 자동 진행한다(§🎯 캠페인 단위 승인 모델). 이 per-command 명시승인 규칙은 이제 ①캠페인 미승인 상태 ②승인된 캠페인 범위를 벗어난 행동(유료GPU·삭제·외부kill·실제제출·코드/데이터/학습법 변경 등)에만 적용된다.**
- **파일 삭제 주의:** zsh는 변수 단어분리 안 함 → whitelist 변수로 삭제 금지(전삭제 사고 이력). find/명시적 배열 사용.

## 🖥️ 인프라 (상세·런치패턴은 HANDOFF.md §7)
- 치타(무료 A5000×2, 주력): `ssh -i ~/.ssh/cheetah.pem -p <CHEETAH_PORT> <REMOTE_USER>@<CHEETAH_IP>`. GRAD_CKPT=0(deberta계열 필수). 좀비 kill은 `pkill -9 -f colab_train_base2.py`.
- **★재현성 manifest 규칙(07-12 교정10 확정): 전체 printenv 덤프 금지(토큰·자격증명 유출 위험).** 대신 학습 스크립트가 **파싱한 effective config JSON**(`effective_config.json`)을 run 디렉터리에 자동 저장 — 20개 플래그 전부(미설정도 기본값 명시: MODEL_NAME/MAX_LEN/EPOCHS/BATCH/GRAD_ACCUM/LR/SEED/GRAD_CKPT/PRETEXT/PRETEXT_META/MASK_MINT_META/EXTRA_DATA/INIT_BACKBONE/COARSE_AUX/LABEL_SMOOTH/SESSION_EQUAL/AU_BOOST/SOFT_TARGET/SKIP_REFIT/REFIT_ONLY) + 코드·feat·데이터 SHA256 + torch/CUDA/transformers/sklearn 버전. colab_train_base2.py에 구현됨. REFIT_ONLY 등 변형 경로 산출물은 "계열 후보"로만 해석. ±0.001급 차이는 "노이즈" 금지 — "학습 draw·seed·처리 상호작용 범위"로 서술(서버는 결정론). matched 실험은 frozen 디렉터리(코드·데이터 SHA 고정)에서 실행.
- 엘리스(과금 H100): 사전허락 + 종료 필수.

## ✅ 검증 규약 (07-12 사용자 지시 — 강제)
- **★작업 유형별 체크리스트 `open/docs/checklists/`를 해당 작업 직전에 읽고 전 항목 점검한다**: 학습전=`10_pre_training` / OOF·증류=`20_oof_distill` / zip 생성전=`30_pre_zip` / zip 자체=`40_zip_self` / 후보별 대조=`50_per_candidate` / 제출후=`60_post_submit` / 관측오류 사전=`00_observed_errors`. 특정 작업 시 **해당 파일만** 읽으면 된다(전체 통독 불필요). `.claude/hooks/checklist_guard.py`(PreToolUse·Bash)가 명령 패턴별로 해당 파일 경로를 자동 주입한다.
- **★모든 항목을 통과한 결과만 "검증 완료"로 보고할 수 있다.** 미통과·미확인 항목이 하나라도 있으면 "검증 완료" 표현 금지 — 미통과 항목을 명시해 보고한다.
- 자동화 러너: `scratchpad/verify_zip.py`(크기·CRC·경로위생·zip↔staging 전수SHA·v49 diff 선언·비복사본·fp16/finite·kf/RoBERTa 계약·prep/블렌드 계약) + 5행 E2E(`open/data/test.jsonl`, seed참여 3.00·고전단독 0 확인).
- **★(07-13 교정) 훅은 강제가 아니라 안내다.** `checklist_guard.py`·`bg_hygiene.py`는 컨텍스트를 **주입만** 하고 체크리스트 실독·실패항목 무시를 **차단하지 않는다**. 실질 강제는 ①에이전트 규율 ②**내부 교차검증 PASS**(구 data-auditor, 세분화됨) ③(원하면) 훅을 `permissionDecision=deny` 차단형으로 격상. 문서의 "하네스 강제" 표현은 이 한계를 전제로 읽을 것.
- **★(07-14 갱신 — 07-13 "v49 전용" 문구는 낡음, 외부감사 지적으로 정정) `verify_zip.py`는 member-spec 레지스트리로 일반화됨.** 인자 `<zip> <staging> <ref-staging> <new멤버> <old멤버> <member_type>`, member_type ∈ {kf768:185,290,752 / s48:110,618,112 / dv3ko:134,679,552}. ref는 v49든 챔피언 staging이든 가능(★ref는 zip 경로가 아니라 **staging 디렉터리** — zip을 넣으면 오탐 FAIL). 단일멤버 교체는 전부 지원. **레지스트리 밖 신규 백본·다중멤버 동시교체는 미지원** → unverified/FAIL 명시하고 "검증 완료" 금지.
- **★(07-14) 하드링크 staging 규칙(cA~cF 오염 사고):** `cp -al` 클론 후 파일을 `open('w')`/`sed -i` 없이 in-place 편집하면 **공유 inode로 전 staging에 전파**된다. 클론 staging의 파일을 편집하기 전 반드시 하드링크 분리(cp→rm→mv 또는 shutil.copy 후 교체). 오염된 staging에서 재zip 금지 — zip 내부(`unzip -p`)가 진실.
- **★(07-13 교정) 5행 E2E는 기능 스모크이지 런타임 검증이 아니다.** T4 600초 통과는 별도 `runtime_status=measured|estimated|unverified`로 보고(서버 probe만 measured, A5000 로컬은 최대 estimated). 5행 통과를 시간마진 통과로 해석 금지.

## 🎯 캠페인 단위 승인 모델 (2026-07-14 오후 사용자 선호 변경 — per-command 승인 대체)
- **행동 승인의 단위 = campaign manifest(개별 명령 아님).** 사용자가 고정된 실험 캠페인을 **1회 명시 승인**하면, Claude는 승인 범위 안에서 **ZIP_READY까지 재승인 없이 자동 진행**한다. 정상 단계마다 승인을 다시 묻지 않는다.
- **★AUDIT_PASS ≠ 행동승인 분리는 유지.** 내부 교차검증 PASS = 진행 자격일 뿐, **campaign manifest 승인이 행동 승인**이다. 캠페인 미승인 상태에선 종전 per-command 명시승인 규칙(§핵심규칙 요약) 유지.
- **자동 상태기계:** `PLANNED → PREFLIGHT → GATE_RUNNING → BAND_EVAL → (PARKED | DEPLOY_REFIT) → PACKAGING → VERIFYING → ZIP_READY`. 정상 전이는 자동, 단계마다 재승인 없음.
- **사용자 개입 상태(여기서만 보고/질문):** ①**ZIP_READY** = zip 경로·SHA·크기·검증결과·사이트메모 보고(실제 제출은 사용자 직접) ②**PARKED** = 밴드 탈락 근거만 보고 ③**BLOCKED** = 계약위반·크래시·변경필요만 질문 ④**NEEDS_USER** = 캠페인 범위 벗어난 행동 필요 시.
- **진행 메시지:** 시작·gate 완료·ZIP_READY/BLOCKED만. **epoch별 알림 금지.** ETA·완료감시는 자동 관리(§완료감시 프로토콜).
- **캠페인 범위 밖 = 자동진행 금지(승인/질문/정지):** 실제 Dacon 제출(ZIP_READY에서 정지) · 외부 Codex 호출 · 유료 GPU · 기존 파일·다른 run 삭제 · 캠페인 밖 프로세스 kill · **코드/데이터/학습법 변경**(자동 수정 금지 → BLOCKED 정지). 캠페인 소유 프로세스의 **명백한 크래시/OOM은 최대 1회 재시작 허용**.
- **campaign manifest 필수 필드:** campaign_id · GPU/seed 배정 · frozen code/data/config SHA · Stage A/B 레시피 · 밴드 규칙 · 배포(슬롯 교체) 정의 · 포장/검증 범위 · 정지점(ZIP_READY) · 금지항목. **첫 적용 캠페인 = `mint2v2-dualseed-0714`**(§실험로그 2026-07-14 정책변경 항목).
- **근거:** 2026-07-14 사용자 지시 — gate·refit·포장마다 per-command 재승인은 과도. 한 번 승인한 캠페인은 ZIP_READY까지 자동.

## 🔀 감사 게이트 (2026-07-14 재설계 — "독립 감사"→"내부 교차검증", prepare/execute 분리)
- **★용어: Claude 내부 검토는 "독립 감사"가 아니라 "내부 교차검증(internal cross-check)"이다** — 실행자·검증자 모두 Claude 계열이라 오류 상관성 잔존. **외부 감사(Codex)** 만 진짜 외부 눈이며 **사용자 명시 승인 시에만** 실행. 계약·상태기계 상세: `open/coordination/README.md` + `open/coordination/agent_contracts.md`.
- **Claude 지휘자=실행자, 내부 교차검증 서브에이전트=read-only 판정.** 주요 전환점(plan_review·pre_launch·pre_submit·post_lb)에서 **교차검증 PASS 없이 다음 위험 단계로 넘어가지 않는다**(미세작업마다 호출 금지). **★단 PASS는 "진행 자격"일 뿐 — 발사·중단·kill·유료GPU·삭제·제출은 각각 별도 사용자 명시 승인**(`action_state=WAITING_USER`). safe_next_action은 권고문일 뿐 실행 권한 아님. **★(2026-07-14 오후 갱신) 행동 승인의 단위는 개별 명령이 아니라 campaign manifest다 — 사용자가 캠페인을 1회 명시 승인하면 그 범위 내 정상 단계전이(발사·gate·refit·포장)는 재승인 없이 자동(§🎯 캠페인 단위 승인 모델). AUDIT_PASS≠행동승인 분리는 유지: 교차검증 PASS=진행 자격, campaign manifest 승인=행동 승인.**
- **단일 data-auditor는 폐지·세분화**(계약: `open/coordination/agent_contracts.md`): data-lineage/split-leakage/training-contract/code-diff/metric/blend/integration/zip-verifier. 각 역할은 `context-curator` **최소 패킷 + 담당 체크리스트만** 읽는다(HANDOFF/실험로그 전체 통독 금지 — 컨텍스트 위생 #13).
- **역할 불변식:** ①Claude 지휘자만 파일 수정(감사자·역할에이전트는 read-only) ②감사자 출력은 판정뿐(무수정) ③실제 제출·유료GPU·삭제·원격kill은 **계속 사용자 승인** ④감사자 BLOCK 임의 무시 금지 ⑤동일 finding 2회 BLOCK→NEEDS_HUMAN 승격 ⑥감사자 실패/불확실→위험단계 fail-closed.
- **post_lb:** data-auditor PASS 이후에만 append-only 로그·HANDOFF 갱신. "노이즈"·"단독효과"·"순수 A/B" 표현이 증거보다 강하면 감사자가 BLOCK.
- **★(2026-07-14 재설계) 외부 감사(Codex)는 기본 PREPARE_ONLY·사용자 명시 승인 1건당 정확히 1회·자동 재시도 없음(오류·timeout·400이어도).** 준비는 `external-audit-packager`(=`run_codex_gate.py prepare`, **호출 안 함**), 실제 1회 실행은 사용자 명시 승인 후 **지휘자만** `run_codex_gate.py execute <prepared.json> --approve-request-id <rid> --approve-audit-key <key>`. 재시도·force-reaudit는 **새 승인** 필요. 인프라 실패는 verdict가 아니라 **INFRA_ERROR**로 기록(가짜 NEEDS_HUMAN 금지). **병행 필수 지점 = 새 학습축 plan_review·챔피언 교체급 판정·본선 재현성 동결.**(실사례: Claude 이중검증 4회 통과분을 07-14 외부 감사가 적발.) SSH키·pem·전체 printenv는 주지 않음. 일상 pre_launch/pre_submit/post_lb는 내부 교차검증 단독 유지.
- **★(2026-07-14 재설계) 감사 파일 스키마 준수:** 신규 기록은 **schema_version="3"** 필수, 결과마다 **짝 request를 먼저 저장**, v3 스키마(`schemas/audit_result.schema.json`·`infra_error.schema.json`) 필드·타입 준수, created_at/completed_at은 실제 시각. 검증기 `scratchpad/validate_audits.py`를 감사 저장 직후 1회 실행 — **legacy 판정은 mtime cutoff가 아니라 schema_version+backfilled+`legacy_manifest.json`으로만**(신규 위반이 mtime으로 숨지 않음, #6). 과거 위반분은 소급 수정 없이 legacy로 동결(실험로그 정정으로만 기록).
- 기존 zip 검증 하네스(`verify_zip.py`·체크리스트)는 재사용하되 **v49 계약 한계**(§검증규약)를 감사자가 확인 — 비-v49 조합은 unverified 명시. 중복 구현 금지.
- **★(07-13 사용자 지시, 2026-07-14 재설계 반영) 모든 내부 교차검증·외부 감사 판정은 파일로 영속화한다(구두/채팅만 금지).** `open/coordination/results/<stage>-<UTC>-<slug>.json`(스키마 `schemas/audit_result.schema.json`)에 최소: **objective(요청/목적)·verified_claims(주장+실제근거)·verdict·unverified_claims/confounders·auditor_model·completed_at(UTC)·artifact_sha(코드·데이터 SHA)**. 짝 request도 `requests/`에 저장. 감사 후 confounder를 뒤늦게 발견하면 `post_audit_correction` 필드로 추가(소급 삭제 금지). 근거: 07-13 mint2b plan_review PASS가 파일로 안 남고 Stage-A epoch confound를 놓친 사고.
- **★(07-13 사용자 지시) plan_review는 stage별 하이퍼파라미터를 matched-control 변수로 명시 점검한다.** 2단계(pretext→real-only)·다단계 학습에서 **각 stage의 epoch/LR/batch가 대조군(기존 pretext/real-only 런)과 일치하는지** 확인 — 공유 `COMMON` 변수로 전 stage에 같은 값을 뿌리면 조용한 confound가 된다(mint2b 최초 발사: stage A를 6ep로 돌렸으나 기존 pretext는 3ep). 데이터/처리 1변수만 바꾸고 나머지(stage별 epoch 포함)는 대조군과 일치시킬 것. 미일치면 "단독효과" 호칭 금지·"묶음 probe"로만 기록.

## 🔭 완료감시 프로토콜 (07-13 워처 유실 사고 후 확립 — 강제)
- **★원격 학습 완료감시는 재접속 폴링 워처만 쓴다(단일 장기 블로킹 SSH 금지).** 하나의 SSH 안에서 `while…sleep` 블로킹하면 연결이 조용히 끊길 때 완주를 못 잡는다 — **07-13 mint2b: 05:53 완주를 14:08까지 못 잡아 ~8h GPU 유휴·제출일정 대폭 지연.** 워처 = `run_in_background`로 **매 90~120s 짧은 새 SSH**를 반복해 각 run의 DONE 마커/체인 프로세스 수만 확인(연결 실패·빈 응답은 다음 폴에서 자동 재시도 = 네트워크 blip 내성). 상세·골격은 `.claude/agents/run-monitor.md`(완료감시 전담 — 발사는 remote-launcher와 분리).
- **워처 상한 도달 시 조용히 종료 금지** — "still running — 재부착 필요"를 출력해 지휘자가 재부착. 상한은 ETA×2+여유로 넉넉히.
- **★zsh 파싱 함정:** 워처는 `set -- $out`/공백분리로 폴 결과를 파싱하지 말 것 — **zsh는 변수 단어분리를 안 해** "1 1 0"이 통짜 $1로 들어가 완료조건이 영영 안 맞는다(07-13 워처가 이 버그로 완주를 못 잡음). SSH가 `MARK<d1>-<d2>-<proc>` 프리픽스를 echo하고 `grep -oE 'MARK[0-9-]+'` + `case "$out" in MARK1-1-*)`로 매칭한다(단어분리 무관).
- **★ETA 백스톱(워처 단독 신뢰 금지):** 발사 시 ETA를 명시하고, 워처 알림이 **ETA+여유까지 안 오면** 즉답형 foreground SSH로 직접 완주를 확인한다(로컬 bg 워처가 죽어도 완주를 놓치지 않게). *(주의: 치타 SSH는 로컬 pem 필요 → 클라우드 cron/routine은 pem 없어 하트비트 불가. 완료감시는 세션 내 재접속 워처 + ETA 백스톱으로만.)*
- **완료 즉시 수확**(effective_config·홀드아웃·산출물 회수) 후 **다음 GPU 작업을 즉시 배정**해 GPU 유휴를 최소화한다. GPU idle = 슬롯·일정 손실.
- 워처 교체 시 구 워처 즉시 TaskStop(§bg위생). 동시 한도 워처1+연산1.

## 🤝 서브에이전트 활용 (오케스트레이션 가이드 — 07-13 사용자 지시)
지휘자(Claude)는 **사용자 소통·전략·승인 상태·최종 조립·파일수정**을 하고, **장시간·전문 작업은 세분화된 17개 서브에이전트에 위임**한다(전체 역할·입출력·쓰기권한: `open/coordination/agent_contracts.md`). 한 줄 사실확인은 직접, 아래는 반드시 해당 에이전트로. **★모든 역할은 HANDOFF/실험로그 전체를 기본으로 읽지 않고 `context-curator` 최소 패킷 + 담당 체크리스트만 읽는다(컨텍스트 위생 #13).**
- **준비/증거:** context-curator(최소 패킷), experiment-planner(대조군·변경변수 설계), external-audit-packager(Codex용 준비=PREPARE_ONLY, 호출 금지).
- **내부 교차검증(read-only 판정):** data-lineage-auditor, split-leakage-auditor, training-contract-auditor, code-diff-auditor, metric-auditor, blend-evaluator(밴드 자격만), integration-auditor(상태전이·문서정합), zip-verifier.
- **실행/수확(부작용은 사용자 승인 게이트):** remote-launcher(**승인된 발사만**), run-monitor(**완료감시만**, 발사와 분리), artifact-harvester(회수), package-builder(**승인 구성** staging/zip).
- **해석/기록:** lb-interpreter(서버 delta·해석), docs-ledger-writer(**유일한 문서 writer** — HANDOFF 갱신 + 실험로그 EOF append).
- **병렬화 규율:** 워처1+연산1(bg위생). 독립 조사 다수는 Workflow(팬아웃)로 묶고 결과는 지휘자가 조립. 같은 파일 동시편집 금지(서로 다른 파일만 위임). 위임 기준 = 완료까지 수 분+ 걸리거나 격리 컨텍스트가 이로운 조사/감사.
