# Dacon 236694 — AI 코딩 에이전트 다음 행동 예측 (예선)

AI 코딩 에이전트의 세션 로그(대화 history + 세션 메타)를 보고 **다음 행동 14클래스**를 예측하는 대회.
Macro-F1 · 코드 제출형(script.py가 숨겨진 test 서버에서 실행 · T4 16GB · 추론 600초 · zip 1GB · 오프라인).

이 저장소는 두 가지를 담고 있다:
1. **모델링 결과** — 최종 LB 0.7867의 앙상블과, 서버 실측으로 확립한 법칙들
2. **운영 하네스** — AI 에이전트(Claude)가 실험을 자율 수행하도록 직접 설계·진화시킨 승인·검증·기록 체계 (사고가 날 때마다 규칙으로 편입시킨 과정 전체가 원장에 남아 있음)

---

## 1. 최종 결과 (2026-07-15 예선 종료)

| | |
|---|---|
| **최종 스코어 (LB)** | **0.7866974871** (시작 0.7809 → +0.0058) |
| 최종 구성 | 고전 ML 0.38 + [신경망 3멤버 균등평균] 0.62 |
| 본선 컷 | 0.79591 — 미달(−0.0092)로 예선 종료 |

**최종 앙상블 (metaT_w062):**
- **고전 멤버**: TF-IDF(word+char) → HGB + LogisticRegression + ComplementNB 확률 블렌드
- **신경망 3멤버** (전부 HybridNet: 백본 CLS ⊕ 119d 숫자메타 → 256 → 14):
  - `s48` — RoBERTa 계열 512-len (탈상관 다양성 기여 최대: LOO +0.011)
  - `kf768_81` — kf-deberta-base 768-len, direct-mint 증강 (레시피 다양성)
  - `kf768_metaT` — kf-deberta-base 768-len + **transcript-meta**(공식 세션메타를 transcript에 직렬화, 최대 단일 개선 +0.0018) + mint pretext 2단계 학습

---

## 2. 운영 하네스 — AI 에이전트 자율 실험 체계

이 프로젝트의 실험 실행자는 Claude 에이전트였고, 사용자는 **승인·전략·제출**만 담당했다.
그 협업이 안전하게 돌아가도록 만든 장치들이 이 저장소의 절반이다. 핵심 설계 원칙: **"사고 → 규칙 → 자동 강제"** 피드백 루프.

### 2.1 역할 구조 — 지휘자 + 17개 서브에이전트

```
사용자 (승인·전략·제출)
  └─ 지휘자 Claude (소통·상태기계·최종조립 — 유일한 파일 수정권자)
       ├─ 준비/증거:   context-curator(최소 증거패킷) · experiment-planner(대조군 설계)
       ├─ 내부 교차검증(read-only 판정만):
       │    code-diff-auditor · data-lineage-auditor · training-contract-auditor
       │    split-leakage-auditor · metric-auditor · blend-evaluator
       │    integration-auditor · zip-verifier
       ├─ 실행/수확:   remote-launcher(승인된 발사만) · run-monitor(완료감시 전담)
       │              artifact-harvester(회수) · package-builder(승인 구성 포장)
       ├─ 해석/기록:   lb-interpreter(LB delta·confounder 해석)
       │              docs-ledger-writer(★유일한 문서 writer)
       └─ 외부 감사:   external-audit-packager(Codex용 준비만 — 호출은 사용자 승인 후 지휘자만)
```
역할 계약 전문: [open/coordination/agent_contracts.md](open/coordination/agent_contracts.md)

**불변식**: 감사자는 판정만(무수정) · 문서는 docs-ledger-writer만 · 발사는 승인된 것만 · 각 역할은 전체 문서를 읽지 않고 최소 증거 패킷만 읽음(컨텍스트 위생).

### 2.2 캠페인 승인 모델 — 명령 단위가 아니라 캠페인 단위

초기엔 발사·refit·포장마다 사용자 승인을 받았으나(과도), **campaign manifest 1회 승인** 모델로 진화:

```
PLANNED → PREFLIGHT → GATE_RUNNING → BAND_EVAL → DEPLOY_REFIT → PACKAGING → VERIFYING → ZIP_READY
```
- manifest에 GPU 배정·frozen SHA·레시피·포장 정의·**금지항목**을 동결하면 위 전이는 재승인 없이 자동
- **항상 사용자 게이트로 남는 것**: 실제 제출 · 유료 GPU · 삭제 · 외부(Codex) 감사 호출 · 캠페인 밖 코드/데이터/학습법 변경
- **AUDIT_PASS ≠ 행동 승인** — 교차검증 통과는 "진행 자격"일 뿐, 행동 권한은 manifest 승인에서만 나옴
- manifest 실물: [open/coordination/campaign-*.json](open/coordination/)

### 2.3 검증 스택 — 4개 층위

| 층위 | 장치 | 역할 |
|---|---|---|
| ① 절차 | **체크리스트 7종** ([open/docs/checklists/](open/docs/checklists/)) + 훅 자동주입([.claude/hooks/checklist_guard.py](.claude/hooks/checklist_guard.py)) | 학습전/OOF/zip 생성·자체/후보대조/제출후 — 전 항목 통과 없이는 "검증 완료" 보고 금지 |
| ② 자동 러너 | **[verify_zip.py](scratchpad/verify_zip.py)** + 5행 E2E | zip 크기·CRC·경로위생·zip↔staging 전수 SHA·fp16/finite·head 차원 계약·member-spec 레지스트리 + 실제 추론 스모크(시드참여 수·고전단독 0) |
| ③ 내부 교차검증 | read-only 감사 서브에이전트군 | 주요 전환점(plan_review/pre_launch/pre_submit/post_lb)마다 판정. **실제로 발사 전 결함 2회 차단**: 가중 플래그(TARGET_BALANCE) 누락 BLOCK, set -e 부재로 생긴 불량 zip 적발 |
| ④ 외부 감사 | Codex (prepare/execute 분리) | 준비는 자동, 실행은 사용자 명시 승인 1건당 정확히 1회. Claude-Claude 오류 상관성을 외부 눈으로 보완 — 실제로 내부검증 4회 통과분의 결함을 적발한 사례 있음 |

### 2.4 재현성·기록 장치

- **effective_config 자동 기록**: 학습 스크립트가 파싱한 전체 플래그(미설정 기본값 포함) + 코드·데이터 SHA256 + 라이브러리 버전을 run마다 JSON으로 남김 (printenv 덤프 금지 — 토큰 유출 방지)
- **append-only 실험로그** ([open/docs/실험로그.md](open/docs/실험로그.md)): 모든 제출·오류발견·정정·negative 결과를 발생 시각의 새 항목으로만 기록. 과거 소급 수정 금지 — 잘못된 해석도 "정정 항목"으로 남김
- **감사 판정 영속화** ([open/coordination/results/](open/coordination/results/)): 구두 판정 금지, 스키마 v3 JSON(objective·verified_claims·confounders·artifact_sha)으로 저장 + [validate_audits.py](scratchpad/validate_audits.py) 스키마 검증
- **matched-control 규율**: 축 종결 선언은 동일 시드·동일 split·1변수 변경 대조군이 있을 때만. 다변수 변경은 "묶음 probe"로만 호칭("단독효과" 금지)
- **HANDOFF 단일 진실 소스** ([open/docs/HANDOFF.md](open/docs/HANDOFF.md)): 챔피언·법칙·죽은축/살아있는축·인프라 — 세션이 바뀌어도 여기서 이어짐

### 2.5 완료감시 프로토콜 (원격 GPU)

- **재접속 폴링 워처**: 90~120초마다 짧은 새 SSH로 DONE 마커만 확인(네트워크 blip 자동 내성)
- **ETA 백스톱**: 발사 시 ETA 명시, ETA+여유까지 알림 없으면 워처를 불신하고 직접 확인
- 완료 즉시 수확(fp16 변환→SHA 대조→로컬 회수)→다음 작업 배정으로 GPU 유휴 최소화

### 2.6 하네스 도입 연대기 (사고가 규칙이 된 순서)

| 날짜 | 도입 | 계기 |
|---|---|---|
| 07-10 | 외부 독립 감사 · matched-control 없인 축 종결 금지 · append-only 원장 | 보고 신뢰성 의심 → 다수 "죽은축"이 구현결함으로 판명 |
| 07-11 | 백그라운드 위생(워처 정리·동시한도) · 홀드아웃=자격필터 재교정 | 홀드아웃 서열로 학습 중단·폐기한 실수 |
| 07-12 | 체크리스트 7종 강제 · verify_zip 자동 러너 · effective_config 재현성 | 검증 누락·재현 불가 런 |
| 07-13 | 완료감시 프로토콜 · 감사 판정 파일 영속화 · stage별 matched-control · 훅 자동주입 | 워처 유실 8h 사고 · epoch confound 사고 · 구두 판정 유실 |
| 07-14 | 17개 역할 세분화 · 외부감사 prepare/execute 분리 · 감사 스키마 v3 · **캠페인 승인 모델** | 단일 감사자 한계 · per-command 승인 과부하 |
| 07-15 | (실전 총합) 3캠페인 자동 완주 · 발사 전 결함 2회 차단 · 신규 사고 3건 즉시 규칙화 | 마지막 밤 — H100 터널 장애를 A100 전환으로 돌파 |

상세는 실험로그의 "하네스 도입 연대기" 항목 및 각 날짜 항목 참조.

---

## 3. 저장소 구조

```
├── README.md
├── CLAUDE.md / AGENTS.md       # 에이전트 하네스 규칙 전문 (메타규칙·검증규약·승인모델·감시 프로토콜)
├── open/
│   ├── docs/
│   │   ├── HANDOFF.md           # ★단일 진실 소스 — 챔피언·법칙·죽은축/살아있는축·인프라
│   │   ├── 실험로그.md           # ★append-only 시간순 원장 — 전 제출·실측·정정·연대기
│   │   ├── checklists/          # 작업 유형별 강제 검증 체크리스트 7종
│   │   └── model_notes/         # 백본별 배포 검토 노트
│   ├── scripts/
│   │   ├── colab_train_base2.py # 중앙 학습 스크립트 (HybridNet·2단계 pretext·effective_config 자동기록)
│   │   ├── feat.py              # 피처라이저 (transcript 직렬화·119/125d 메타)
│   │   ├── train.py             # 고전 3모델 학습
│   │   └── mint_data.py         # history 민팅(증강)
│   └── coordination/            # 감사 하네스 원장
│       ├── agent_contracts.md   # 17개 서브에이전트 역할 계약
│       ├── schemas/             # 감사 결과 JSON 스키마 (v3)
│       ├── results/ requests/   # 전 감사 판정 영속화 (pre_launch/pre_submit/post_lb/plan_review)
│       └── campaign-*.json      # 캠페인 manifest (승인 단위)
├── scratchpad/                  # 실험 도구·증거
│   ├── verify_zip.py            # 제출 zip 자동 검증 러너
│   ├── validate_audits.py       # 감사 JSON 스키마 검증기
│   ├── build_*.py               # 증강 데이터 빌더 (mint balanced v2·allinfo stable 등)
│   ├── *_band.py                # 홀드아웃 밴드(제출자격) 계산기
│   ├── allinfo_{chain,launch,setup}.sh · launch_h100.sh   # 원격 GPU 학습 체인/발사기
│   └── watch_*.sh               # 재접속 폴링 완료감시 워처
└── .claude/
    ├── agents/                  # 서브에이전트 정의 (감사자·실행자·기록자)
    └── hooks/                   # checklist_guard(체크리스트 자동주입) · bg_hygiene(백그라운드 위생)
```

> 대회 데이터(`open/data/`)와 학습된 모델 아티팩트는 라이선스·용량 문제로 저장소에서 제외(.gitignore). 원격 서버 주소·계정 등은 `<CHEETAH_IP>` 류 플레이스홀더로 마스킹됨.

## 4. 읽는 순서 (추천)

1. [open/docs/HANDOFF.md](open/docs/HANDOFF.md) — 최종 상태·법칙 요약
2. [open/docs/실험로그.md](open/docs/실험로그.md) — 전체 서사 (제출 스코어보드 → 시간순 실험·정정·사고·연대기)
3. [CLAUDE.md](CLAUDE.md) — 하네스 규칙이 실제로 어떻게 강제됐는지
4. [open/coordination/](open/coordination/) — 감사 판정 원본 JSON들
