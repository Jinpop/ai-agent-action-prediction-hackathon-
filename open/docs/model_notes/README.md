# 모델 노트 (Dacon 236694)

각 학습 런/모델의 완전한 기록. **이 폴더만 읽어도 무엇을 어떻게 했는지 재현 가능해야 한다.**

## 파일 목록

| 노트 | 모델 | 홀드아웃 | 비고 |
|---|---|:---:|---|
| [classic_v7.md](classic_v7.md) | 고전 3모델 (HGB/LogReg/NB) | 0.6089 | 모든 블렌드의 고정 멤버 |
| [hybrid_v10.md](hybrid_v10.md) | RoBERTa-base 하이브리드 1세대 | 0.6519 | [STATE] 직렬화 (구형) |
| [base_v12.md](base_v12.md) | base 풀트랜스크립트 | 0.7031 | [SRC] 없음, 384토큰 |
| [large_v14.md](large_v14.md) | RoBERTa-large | 0.7129 | [SRC] 없음 — 서버에서 약함 |
| [base_seed44.md](base_seed44.md) | base [SRC]+512, 5ep | 0.7440 | 표준 레시피 확립 |
| [base_seed45.md](base_seed45.md) | 〃 6ep | 0.7467 | |
| [base_seed46.md](base_seed46.md) | 〃 7ep | 0.7449 | 치타서버 1호 |
| [base_seed47.md](base_seed47.md) | 〃 8ep | 0.7481 | 단일 최고 |
| [base_seed48.md](base_seed48.md) | 〃 6ep | 0.7482 | s47과 동률 |
| [large_src_seed50.md](large_src_seed50.md) | **[SRC]-large** 5ep | 0.7450 | 다양성 기여 미미(+0.0006), v23b로 최종판정 |
| [base_seed51.md](base_seed51.md) | 〃 6ep | **0.7506** | ★단독 신기록 → v25 |
| [bert_seed52.md](bert_seed52.md) | **klue/bert-base** 6ep | (학습 중) | 백본 다양성 실험 |
| [sibling_specialist.md](sibling_specialist.md) | 4형제 전용 4-way s60 | 0.599(단독) | **★통합 +0.0124 → v24** |
| [mdeberta_v3.md](mdeberta_v3.md) | mDeBERTa-v3-base | 0.712@6ep | 기각(roberta보다 약함) |
| [self_distillation.md](self_distillation.md) | **자기증류 d90** | **0.7585** | ★성공, 학생3 확장중 |
| [au_boost.md](au_boost.md) | **au×3 가중** (se56au) | 0.7464(au 0.926) | ★반전보너스 공략 → v27 |
| [experiments_negative.md](experiments_negative.md) | (기각 실험 모음) | — | 피처 재랭커 실패·large 무효 |

## 공통 레시피 (v13 이후 표준 — 개별 노트는 차이만 기록)

### 입력 직렬화 (풀 트랜스크립트)
세션을 대화록 그대로 텍스트화. 학습·추론이 반드시 동일해야 함:
```
[SRC] au|sim                     ← 세션 집단 (sess_au_* 여부; seed44+부터)
[U] 사용자 발화...
[A] 행동이름 args키=값... -> result_summary
... (교차 반복)
[META] tier=.. ci=.. dirty=.. turn=.. open=파일들
[P] 현재 프롬프트
```
- 토크나이저 `truncation_side="left"` — 긴 세션에서 **오래된 history부터** 자르고 [META]/[P] 보존 (이거 안 하면 붕괴: v12 1차 0.5459 사고)

### 아키텍처 (HybridNet)
- 백본: klue/roberta-base(110M) 또는 -large(337M), HuggingFace
- CLS 벡터(768/1024d) ⊕ **feat.py 메타피처 119d**(StandardScaler) → Linear(→256)+GELU+Dropout(0.1)+Linear(→14)
- 손실: sqrt-balanced 가중 CE (희귀클래스 과보정 방지 — balanced는 web_search 붕괴 유발)

### 학습 설정 (기본값)
- MAX_LEN 512, LR 2e-5(base)/1e-5(large), warmup 10%, weight decay 0.01, fp16
- 실효 배치 48 (batch×grad_accum으로 맞춤; GPU 메모리에 따라 조정)
- 홀드아웃: GroupKFold(5) 첫 분할, **세션 단위** (id의 "-step" 앞부분이 그룹)
- 파이프라인: 80%로 홀드아웃 학습·평가 → 100%로 refit(제출용) → zip 포장
- 스크립트: `open/scripts/colab_train_base2.py` (env: MODEL_NAME/EPOCHS/SEED/BATCH/GRAD_ACCUM/LR/GRAD_CKPT/SKIP_REFIT)

### 산출물 (런마다)
- `holdout_probs3.npy` (홀드아웃 로짓, **ACTIONS 순서**) + `holdout_idx3.npy` (행 인덱스)
- `submit_base2.zip` (refit 모델 + 단독 제출용 script.py)
- 로컬 보관: probs→`open/artifacts/hybrid/holdout_probs_s{seed}.npy`, zip→`submits/submit_base2_s{seed}.zip`

### ⚠️ 분할 호환성 (중요)
sklearn 버전에 따라 GroupKFold 분할이 다름:
- Kaggle/Colab(최신 sklearn) 분할 = 서로 동일 (s44, s45, v12, v14, large가 이 분할)
- **치타서버 분할 = 다름** (s46, s47, s48, s50) — 자체 idx로 채점해야 하며, 구분할과는 교집합(~2.8k행)으로만 비교 가능
- 로컬 sklearn 1.6.1 = 또 다름 (고전 확률 캐시는 Kaggle 분할 기준으로 재계산돼 있음)

### 배포(제출) 공통 규칙
- 서버: T4 16GB, 3vCPU/12GB RAM, 스크립트 10분(설치 별도 10분), zip ≤1GB, 오프라인
- **`net.half()` 명시 필수** (from_pretrained는 fp16 저장본도 fp32로 업캐스트 → 3배 느려짐)
- fp16 저장으로 용량 절반 (base 443→213MB)
- 길이정렬 배칭 + pad_to_multiple_of=8 + 데드라인 캐스케이드 (v15 시간초과 3연속의 교훈)
- requirements 전 버전 고정 (numpy==2.0.2 등 — pickle 생성 환경과 일치)

## 블렌드 계보 (제출 조합)

| 제출 | 조합 | LB |
|---|---|:---:|
| v21c | 0.6×평균(s44,s45) + 0.4×고전블렌드(0.45/0.40/0.15) | 0.7728 |
| v23a | 0.6×평균(s45,s46,s47,s48) + 0.4×고전 | **0.7759** |
| v24 | v23d + 4형제 specialist 재결정(w=1.0) | 0.7697 ❌(-0.0035 vs v23d) |
| v25 | 0.6×평균(s46,s47,s48,**s51**) + 0.4×고전 | 0.7748 (v23a-0.0011, 시드교체 무익 확정) |
| v26 | 고전+자기증류 학생4 | 0.7641 ❌ 증류 서버반납 |
| v27 | 고전+s47/s48/s51+au×3 | 0.7752 (au 동률권 — 구조 포화 확정) |
| **v22** | 0.6×평균(s45,s46,s47) + 0.4×고전블렌드 | **0.7746** |

**결합 법칙 (LB 실측으로 확립)**: 결합기를 홀드아웃에 맞출수록 서버 점수가 깎인다
(v21b 강튜닝 CV 0.7661→LB 0.7714 < v21c 무튜닝 CV 0.7551→LB 0.7728).
→ 메타러너 대신 **고정 가중 순수 블렌드** 사용. 근거: 서버 테스트는 세션당 1 step
(초반 스텝·au 비중 높음)이라 홀드아웃과 분포가 다르고, [SRC]+장기학습 모델이 그 분포에 유리.
