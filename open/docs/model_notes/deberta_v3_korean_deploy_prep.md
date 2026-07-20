# 신규 백본 deberta-v3-base-korean 배포준비 (카드④, 07-13 prep — 학습 미발사)

> **대상:** `team-lucid/deberta-v3-base-korean`. **배포형태:** v58 프레임(`classic + s48 + kf768_79 + kf768_81`)에서 **kf768 슬롯 1개를 dv3ko로 교체**(2seed는 kf768 2슬롯). **작성:** 2026-07-13 02:2x KST. 근거: 백본 카나리(치타 CPU 실측) + 3-agent 배포준비 워크플로우 + verify_zip 회귀테스트.
> **표현 규율:** envelope/파이프라인 호환/verify_zip 회귀 = **measured/코드대조**. 성능·T4 runtime·SPM 라운드트립 = **estimated/unverified**. dv3ko 조합은 실 멤버 빌드 전까지 **"검증 완료" 선언 금지**.

## 1. 백본 계약 (measured — 치타 CPU 카나리)
| 항목 | 값 | 비고 |
|---|---|---|
| model_type | **deberta-v2** | kf-deberta와 동일 계열(유일 생존 §5). 한국어 deberta-v3 사전학습 = 다른 오류집합 잠재 |
| hidden_size | **768** | = kf-deberta → HybridNet(CLS 768d ⊕ meta 119d → Linear(887→256→14)) **무변경 호환** |
| params | **134,679,552** | kf768(185,290,752)보다 작음. 차 50,611,200 = (130000−64100)×768 = **정확히 vocab 임베딩 테이블 차** → 인코더 FLOPs 동일 |
| vocab_size | 64100 | (kf 130000) |
| max_position / pos_buckets / max_rel | 512 / 256 / −1 | kf-deberta와 **relative-attention config 완전 동일**. 챔피언이 kf를 768로 돌리는 그 메커니즘 |
| tokenizer | **DebertaV2TokenizerFast** (is_fast) | 캐시에 `tokenizer.json + spm.model + tokenizer_config.json` 전부 존재 |
| 캐시 | 치타 `~/.cache/huggingface/hub/models--team-lucid--deberta-v3-base-korean` 518M, **model.safetensors 단일파일**(bin 변환 불요) | 오프라인 준비됨 |
| **768-token forward 카나리** | ✅ **OK** (CPU, last_hidden (1,768,768), CLS dim 768, 무에러) | max_pos 512 우려 해소(relative attention) |
| has_pooler | False | HybridNet은 `last_hidden_state[:,0]`(CLS) 직접 사용 → 무관 |

## 2. 배포 envelope (measured 기반 추정)
- v58 zip = 978,557,947B/1GB decimal, kf768 슬롯 압축 ~343.7MB. dv3ko 슬롯 fp16 backbone ~269MB(+head+tokenizer 압축 ~251MB 추정).
- **1슬롯 교체 zip ≈ 886MB**(여유 ~114MB) / **2슬롯 교체(2seed) ≈ 794MB**(여유 ~206MB). 둘 다 < 1GB decimal. tokenizer ±6MB 불확실폭에도 판정 불변.
- **★A/B/C와 정반대:** dv3ko는 kf768보다 **작아 zip이 줄어든다**(A/B/C는 s48(512)→kf768(768) 확대로 +116MB envelope 위반). **반드시 kf768 슬롯만 교체**(s48 슬롯 교체 시 3×768 → 시간·용량 붕괴). **4번째 멤버 추가 금지**(마진 14s).

## 3. 파이프라인 호환 (코드 대조 확인 — 무코드변경)
- `conv_fp16.py`: `AutoModel.from_pretrained(p).half().save_pretrained()` — 아키텍처 무관 제네릭 ✅
- `colab_train_base2.py` extend_positions: DeBERTa는 `position_embeddings=None`→즉시 return(no-op) ✅
- wordpiece 패치: `deberta`면 스킵 → spm.model/tokenizer.json 네이티브 저장 ✅
- 추론 script.py: 백본/tokenizer/head/meta/actions 전부 `./model_X/backbone`에서 데이터드리븐 로드 ✅. `net.half()` fp16 계약 ✅
- **학습 env만 변경:** `MODEL_NAME=team-lucid/deberta-v3-base-korean MAX_LEN=768 EPOCHS=6 BATCH=16 GRAD_ACCUM=3 LR=2e-5 GRAD_CKPT=0 HOLDOUT_IDX=data/hidx.npy SEED=<n>`. (deberta-v3는 768서 GRAD_CKPT=0 필수 — backward-twice 크래시)

## 4. verify_zip 일반화 (구현 완료 — CL-1)
- `scratchpad/verify_zip.py`를 **member-spec 레지스트리**로 일반화: dirname 'kf768' 추측 제거, `MEMBER_SPECS={kf768,s48,dv3ko}` + 6번째 인자(또는 dirname 자동탐지)로 spec 선택.
- 해소된 하드코딩: **max_len assert(구 hard break: dv3ko dir→512 기대 vs 768 실제 FAIL)**, param count(구 else-branch서 미검증·'RoBERTa' 오라벨 → 항상 assert, **s48 param 구멍도 동시 폐색**), head 887→`hidden+119` 유도. 신규 게이트: **backbone config(model_type·hidden·vocab)** + **SEED_DIRS 스왑 정합**(신규멤버 포함·구멤버 제거 — 무증상 폴백 방지).
- **회귀 실증:** 일반화판을 **v58 zip에 실행 → 27검사 ALL PASS**(kf768 자동탐지, 신규 체크 전부 통과 = 하위호환 + 강화 실증). ref는 v49→**직전 프레임(v58)** 로 호출: `verify_zip.py <zip> <staging> scratchpad/stage_v58 model_dv3ko model_kf768_81 dv3ko`.
- **⚠️미실증:** dv3ko spec은 유닛테스트만(실 dv3ko zip 없음) → 멤버 빌드 시 E2E. **T4 런타임·SPM 라운드트립은 verify_zip 범위 밖**(여전히 unverified).

## 5. 카나리 게이트 + pass 기준
| # | 게이트 | pass 기준 | 성격 |
|---|---|---|---|
| ① | zip < 1GB | 1슬롯 ≈886MB / 2seed ≈794MB | **measured**(v58 실측 − kf압축 + dv3ko추정) |
| ② | net.half() fp16 | backbone safetensors 전텐서 fp16 + finite | 로컬 hard-pass |
| ③ | fast tokenizer + E2E | is_fast==True + 5행 E2E dv3ko 참여 무에러 | 로컬 hard-pass (단 SPM 라운드트립은 ④ 참조) |
| ④ | T4 600초 | 완주 ≤600s. estimated ≈9:46/마진 ~14s(인코더=kf768 동일, FLOPs 동일) | **estimated only — 서버 제출만 measured** |

## 6. unverified / 리스크 (실측 전 미해소)
- **T4 600초 실측**: `runtime_status=estimated`(A5000 measured 불가). 서버 제출 1회 = 유일 measured. 마진 14s draw별 ±수초.
- **★SPM 토크나이저 라운드트립**: 현행 챔피언 멤버는 전부 WordPiece(vocab.txt) — dv3ko의 DebertaV2TokenizerFast save→오프라인 reload가 이 배포 파이프라인에서 미실증. 캐시에 tokenizer.json+spm.model 존재로 리스크↓, 단 실 train→save→reload E2E로 확인 필요.
- **dv3ko 멤버 부재**: 학습 미수행. 배포팩·홀드아웃·밴드통과 미측정.
- **우열·채택은 서버 LB로만**(홀드아웃 밴드 = 자격필터일 뿐).

## 7. 다음 실행 (GPU는 mint2 완주 ~05:00 KST 후 free)
1. **data-auditor pre_launch 게이트** (학습 발사 전 필수 PASS).
2. **1seed 학습**(mint2 free 후): 위 §3 env. 학습시각+ETA 명시, train-launcher 워처.
3. 카나리 ①②③ 로컬 hard-pass → hidx-pin 홀드아웃 밴드체크(무료 스크리닝).
4. 밴드 통과 시 → SEED_DIRS 편집(script.py 라인63 kf768 슬롯 1개→model_dv3ko) + zip 빌드 + 일반화 verify_zip(type=dv3ko) + 5행 E2E(SPM 라운드트립 확인).
5. **data-auditor pre_submit** → 사용자 승인 → 제출(④ measured + 실 LB). 통과 시에만 2seed.
