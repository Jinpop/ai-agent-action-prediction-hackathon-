# kf-deberta 계열 (★아키텍처 다양성 축)

## 판정 (2026-07-08)
- kakaobank/kf-deberta-base (185M, 한국어 전용 DeBERTa-v2, 상대위치)
- seed70 6ep(민트포함, LR 1.5e-5): **홀드아웃 0.7470** — 로베르타 동급
- 4시드 앙상블과 불일치율 8.6% (시드끼리 11.3%, 증류학생 5.3%)
- **LB 검증: v29(고전+s48+s51+kf70) 0.77567 — 같은 3멤버 로베르타 트리오(0.7732) 대비 +0.0025**
- 참고: mdeberta(다국어)·bert·electra는 전부 기각 — 한국어 전용 + DeBERTa 구조 조합만 생존

## 운영 노트
- fp16 370MB (로베르타 222MB의 1.7배) → zip 조합수학 주의: 고전+로베르타2+kf=829MB ✓ / 고전+로베르타3+kf=1.09GB ✗
- 상대위치 임베딩: extend_positions는 None 가드로 스킵 (v2026-07-08 수정)
- 추론 ~1.4배 무거움 → SEED_DIRS 마지막(최후 데드라인) 배치. v29 실측 8m4s 완주
- 토크나이저: BertTokenizerFast (wordpiece — sentencepiece 불필요). 신버전 transformers의 regex 경고는 무해(서버 4.57.6)

## 산출물
- hybrid/holdout_probs_kf70.npy (치타분할), submits/submit_kf70.zip(80%판)
- kf71(풀런+refit): 홀드아웃 0.7462 → **v30(고전+s48+s51+kf71) LB 0.7774001002 신기록(+0.0015)**. submits/submit_kf71.zip
- kf72/kf73: 밤샘 증산 중 (내일 v31 재료)
