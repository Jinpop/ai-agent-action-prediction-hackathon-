# mDeBERTa-v3-base (실험 예정)

## 가설
large(337M)는 base(110M)를 못 이겼다(파라미터수≠답). 하지만 mDeBERTa-v3는 **RTD+disentangled attention**
으로 같은 크기서 roberta 상회하는 경우多 = 아키텍처 우위. 278M(fp16 556MB, 1GB내). 4형제 표현 개선 기대.

## 구성 / 배포 리스크
- MODEL_NAME=microsoft/mdeberta-v3-base, 공통레시피 8ep seed 61, batch16×accum3
- **토크나이저 리스크**: sentencepiece 필수(설치확인), DebertaV2TokenizerFast. 학습스크립트 토크나이저패치 조건부화 완료(deberta는 tokenizer_class 안건드림)
- 배포시 requirements에 sentencepiece 포함 필수

## 완료 후 기입: 홀드아웃, 4형제 F1, 배포 스텁 검증, 블렌드 편입 여부
