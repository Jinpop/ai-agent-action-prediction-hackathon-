#!/bin/zsh
# 격리 테스트 전용 Codex 에뮬레이터 (2026-07-14 재설계 — raw 스키마 audit_result.codex.schema.json 준수).
# `codex exec ... --output-last-message <raw_path> ...` 형태를 파싱해 request_id를 반향하는 raw JSON을 기록.
# 실제 Codex 대신 CODEX_GATE_BIN로 주입해 게이트 파이프라인을 결정론적으로 검증한다.
#
# FAKE_SCENARIO ∈ {pass, block, invalid, timeout, modify, exit1, http400} (기본 pass)
#   pass    : verdict=PASS. safe_next_action="launch"(행동 승인과 분리 테스트용).
#   block   : verdict=BLOCK + blocking_finding(id=FAKE_FID, 기본 demo_finding).
#   invalid : 스키마 위반 raw(verdict enum 밖 + objective 누락) — result_schema_invalid 경로 검증.
#   timeout : FAKE_SLEEP(기본 30초) 대기 — 러너 timeout(작은 CODEX_GATE_TIMEOUT)에 걸림.
#   modify  : read-only 위반 모의 — scratchpad/_codex_added_test.py 생성 후 PASS raw.
#   exit1   : 비정상 종료(exit 1). result 미생성.
#   http400 : stderr에 "400" 남기고 exit 1 — codex_http_error 분류 검증.
# FAKE_RID 로 request_id 강제(stale/rid_mismatch 테스트).
# FAKE_CALL_LOG 설정 시 매 실제 기동을 그 파일에 1줄 append(subprocess 호출 횟수 카운트).
set -e
OUT=""; prev=""
for a in "$@"; do
  [ "$prev" = "--output-last-message" ] && OUT="$a"
  prev="$a"
done
# ★실제 기동 카운트: 어떤 시나리오든 프로세스가 뜨면 1줄 기록(재시도 0회 검증용).
[ -n "$FAKE_CALL_LOG" ] && echo "call ${FAKE_SCENARIO:-pass} $(basename ${OUT:-none})" >> "$FAKE_CALL_LOG"
[ -z "$OUT" ] && { echo "fake_codex: --output-last-message 없음" >&2; exit 2; }

RID=$(basename "$OUT" .json)
[ -n "$FAKE_RID" ] && RID="$FAKE_RID"
SCEN="${FAKE_SCENARIO:-pass}"
FID="${FAKE_FID:-demo_finding}"
SHA="1111111111111111111111111111111111111111111111111111111111111111"
REPO="/Users/<USER>/Documents/Dacon_236694_AI_Agent"

# codex --json 이벤트 모의(러너 stdout→jsonl). token usage 미포함(→ ledger tokens=null).
echo '{"type":"item.started","item":{"type":"reasoning"}}'
echo '{"type":"item.completed","item":{"type":"agent_message"}}'

case "$SCEN" in
  exit1)   echo "fake: simulated crash" >&2; exit 1 ;;
  http400) echo "stream error: unexpected status 400 Bad Request" >&2; exit 1 ;;
  timeout) sleep "${FAKE_SLEEP:-30}" ;;
  modify)  echo "def _injected(): pass" > "$REPO/scratchpad/_codex_added_test.py" ;;
esac

if [ "$SCEN" = "invalid" ]; then
  # 유효 JSON이지만 스키마 위반(verdict enum 밖 + objective 누락) → result_schema_invalid
  cat > "$OUT" <<JSON
{"request_id":"$RID","verdict":"MAYBE","auditor_model":"fake-codex",
"completed_at":"2026-07-14T00:00:00Z","blocking_findings":[],"nonblocking_findings":[],
"verified_claims":[],"unverified_claims":[],"confounders":[],"required_fixes":[],
"safe_next_action":"none","evidence_references":[],"artifact_sha":[]}
JSON
  exit 0
fi

if [ "$SCEN" = "block" ]; then
  BF="[{\"id\":\"$FID\",\"summary\":\"fake BLOCK finding\",\"evidence\":\"fake evidence\"}]"
  VERDICT="BLOCK"; RF="[\"fake 수정지시\"]"; SNA="fix then re-audit"
else
  BF="[]"; VERDICT="PASS"; RF="[]"; SNA="launch"
fi

cat > "$OUT" <<JSON
{"request_id":"$RID","objective":"fake audit objective","verdict":"$VERDICT",
"auditor_model":"fake-codex","completed_at":"2026-07-14T00:00:00Z",
"blocking_findings":$BF,"nonblocking_findings":[],
"verified_claims":["fake verified"],"unverified_claims":[],"confounders":[],
"required_fixes":$RF,"safe_next_action":"$SNA","evidence_references":["AGENTS.md"],
"artifact_sha":[{"path":"AGENTS.md","sha256":"$SHA"}]}
JSON
exit 0
