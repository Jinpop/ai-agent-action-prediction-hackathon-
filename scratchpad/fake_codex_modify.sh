#!/bin/zsh
# (2026-07-14 재설계) 하위호환 alias: modify 시나리오는 fake_codex.sh FAKE_SCENARIO=modify로 통합.
# read-only 위반(파일수정) 감지 경로를 그대로 재현한다.
exec env FAKE_SCENARIO=modify "$(dirname "$0")/fake_codex.sh" "$@"
