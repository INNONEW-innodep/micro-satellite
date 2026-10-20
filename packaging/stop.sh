#!/usr/bin/env bash
# 중지 (컨테이너만 내린다. 이미지·데이터는 남는다). 데모 프로필을 꺼 두었어도 데모 컨테이너까지 내린다.
. "$(dirname "$0")/common.sh"; load_env
compose_down
ok "중지"
