"""표 출력 공통 함수 — 데이터 그리드(st.dataframe) 또는 HTML 표(st.table).

기본은 HTML 표(st.table)다. 데이터 그리드(st.dataframe)는 canvas 로 그려서 일부 브라우저·원격 화면·headless
캡처에서 빈 상자로 보이는 문제가 실제 시연 화면에서 확인됐다(2026-09-08). 정렬·스크롤이 필요한 큰 표에서만
`WATER_UI_STATIC_TABLES=0` 으로 데이터 그리드를 켤 수 있다.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st


def static_tables_enabled() -> bool:
    return os.getenv("WATER_UI_STATIC_TABLES", "1").strip().lower() not in ("0", "false", "no")


def _markdown_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """st.table 은 셀을 마크다운으로 그려서 홀로 있는 '-' 가 글머리 기호(•)로 바뀐다 → 대시(–)로 바꾼다."""
    return frame.map(lambda v: "–" if isinstance(v, str) and v.strip() in ("-", "") else v) if hasattr(frame, "map") else frame.applymap(lambda v: "–" if isinstance(v, str) and v.strip() in ("-", "") else v)


def show_table(data, *, key: str | None = None) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if static_tables_enabled():
        st.table(_markdown_safe(frame.reset_index(drop=True)))
    else:
        st.dataframe(frame, width="stretch", hide_index=True, key=key)


def static_table(rows) -> None:
    """항상 HTML 표(st.table)로 그리는 짧은 표. '-' 자리표시자를 안전하게 바꾼다."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    st.table(_markdown_safe(frame.reset_index(drop=True)))
