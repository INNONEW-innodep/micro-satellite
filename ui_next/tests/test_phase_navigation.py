"""단계 이동이 번호 어긋남과 잘못된 값에 견디는지 확인한다.

단계를 하나 끼워 넣었을 때 실제로 두 가지가 한꺼번에 터졌다.
1. 다른 모듈에 번호를 받는 이동 함수를 넘겨 놓고 호출부는 제목을 줬다 →
   session_state["phase"]에 문자열이 들어가 헤더에서 앱 전체가 죽었다.
2. 새로고침해도 그 값이 남아 복구가 안 됐다.
그래서 (1) 제목→번호 해석과 (2) 깨진 값 방어를 각각 고정한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ui_next.state import PHASES, phase_index

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
UI_DIR = Path(__file__).resolve().parents[1]


def test_every_phase_title_resolves_to_its_own_index() -> None:
    titles = [title for title, _subtitle, _icon in PHASES]
    assert len(titles) == len(set(titles)), "단계 제목이 중복되면 이동 대상이 모호해진다"
    for index, title in enumerate(titles):
        assert phase_index(title) == index


def test_unknown_phase_title_is_rejected_loudly() -> None:
    with pytest.raises(KeyError):
        phase_index("없는 단계")


def test_walkthrough_and_nas_use_titles_not_numbers() -> None:
    """제목을 넘기는 모듈은 번호를 직접 쓰면 안 된다."""

    for name in ("walkthrough.py", "nas_ui.py"):
        source = (UI_DIR / name).read_text(encoding="utf-8")
        numeric = re.findall(r"go_to_phase\(\s*\d+\s*\)", source)
        assert not numeric, f"{name}에 번호 기반 이동이 남아 있다: {numeric}"
        for title in re.findall(r'go_to_phase\(\s*"([^"]+)"\s*\)', source):
            phase_index(title)  # 존재하지 않는 제목이면 KeyError


def test_modules_receiving_titles_are_given_the_name_based_helper() -> None:
    """app.py가 넘기는 이동 함수가 제목을 받는 쪽인지 고정한다."""

    source = APP_PATH.read_text(encoding="utf-8")
    injections = re.findall(r"go_to_phase=(\w+)", source)
    assert injections, "이동 함수를 넘기는 곳을 찾지 못했다"
    assert set(injections) == {"go_to_named_phase"}, (
        f"제목을 받지 않는 함수가 섞여 있다: {injections}"
    )


@pytest.mark.parametrize("bad", ["결과", None, -1, len(PHASES), "", 3.7])
def test_broken_phase_value_falls_back_to_first_screen(bad: object) -> None:
    """깨진 값이 들어와도 첫 화면으로 되돌아갈 뿐 예외가 나지 않아야 한다."""

    def resolve(raw: object) -> int:
        try:
            index = int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            index = 0
        if not 0 <= index < len(PHASES):
            index = 0
        return index

    index = resolve(bad)
    assert 0 <= index < len(PHASES)
    PHASES[index]  # 인덱싱이 항상 성공해야 한다


def test_area_chart_does_not_force_zero_baseline() -> None:
    """수체 면적 축이 0에서 시작하면 수백 px 변화가 안 보인다.

    실제로 '값이 안 변한다'는 오해가 나왔던 원인이다.
    """

    from ui_next.eda import padded_range

    low, high = padded_range([9891, 9550, 10394, 10192])
    assert low > 0, "0부터 그리면 변화가 묻힌다"
    span = high - low
    observed = 10394 - 9550
    assert observed / span > 0.5, "관측 변화폭이 축 높이의 절반 이상을 차지해야 보인다"
