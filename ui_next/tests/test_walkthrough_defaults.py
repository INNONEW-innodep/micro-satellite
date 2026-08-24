"""따라하기 화면의 기본 샘플과 모델 안내를 고정한다.

부분 문자열로 기본 샘플을 고르면 'busan-doc-recovered'(문서 그림 복원 데모)가
'busan-nas-water-labels'보다 먼저 걸린다. 그 데모의 권장 모델은 persistence라
예측 면적이 모든 날짜에서 같게 나오고, 처음 보는 사람은 앱이 고장 난 것으로
읽는다. 실제로 그 사고가 났기 때문에 기본값과 안내 문구를 함께 고정한다.
"""

from __future__ import annotations

import pytest

from ui_next.samples import get_sample, list_samples
from ui_next.walkthrough import DEFAULT_SAMPLE_ID, MODEL_NOTES


def test_default_sample_exists_and_is_the_real_label_set() -> None:
    ids = {item.sample_id for item in list_samples()}
    assert DEFAULT_SAMPLE_ID in ids
    sample = get_sample(DEFAULT_SAMPLE_ID)
    assert sample.raw_data_provenance_known, "기본 샘플은 실자료 계보가 확인된 것이어야 한다"
    assert not sample.derived_demo, "문서 그림 복원 데모를 기본값으로 두면 안 된다"


def test_default_sample_model_actually_varies_over_horizons() -> None:
    """기본 샘플의 권장 모델이 날짜마다 같은 값을 내는 기준선이면 안 된다."""

    sample = get_sample(DEFAULT_SAMPLE_ID)
    assert sample.recommended_model_id != "persistence", (
        "persistence는 직전 관측을 그대로 반복해 예측선이 평평하다 — "
        "따라하기 첫 화면 기본값으로 부적절하다"
    )


def test_substring_match_still_picks_a_different_sample() -> None:
    """원래 버그를 재현해 두어, 부분 문자열 방식으로 되돌아가면 드러나게 한다.

    처음 사고는 'busan' 부분 문자열이 문서 복원 데모를 먼저 집어 persistence로
    예측선이 평평해진 것이었다. 그 데모의 권장 모델은 이후 교체했지만, 기본값을
    ID로 명시해야 한다는 요구는 그대로다.
    """

    samples = list_samples()
    naive = next(item.sample_id for item in samples if "busan" in item.sample_id.lower())
    assert naive != DEFAULT_SAMPLE_ID, (
        "부분 문자열 매칭이 우연히 맞아떨어지고 있다. 이 테스트의 전제가 바뀌었으니 "
        "기본값 선택 로직을 다시 확인하라"
    )
    assert get_sample(naive).derived_demo, "부분 문자열은 여전히 파생 데모를 먼저 집는다"


@pytest.mark.parametrize("sample", list_samples(), ids=lambda item: item.sample_id)
def test_every_recommended_model_has_a_plain_language_note(sample) -> None:
    """모든 샘플의 권장 모델은 결과 모양을 설명하는 문구를 갖는다."""

    assert sample.recommended_model_id in MODEL_NOTES, (
        f"{sample.sample_id}의 권장 모델 {sample.recommended_model_id!r} 설명이 없다 — "
        "사용자가 결과를 오해할 수 있다"
    )


def test_persistence_note_warns_about_flat_output() -> None:
    note = MODEL_NOTES["persistence"]
    assert "변하지 않" in note or "반복" in note
