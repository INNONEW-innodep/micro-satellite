"""슬라이드 화면의 순수 도우미·배선. 화면 자체는 AppTest로 따로 확인한다."""

from __future__ import annotations

from ui_next.region_samples import REGIONS_DIR, region_mask_datasets, region_sample_id
from ui_next.samples import list_samples
from ui_next.slides import SLIDE_KEYS, SLIDES, slide_index


def test_slide_order_is_the_easy_story() -> None:
    assert SLIDE_KEYS == ("region", "data", "detect", "forecast", "result", "evaluate")
    assert slide_index("evaluate") == len(SLIDES) - 1
    assert all(title and sub for _key, title, sub in SLIDES)


def test_region_samples_come_from_registry_assets() -> None:
    samples = region_mask_datasets()
    ids = {s.sample_id for s in samples}
    assert region_sample_id("nakdong_busan", "ICEYE") in ids
    assert region_sample_id("geum_daecheong", "PlanetScope") in ids
    for sample in samples:
        assert len(sample.frames) >= 2
        assert sample.pixel_area_m2 and sample.pixel_area_m2 > 0
        assert sample.kma_station_id, "지역 샘플은 기상청 지점을 알아야 과거 관측을 재현한다"
        assert sample.recommended_model_id != "persistence"
        sizes = {len(frame.png_bytes) > 0 for frame in sample.frames}
        assert sizes == {True}
    # 카탈로그에도 같은 id로 들어가 있어야 빠른 실행이 그대로 쓴다
    catalog_ids = {s.sample_id for s in list_samples()}
    assert ids <= catalog_ids


def test_app_defaults_to_easy_mode_and_keeps_crop_page() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'UI_MODES = ("쉬운 화면 (슬라이드)", "위성 파이프라인 (입력→예측)", "고급 화면 (상세 탭)")' in source
    assert "render_pipeline_page(client_factory=api_client, execute_quick_sample=execute_quick_sample)" in source
    assert "render_slides(client_factory=api_client, execute_quick_sample=execute_quick_sample)" in source
    assert "render_crop_detection_page()" in source


def test_user_copy_without_satellite_assets_does_not_hide_builtin_samples(tmp_path) -> None:
    # 재학습 산출물만 든 사용자 사본(region.json + models/)이 생겨도 내장 위성 장면 샘플은 그대로 보여야 한다
    user_copy = tmp_path / "nakdong_busan"
    user_copy.mkdir()
    (user_copy / "region.json").write_text((REGIONS_DIR / "nakdong_busan" / "region.json").read_text(encoding="utf-8"), encoding="utf-8")
    ids = {s.sample_id for s in region_mask_datasets(user_dir=tmp_path)}
    assert region_sample_id("nakdong_busan", "ICEYE") in ids
    assert region_sample_id("nakdong_busan", "PlanetScope") in ids
