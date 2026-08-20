from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest

from ui_next.model_eval import (
    BATCH_WAITING_MESSAGE_KO,
    FUSED_WAITING_MESSAGE_KO,
    GAUGE_CSV_PATH,
    IN_SAMPLE_LABEL_KO,
    POOLED_CAUTION_KO,
    PROTOCOL_DEPLOY_KO,
    PROTOCOL_RESEARCH_KO,
    PROTOCOL_SMOKE_KO,
    SMOKE_EVAL_PATH,
    WEATHER_CSV_PATH,
    build_gauge_level_figure,
    build_scene_iou_figure,
    build_weather_figure,
    gauge_detail_rows,
    gauge_pivot_rows,
    load_batch_evals,
    load_gauge_levels,
    load_smoke_eval,
    load_valreports,
    load_weather_series,
    per_scene_rows,
    pooled_rows,
    protocol_comparison_rows,
)

VALREPORTS_PRESENT = all(
    (GAUGE_CSV_PATH.parent.parent / "07_test_evidence" / name).is_file()
    for name in ("VALREPORT_Busan_ICEYE.json", "VALREPORT_Busan_PlanetScope.json")
)

requires_handover = pytest.mark.skipif(
    not VALREPORTS_PRESENT, reason="handover evidence package not present"
)
requires_gauge_csv = pytest.mark.skipif(
    not GAUGE_CSV_PATH.is_file(), reason="handover gauge CSV not present"
)
requires_weather_csv = pytest.mark.skipif(
    not WEATHER_CSV_PATH.is_file(), reason="handover weather CSV not present"
)
requires_smoke = pytest.mark.skipif(
    not SMOKE_EVAL_PATH.is_file(), reason="local smoke evaluation not present"
)


# --- deployment-protocol validation reports ----------------------------------


@requires_handover
def test_valreports_carry_deploy_protocol_labels_and_exact_scene_metrics() -> None:
    iceye, planet = load_valreports()

    assert iceye.status == "ok" and planet.status == "ok"
    assert iceye.sensor_type == "SAR" and iceye.sensor_name == "ICEYE"
    assert planet.sensor_type == "OPTIC" and planet.sensor_name == "PlanetScope"
    assert iceye.protocol_label_ko == PROTOCOL_DEPLOY_KO
    assert iceye.provenance == (
        "data/incoming/handover/07_test_evidence/VALREPORT_Busan_ICEYE.json"
    )
    assert not iceye.provenance.startswith("/")

    by_date = {scene.date: scene for scene in iceye.per_scene}
    assert by_date["2020-04-16"].water_iou == pytest.approx(0.878575)
    planet_by_date = {scene.date: scene for scene in planet.per_scene}
    assert planet_by_date["2020-03-25"].water_iou == pytest.approx(0.948433)

    # pooled 수치는 별도 라벨을 강제해 대표값처럼 보이지 않게 합니다.
    assert iceye.pooled is not None
    assert iceye.pooled.date == "pooled(4씬)"
    assert iceye.pooled.water_iou == pytest.approx(0.925887)
    assert "대표 성능이 아닙니다" in iceye.pooled_caution_ko

    scene_rows = per_scene_rows(iceye)
    assert len(scene_rows) == 4
    assert all(row["수치 계열"] == PROTOCOL_DEPLOY_KO for row in scene_rows)
    pooled = pooled_rows((iceye, planet))
    assert len(pooled) == 2
    assert all("pooled(4씬)" in row["수치 계열"] for row in pooled)
    assert all(row["주의"] == POOLED_CAUTION_KO for row in pooled)


def test_valreport_loader_degrades_on_missing_and_invalid_files(tmp_path) -> None:
    missing_iceye, missing_planet = load_valreports(tmp_path)
    assert missing_iceye.status == "missing" and missing_planet.status == "missing"
    assert "찾지 못했습니다" in missing_iceye.message_ko
    assert missing_iceye.per_scene == ()

    broken = {
        "mode": "with_gt",
        "sensor_type": "SAR",
        "n_evaluated": 1,
        "threshold": 0.5,
        "metrics": {
            "water_iou": 1.5,  # out of [0, 1]
            "accuracy": 0.9,
            "recall": 0.9,
            "precision": 0.9,
            "f1": 0.9,
        },
        "per_scene": [
            {
                "date": "20200416",
                "water_iou": 0.9,
                "accuracy": 0.9,
                "recall": 0.9,
                "precision": 0.9,
                "f1": 0.9,
            }
        ],
    }
    (tmp_path / "VALREPORT_Busan_ICEYE.json").write_text(
        json.dumps(broken), encoding="utf-8"
    )
    invalid, _ = load_valreports(tmp_path)
    assert invalid.status == "invalid"
    assert "계약과 다릅니다" in invalid.message_ko


# --- smoke re-inference -------------------------------------------------------


@requires_smoke
def test_smoke_eval_parses_scene_identity_and_local_metrics() -> None:
    smoke = load_smoke_eval()

    assert smoke.status == "ok"
    assert smoke.protocol_label_ko == PROTOCOL_SMOKE_KO
    assert smoke.sensor_name == "ICEYE"
    assert smoke.scene_date == "2020-04-16"
    assert smoke.metrics is not None
    assert smoke.metrics.water_iou == pytest.approx(0.881026)
    assert smoke.valid_pixels == 99_509_048
    assert smoke.provenance == "data/wb_smoke/smoke_eval.json"


def test_smoke_eval_degrades_without_raising(tmp_path) -> None:
    missing = load_smoke_eval(tmp_path / "smoke_eval.json")
    assert missing.status == "missing"
    assert "스모크" in missing.message_ko

    invalid_path = tmp_path / "smoke_eval.json"
    invalid_path.write_text(json.dumps({"wb": "note.txt"}), encoding="utf-8")
    invalid = load_smoke_eval(invalid_path)
    assert invalid.status == "invalid"
    assert invalid.metrics is None


# --- three-series comparison --------------------------------------------------


@requires_handover
@requires_smoke
def test_protocol_comparison_keeps_research_deploy_and_smoke_series_apart() -> None:
    rows = protocol_comparison_rows(load_valreports(), load_smoke_eval())

    assert [row["수치 계열"] for row in rows] == [
        PROTOCOL_RESEARCH_KO,
        PROTOCOL_DEPLOY_KO,
        PROTOCOL_SMOKE_KO,
    ]
    research, deploy, smoke = rows
    assert research["SAR(ICEYE) 시험씬 water IoU"] == pytest.approx(0.8949)
    assert research["광학(PlanetScope) 시험씬 water IoU"] == pytest.approx(0.9487)
    assert "00_README_인수문서.md" in research["근거"]
    assert deploy["SAR(ICEYE) 시험씬 water IoU"] == pytest.approx(0.878575)
    assert deploy["광학(PlanetScope) 시험씬 water IoU"] == pytest.approx(0.948433)
    assert "배포 UTM 그리드" in deploy["근거"]
    assert smoke["SAR(ICEYE) 시험씬 water IoU"] == pytest.approx(0.881026)
    assert smoke["광학(PlanetScope) 시험씬 water IoU"] is None


def test_protocol_comparison_blanks_degraded_series_instead_of_guessing(
    tmp_path,
) -> None:
    rows = protocol_comparison_rows(
        load_valreports(tmp_path), load_smoke_eval(tmp_path / "missing.json")
    )

    research, deploy, smoke = rows
    assert research["SAR(ICEYE) 시험씬 water IoU"] == pytest.approx(0.8949)
    assert deploy["SAR(ICEYE) 시험씬 water IoU"] is None
    assert deploy["광학(PlanetScope) 시험씬 water IoU"] is None
    assert smoke["SAR(ICEYE) 시험씬 water IoU"] is None


# --- future batch outputs -----------------------------------------------------


def test_batch_evals_wait_explicitly_until_chain_runner_writes(tmp_path) -> None:
    waiting = load_batch_evals(tmp_path / "eval")
    assert waiting.status == "waiting"
    assert waiting.message_ko == BATCH_WAITING_MESSAGE_KO

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert load_batch_evals(empty_dir).status == "waiting"


def test_batch_evals_parse_valid_files_and_skip_broken_ones(tmp_path) -> None:
    (tmp_path / "eval_20200416.json").write_text(
        json.dumps(
            {
                "wb": "/runs/WB_Busan_ICEYE_20200416T183630.tif",
                "metrics": {"water_iou": 0.87, "f1": 0.93},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    collection = load_batch_evals(tmp_path)

    assert collection.status == "ok"
    assert collection.skipped == ("broken.json",)
    assert len(collection.reports) == 1
    report = collection.reports[0]
    assert report.sensor_name == "ICEYE"
    assert report.scene_date == "2020-04-16"
    assert report.water_iou == pytest.approx(0.87)


# --- gauge water levels -------------------------------------------------------


@requires_gauge_csv
def test_gauge_levels_parse_thirty_in_sample_samples_with_verified_mapping(
    tmp_path,
) -> None:
    table = load_gauge_levels(fused_dir=tmp_path / "absent")

    assert table.status == "ok"
    assert table.n_samples == 30
    assert table.in_sample is True
    assert table.in_sample_label_ko == IN_SAMPLE_LABEL_KO
    assert "in-sample" in table.message_ko
    assert table.level_min_m == pytest.approx(1.43)
    assert table.level_max_m == pytest.approx(2.415)
    assert table.date_min == "2020-02-18" and table.date_max == "2020-04-16"

    by_key = {(row.date, row.satellite, row.loc_name): row for row in table.rows}
    gimhae = by_key[("2020-03-02", "ICEYE", "gimhae")]
    assert gimhae.loc_id == 2
    assert gimhae.real_level_m == pytest.approx(1.475)
    assert gimhae.fused_level_m is None

    # 융합 산출 부재 시 대조 열은 생략됩니다.
    assert table.has_fused is False
    assert table.fused_message_ko == FUSED_WAITING_MESSAGE_KO
    detail = gauge_detail_rows(table)
    assert len(detail) == 30
    assert "융합 보정 수위 (m)" not in detail[0]

    pivot = gauge_pivot_rows(table)
    assert len(pivot) == 8  # 광학 4일 + SAR 4일
    by_pivot_key = {(row["관측 날짜"], row["위성"]): row for row in pivot}
    assert by_pivot_key[("2020-03-30", "ICEYE")]["jeongcheon (m)"] is None
    assert by_pivot_key[("2020-03-02", "ICEYE")]["gimhae (m)"] == pytest.approx(1.475)

    iceye_only = gauge_detail_rows(table, satellite="ICEYE")
    assert len(iceye_only) == 14
    assert all(row["위성"] == "ICEYE" for row in iceye_only)


def test_gauge_levels_reject_schema_drift_and_missing_file(tmp_path) -> None:
    missing = load_gauge_levels(tmp_path / "absent.csv", tmp_path / "absent")
    assert missing.status == "missing"

    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("date,file,level\n2020-01-01,x.tif,1.0\n", encoding="utf-8")
    assert load_gauge_levels(bad_header, tmp_path / "absent").status == "invalid"

    bad_row = tmp_path / "bad_row.csv"
    bad_row.write_text(
        "date,file,median_value,n_boundary_pixels,out_h,out_w,sat_type,loc_id,real_water_level\n"
        "2020-03-02,WB_Busan_ICEYE_20200302T183857.tif,0.1,10,100,100,1,9,1.5\n",
        encoding="utf-8",
    )
    invalid = load_gauge_levels(bad_row, tmp_path / "absent")
    assert invalid.status == "invalid"
    assert "loc_id" in invalid.message_ko


@requires_gauge_csv
def test_gauge_levels_add_fused_comparison_only_when_runs_exist(tmp_path) -> None:
    fused_dir = tmp_path / "fused"
    station_dir = fused_dir / "gimhae"
    station_dir.mkdir(parents=True)
    # 체인 러너 실산출과 동일한 중첩 배치 + records 없는 qc 사이드카(스킵돼야 함)
    (station_dir / "FUSED_Busan_gimhae_qc.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8"
    )
    (station_dir / "FUSED_Busan_gimhae.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "loc_id": "gimhae",
                        "satellite": "ICEYE",
                        "date": "20200302",
                        "corrected_water_level_m": 1.469883,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    table = load_gauge_levels(fused_dir=fused_dir)

    assert table.status == "ok"
    assert table.has_fused is True
    assert "융합 보정 수위" in table.fused_message_ko
    by_key = {(row.date, row.satellite, row.loc_name): row for row in table.rows}
    matched = by_key[("2020-03-02", "ICEYE", "gimhae")]
    assert matched.fused_level_m == pytest.approx(1.469883)
    assert matched.fused_delta_m == pytest.approx(1.469883 - 1.475)
    assert by_key[("2020-03-02", "ICEYE", "hupo")].fused_level_m is None

    detail = gauge_detail_rows(table)
    assert "융합 보정 수위 (m)" in detail[0]


# --- weather ------------------------------------------------------------------


@requires_weather_csv
def test_weather_series_parses_the_real_cp949_daily_table() -> None:
    series = load_weather_series()

    assert series.status == "ok"
    assert series.encoding == "cp949"
    assert series.station_ids == ("257",)
    assert series.n_days == 183
    assert series.date_min == "2019-10-31" and series.date_max == "2020-04-30"
    assert series.missing_temperature == 1
    assert series.missing_precipitation == 138  # 빈 칸은 0으로 채우지 않습니다.
    assert series.missing_humidity == 1
    assert series.sha256_matches_fused_evidence is True
    assert series.provenance == (
        "data/incoming/handover/06_aux/busan_aws_2019_202004.csv"
    )

    first = series.rows[0]
    assert first.date == "2019-10-31"
    assert first.temperature_c == pytest.approx(14.1)
    assert first.precipitation_mm is None
    assert first.humidity_pct == pytest.approx(69.9)


def test_weather_loader_handles_cp949_bytes_and_degrades_cleanly(tmp_path) -> None:
    encoded = tmp_path / "aws_cp949.csv"
    encoded.write_bytes(
        (
            "지점,일시,평균기온(°C),일강수량(mm),평균 상대습도(%),합계 일사량(MJ/m2)\n"
            "159,11/1/2019,10.5,2.5,70.1,5.0\n"
            "159,11/2/2019,11,,71,5\n"
            ",,,,,\n"
        ).encode("cp949")
    )
    series = load_weather_series(encoded)
    assert series.status == "ok"
    assert series.encoding == "cp949"
    assert series.n_days == 2
    assert series.rows[0].precipitation_mm == pytest.approx(2.5)
    assert series.rows[1].precipitation_mm is None
    assert series.sha256_matches_fused_evidence is False

    missing = load_weather_series(tmp_path / "absent.csv")
    assert missing.status == "missing"

    wrong = tmp_path / "wrong.csv"
    wrong.write_bytes("a,b,c\n1,2,3\n".encode("cp949"))
    assert load_weather_series(wrong).status == "invalid"


# --- figures ------------------------------------------------------------------


@requires_handover
@requires_smoke
def test_scene_iou_figure_names_every_series_explicitly() -> None:
    figure = build_scene_iou_figure(load_valreports(), load_smoke_eval())

    assert isinstance(figure, go.Figure)
    names = [trace.name for trace in figure.data]
    assert f"{PROTOCOL_DEPLOY_KO} · ICEYE" in names
    assert f"{PROTOCOL_DEPLOY_KO} · PlanetScope" in names
    assert PROTOCOL_RESEARCH_KO in names
    assert PROTOCOL_SMOKE_KO in names
    smoke_trace = figure.data[names.index(PROTOCOL_SMOKE_KO)]
    assert tuple(smoke_trace.x) == ("ICEYE 2020-04-16",)
    assert smoke_trace.y[0] == pytest.approx(0.881026)
    assert "계열" in figure.layout.title.text


def test_figures_degrade_to_partial_or_none_without_data(tmp_path) -> None:
    empty_figure = build_scene_iou_figure(
        load_valreports(tmp_path), load_smoke_eval(tmp_path / "none.json")
    )
    assert [trace.name for trace in empty_figure.data] == []

    degraded_table = load_gauge_levels(tmp_path / "absent.csv", tmp_path / "absent")
    assert build_gauge_level_figure(degraded_table) is None
    assert gauge_pivot_rows(degraded_table) == ()
    assert build_weather_figure(load_weather_series(tmp_path / "absent.csv")) is None


@requires_gauge_csv
@requires_weather_csv
def test_gauge_and_weather_figures_are_plotly_ready(tmp_path) -> None:
    gauge_figure = build_gauge_level_figure(
        load_gauge_levels(fused_dir=tmp_path / "absent")
    )
    assert isinstance(gauge_figure, go.Figure)
    assert len(gauge_figure.data) == 4  # 게이지 지점당 한 계열
    assert "in-sample" in gauge_figure.layout.title.text

    weather_figure = build_weather_figure(load_weather_series())
    assert isinstance(weather_figure, go.Figure)
    names = [trace.name for trace in weather_figure.data]
    assert names == ["일강수량 (mm)", "평균기온 (°C)", "평균 상대습도 (%)"]
    assert len(weather_figure.data[0].x) == 183 - 138  # 강수 빈 칸 제외
