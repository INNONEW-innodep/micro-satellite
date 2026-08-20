from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ui_next.nas_catalog import (
    ICEYE_MATERIALIZED_SAMPLE_ID,
    MATERIALIZED_SAMPLE_ID,
    NAS_CATALOG_SUMMARY,
    NAS_DATASET_BY_ID,
    NAS_DATASET_GROUPS,
    TOTAL_BYTES,
    TOTAL_FILE_COUNT,
    get_nas_dataset_group,
    list_nas_dataset_groups,
    materialized_nas_samples,
)

EXPECTED_IDS = (
    "busan-water-labels",
    "iceye-water-labels",
    "iceye-raw",
    "planetscope-raw",
    "university-single-date",
    "giheung-optical",
    "hoedong-optical",
    "support-assets",
)


def test_catalog_has_stable_groups_and_exact_inventory_totals() -> None:
    groups = list_nas_dataset_groups()

    assert groups is NAS_DATASET_GROUPS
    assert tuple(group.group_id for group in groups) == EXPECTED_IDS
    assert sum(group.file_count for group in groups) == TOTAL_FILE_COUNT == 300
    assert sum(group.total_bytes for group in groups) == TOTAL_BYTES == 57_270_553_908
    assert NAS_CATALOG_SUMMARY.file_count == TOTAL_FILE_COUNT
    assert NAS_CATALOG_SUMMARY.total_bytes == TOTAL_BYTES
    assert NAS_CATALOG_SUMMARY.group_count == 8


def test_catalog_is_immutable_and_lookups_are_stable() -> None:
    busan = get_nas_dataset_group("busan-water-labels")

    assert NAS_DATASET_BY_ID[busan.group_id] is busan
    with pytest.raises(FrozenInstanceError):
        busan.display_name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        NAS_DATASET_BY_ID["changed"] = busan  # type: ignore[index]
    with pytest.raises(KeyError, match="unknown NAS dataset group"):
        get_nas_dataset_group("missing")


def test_busan_and_iceye_are_connected_to_materialized_samples() -> None:
    materialized = materialized_nas_samples()

    assert tuple(group.group_id for group in materialized) == (
        "busan-water-labels",
        "iceye-water-labels",
    )
    assert materialized[0].materialized_sample_id == MATERIALIZED_SAMPLE_ID
    assert materialized[1].materialized_sample_id == ICEYE_MATERIALIZED_SAMPLE_ID
    assert MATERIALIZED_SAMPLE_ID == "busan-nas-water-labels"
    assert ICEYE_MATERIALIZED_SAMPLE_ID == "iceye-nas-water-labels"
    assert NAS_CATALOG_SUMMARY.materialized_sample_count == 2


def test_paired_mask_groups_preserve_dates_intervals_and_pair_counts() -> None:
    busan = get_nas_dataset_group("busan-water-labels")
    iceye = get_nas_dataset_group("iceye-water-labels")

    assert busan.observation_dates == (
        "2020-02-18",
        "2020-03-12",
        "2020-03-25",
        "2020-04-14",
    )
    assert busan.interval_days == (23, 13, 20)
    assert busan.matched_pair_count == 4
    assert iceye.interval_days == (28, 16, 1)
    assert iceye.matched_pair_count == 4
    assert busan.readiness == iceye.readiness == "demo_ready"
    assert busan.direct_prediction is True
    assert iceye.direct_prediction is True
    assert NAS_CATALOG_SUMMARY.direct_prediction_group_count == 2


def test_raw_udm2_dem_and_support_groups_forbid_direct_prediction() -> None:
    iceye_raw = get_nas_dataset_group("iceye-raw")
    planetscope = get_nas_dataset_group("planetscope-raw")
    university = get_nas_dataset_group("university-single-date")
    support = get_nas_dataset_group("support-assets")

    assert "sar_slc" in iceye_raw.roles
    assert "quality_mask_udm2" in planetscope.roles
    assert "dem" in university.roles
    assert "delivery_archive" in support.roles
    assert not iceye_raw.direct_prediction
    assert not planetscope.direct_prediction
    assert not university.direct_prediction
    assert not support.direct_prediction
    assert iceye_raw.readiness == planetscope.readiness == "preprocess_required"
    assert university.readiness == "single_date_reference"
    assert support.readiness == "support_only"


def test_raw_optical_locations_do_not_count_acquisitions_as_unique_dates() -> None:
    planetscope = get_nas_dataset_group("planetscope-raw")
    hoedong = get_nas_dataset_group("hoedong-optical")

    assert planetscope.acquisition_count == 8
    assert planetscope.unique_date_count == 4
    assert hoedong.acquisition_count == 3
    assert hoedong.unique_date_count == 2
    assert hoedong.interval_days == (289,)
    assert not hoedong.direct_prediction


def test_catalog_contains_only_logical_relative_paths_and_no_connection_details() -> None:
    serialized = "\n".join(
        value
        for group in NAS_DATASET_GROUPS
        for value in (
            group.group_id,
            group.display_name,
            *group.relative_paths,
            group.sensor_name,
            group.direct_prediction_reason_ko,
            *group.cautions_ko,
        )
    )

    assert "172." not in serialized
    assert "admin" not in serialized.lower()
    assert "\\\\" not in serialized
    assert all(not path.startswith("/") for group in NAS_DATASET_GROUPS for path in group.relative_paths)
