from __future__ import annotations

import io
import json
import zipfile

from ui_next.app import handoff_template_zip


def test_handoff_template_is_self_describing_and_not_an_upload_claim() -> None:
    payload = handoff_template_zip()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {"README.txt", "frames.csv", "metadata.json"}
        readme = archive.read("README.txt").decode("utf-8-sig")
        frames = archive.read("frames.csv").decode("utf-8-sig")
        metadata = json.loads(archive.read("metadata.json"))

    assert "원본 SAR/광학영상이 아니라" in readme
    assert "ZIP 자체를 UI에 업로드" in readme
    assert frames.startswith("date,filename,water_level_m,quality_flag")
    assert metadata["data_stage"] == "binary_water_mask_after_detection"
    assert metadata["mask_values"] == {"0": "non-water", "1": "water"}
    assert "nodata" in metadata
    assert "checkpoint_sha256" in metadata["detection_model"]
