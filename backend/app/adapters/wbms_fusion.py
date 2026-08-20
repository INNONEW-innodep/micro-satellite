from __future__ import annotations

import csv
import json
from datetime import date as Date
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..wbms.config import WbmsSettings
from ..wbms.jobs import FileJobStore
from ..wbms.runner import (
    CONTAINER_MODULE_DIR,
    DockerRunner,
    docker_prefix,
    summarize_failure,
)
from .wbms_segmentation import TB_CODES, WbmsExecutionError, WbmsInputError

SAT_NAMES = {"iceye": "ICEYE", "planet": "PlanetScope"}
_DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d")


class WbmsFusionAdapter:
    """④ fusion LSTM correction through the wbms container (synchronous).

    The bundle at ``config/WBMS_Fusion_LSTM_Busan`` (meta.json is the contract
    of record) regresses the **corrected absolute water level for the observed
    date** from a 60-day×3-channel weather window (T·H·PP), the satellite
    water level, satellite type and gauge embedding. It is a same-day
    corrector, not a forecaster; zero-padding of a short weather window is
    forbidden (Correct.py fails hard, and we pre-check for a clear 422).
    """

    model_id = "wbms-fusion-lstm"

    def __init__(self, settings: WbmsSettings, runner: DockerRunner) -> None:
        self.settings = settings
        self.runner = runner
        self._meta_cache: tuple[float, dict[str, Any]] | None = None
        self._aws_cache: tuple[float, dict[str, Any]] | None = None

    # ── bundle contract ────────────────────────────────────────────────────
    def bundle_meta(self) -> dict[str, Any]:
        meta_path = self.settings.bundle_dir / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime
        except OSError as exc:
            raise WbmsExecutionError(
                f"fusion bundle meta.json unavailable: {meta_path}"
            ) from exc
        if self._meta_cache is None or self._meta_cache[0] != mtime:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise WbmsExecutionError(f"cannot parse bundle meta.json: {exc}") from exc
            self._meta_cache = (mtime, meta)
        return self._meta_cache[1]

    def bundle_info(self) -> dict[str, Any]:
        meta = self.bundle_meta()
        metrics = meta.get("metrics") or {}
        channels = list(meta.get("weather_channels") or [])
        t_index = channels.index("T") if "T" in channels else 0
        mins = list(meta.get("weather_min") or [])
        maxs = list(meta.get("weather_max") or [])
        train_temp = (
            [float(mins[t_index]), float(maxs[t_index])]
            if len(mins) > t_index and len(maxs) > t_index
            else []
        )
        return {
            "path": str(self.settings.bundle_dir),
            "schema_id": meta.get("schema"),
            "head": meta.get("head"),
            "days": int(meta.get("days", 60)),
            "wl_valid_range_m": [float(meta["wl_min"]), float(meta["wl_max"])],
            "weather_channels": channels,
            "train_temperature_range_c": train_temp,
            "lodo_rmse_m": metrics.get("lodo_rmse_m"),
            "baseline_rmse_m": metrics.get("baseline_rmse_m"),
            "discriminative": metrics.get("discriminative"),
            "model_sha256": meta.get("model_sha256"),
        }

    # ── weather CSV (positional columns, CP949 — mirrors fusion_lstm_core) ─
    def aws_index(self) -> dict[str, Any]:
        path = self.settings.aws_csv
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            raise WbmsExecutionError(f"weather CSV unavailable: {path}") from exc
        if self._aws_cache is not None and self._aws_cache[0] == mtime:
            return self._aws_cache[1]

        raw: str | None = None
        for encoding in ("cp949", "utf-8"):
            try:
                raw = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raise WbmsExecutionError(f"weather CSV is neither CP949 nor UTF-8: {path}")

        rows: list[tuple[Date, float | None]] = []
        for i, cells in enumerate(csv.reader(raw.splitlines())):
            if i == 0 or len(cells) < 3:
                continue  # header (지점,일시,평균기온,...) or short row
            day = _parse_day(cells[1])
            if day is None:
                continue
            rows.append((day, _parse_float(cells[2])))
        if not rows:
            raise WbmsExecutionError(f"weather CSV has no parseable rows: {path}")
        rows.sort(key=lambda item: item[0])
        days = [item[0] for item in rows]
        if len(set(days)) != len(days):
            raise WbmsExecutionError(
                "weather CSV contains duplicate dates — the container window slice is "
                "positional and would silently read the wrong span"
            )
        index = {day.isoformat(): i for i, day in enumerate(days)}
        data = {
            "days": days,
            "index": index,
            "temperature_c": [item[1] for item in rows],
            "start": days[0].isoformat(),
            "end": days[-1].isoformat(),
            "path": str(path),
        }
        self._aws_cache = (mtime, data)
        return data

    def weather_window_check(self, yyyymmdd: str, window_days: int) -> dict[str, Any]:
        aws = self.aws_index()
        iso = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
        j = aws["index"].get(iso)
        if j is None:
            raise WbmsInputError(
                f"observation date {iso} is not in the weather CSV "
                f"({aws['start']}..{aws['end']}) — zero-padding the weather window is "
                "forbidden, so this record cannot be corrected"
            )
        if j - window_days + 1 < 0:
            raise WbmsInputError(
                f"weather window too short for {iso}: need {window_days} daily rows ending "
                f"on that date, CSV starts {aws['start']} — zero-padding is forbidden"
            )
        window_dates = aws["days"][j - window_days + 1 : j + 1]
        temps = [
            t for t in aws["temperature_c"][j - window_days + 1 : j + 1] if t is not None
        ]
        gaps = sum(
            1
            for a, b in zip(window_dates, window_dates[1:])
            if (b - a) != timedelta(days=1)
        )
        return {
            "date": yyyymmdd,
            "window_days": window_days,
            "window_start": window_dates[0].isoformat(),
            "window_end": window_dates[-1].isoformat(),
            "temperature_min_c": min(temps) if temps else None,
            "temperature_max_c": max(temps) if temps else None,
            "calendar_gaps": gaps,
        }

    # ── correction ─────────────────────────────────────────────────────────
    def correct(self, request: dict[str, Any]) -> dict[str, Any]:
        meta = self.bundle_meta()
        info = self.bundle_info()
        days = info["days"]
        wl_min, wl_max = info["wl_valid_range_m"]
        loc_index = meta.get("loc_index") or {}
        loc = str(request["loc_id"]).strip().lower()
        testbed = str(request.get("testbed", "busan")).lower()
        if loc not in loc_index:
            raise WbmsInputError(
                f"loc_id {loc!r} is not in the fusion bundle loc_index "
                f"{sorted(loc_index)} — correcting an untrained gauge requires retraining "
                "(no fallback exists by design)"
            )

        observations = list(request["observations"])
        seen: set[tuple[str, str]] = set()
        for obs in observations:
            key = (obs["satellite"], obs["date"])
            if key in seen:
                raise WbmsInputError(
                    f"duplicate observation for satellite={key[0]} date={key[1]}"
                )
            seen.add(key)

        warnings: list[str] = []
        windows: list[dict[str, Any]] = []
        train_temp = info["train_temperature_range_c"]
        for obs in observations:
            window = self.weather_window_check(obs["date"], days)
            out_of_range = False
            if train_temp and window["temperature_max_c"] is not None:
                t_lo, t_hi = train_temp
                if window["temperature_max_c"] > t_hi or (
                    window["temperature_min_c"] is not None
                    and window["temperature_min_c"] < t_lo
                ):
                    out_of_range = True
                    warnings.append(
                        f"WARN: weather window for {obs['date']} has temperatures "
                        f"[{window['temperature_min_c']}, {window['temperature_max_c']}] °C "
                        f"outside the training range [{t_lo}, {t_hi}] °C — the bundle was "
                        "trained on winter data (2019-10-31..2020-04-30); correction "
                        "quality outside that regime is unvalidated"
                    )
            window["temperature_out_of_train_range"] = out_of_range
            if window["calendar_gaps"]:
                warnings.append(
                    f"WARN: weather window for {obs['date']} has "
                    f"{window['calendar_gaps']} calendar gap(s) — the {days}-day window "
                    "may span more than "
                    f"{days} real days"
                )
            windows.append(window)
            if not (wl_min <= float(obs["water_level_m"]) <= wl_max):
                warnings.append(
                    f"WARN: satellite water level {obs['water_level_m']} m on "
                    f"{obs['date']} is outside the bundle's valid range "
                    f"[{wl_min}, {wl_max}] m — the corrected value is extrapolation"
                )

        workspace, command = self._prepare_run(testbed, loc, observations)
        result = self.runner.run(command, timeout_s=self.settings.fusion_timeout_s)
        if result.returncode != 0:
            raise WbmsExecutionError(
                f"Correct.py failed: {summarize_failure(result)} (workspace {workspace})"
            )

        fused = self._read_fused(workspace, testbed, loc)
        by_key = {
            (str(rec.get("satellite")), str(rec.get("date"))): rec
            for rec in fused.get("records", [])
        }
        records: list[dict[str, Any]] = []
        for obs in observations:
            sat_name = SAT_NAMES[obs["satellite"]]
            rec = by_key.get((sat_name, obs["date"]))
            if rec is None:
                raise WbmsExecutionError(
                    f"fused output is missing the record for {sat_name} {obs['date']} "
                    f"(workspace {workspace})"
                )
            records.append(
                {
                    "date": obs["date"],
                    "satellite": sat_name,
                    "satellite_water_level_m": float(rec["water_level_m"]),
                    "corrected_water_level_m": float(rec["corrected_water_level_m"]),
                    "offset_m": float(rec["offset_m"]),
                    "correction_mode": str(rec.get("correction_mode", "absolute")),
                    "wl_out_of_train_range": bool(rec.get("wl_out_of_train_range", False)),
                    "pair_gap_days": rec.get("pair_gap_days"),
                    "paired": rec.get("paired"),
                    "water_area_km2": rec.get("water_area_km2"),
                }
            )
        if any(rec.get("paired") is False for rec in records):
            warnings.append(
                "WARN: some observations have no cross-sensor pair within the allowed "
                "gap — fusion still corrects them individually"
            )
        if info.get("discriminative") is False:
            warnings.append(
                "bundle honesty note: on the current 30-sample LODO evaluation the "
                f"model ({info.get('lodo_rmse_m')} m RMSE) does not beat the "
                f"station-mean baseline ({info.get('baseline_rmse_m')} m) — do not cite "
                "target_met without this context"
            )

        persistence: list[dict[str, Any]] = []
        horizon = int(request.get("persistence_horizon_days") or 0)
        if horizon > 0:
            last = max(records, key=lambda rec: rec["date"])
            base_day = Date(
                int(last["date"][0:4]), int(last["date"][4:6]), int(last["date"][6:8])
            )
            for step in range(1, horizon + 1):
                day = base_day + timedelta(days=step)
                persistence.append(
                    {
                        "date": day.strftime("%Y%m%d"),
                        "corrected_water_level_m": last["corrected_water_level_m"],
                        "method": "persistence",
                    }
                )
            warnings.append(
                "future steps are persistence of the corrected level on "
                f"{last['date']} — the fusion LSTM corrects the observed date only and "
                "does not forecast"
            )

        return {
            "model_id": self.model_id,
            "semantics": "same_day_correction",
            "loc_id": loc,
            "testbed": testbed,
            "records": records,
            "persistence_forecast": persistence,
            "weather_windows": windows,
            "warnings": warnings,
            "bundle": info,
            "runtime": {
                "elapsed_s": round(result.elapsed_s, 1),
                "command": command,
                "workspace": str(workspace),
            },
        }

    # ── container plumbing ─────────────────────────────────────────────────
    def _prepare_run(
        self, testbed: str, loc: str, observations: list[dict[str, Any]]
    ) -> tuple[Path, list[str]]:
        run_id = FileJobStore.new_job_id("fus")
        workspace = self.settings.fusion_work_dir / run_id
        tb = TB_CODES.get(testbed, "Busan")
        wlwa_dirs = {
            "iceye": workspace / "wlwa" / "ICEYE",
            "planet": workspace / "wlwa" / "PlanetScope",
        }
        for path in (*wlwa_dirs.values(), workspace / "fusion"):
            path.mkdir(parents=True, exist_ok=True)

        for sat_key, directory in wlwa_dirs.items():
            group = [obs for obs in observations if obs["satellite"] == sat_key]
            if not group:
                continue
            dates = sorted(obs["date"] for obs in group)
            sat_name = SAT_NAMES[sat_key]
            stem = f"WLWA_{tb}_{sat_name}_{loc}_{dates[0]}_{dates[-1]}"
            doc = {
                "run_id": run_id,
                "schema": "backend.synthesized.wlwa/1",
                "note": (
                    "synthesized by the backend fusion adapter from API-supplied ③ "
                    "records; not a calc_wlwa product"
                ),
                "testbed": testbed,
                "loc_id": loc,
                "roi": None,
                "records": [
                    {
                        "testbed": testbed,
                        "satellite": sat_name,
                        "date": obs["date"],
                        "water_level_m": float(obs["water_level_m"]),
                        "water_area_km2": obs.get("water_area_km2"),
                        "quality": "api",
                        "source_wb": f"api:{sat_name}:{obs['date']}",
                    }
                    for obs in group
                ],
            }
            (directory / f"{stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            qc = {"base": stem, "status": "COMPLETE", "products": [f"{stem}.json"]}
            (directory / f"{stem}_qc.json").write_text(
                json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        mounts: list[tuple[Path, str, bool]] = [
            (self.settings.bundle_dir, "/bundle", True),
            (self.settings.aws_csv.parent, "/aux", True),
            (workspace, "/work", False),
        ]
        argv = docker_prefix(self.settings.docker_bin, self.settings.docker_image, mounts)
        argv += [
            "python3",
            f"{CONTAINER_MODULE_DIR}/Correct.py",
            "--sar_result_dir", "/work/wlwa/ICEYE",
            "--optic_result_dir", "/work/wlwa/PlanetScope",
            "--model", "/bundle",
            "--aws_csv", f"/aux/{self.settings.aws_csv.name}",
            "--testbed", testbed,
            "--loc_id", loc,
            "--output_dir", "/work/fusion",
            "--gpu", "cpu",
        ]
        return workspace, argv

    def _read_fused(self, workspace: Path, testbed: str, loc: str) -> dict[str, Any]:
        tb = TB_CODES.get(testbed, "Busan")
        fused_path = workspace / "fusion" / f"FUSED_{tb}_{loc}.json"
        if not fused_path.is_file():
            candidates = sorted((workspace / "fusion").glob("FUSED_*.json"))
            candidates = [c for c in candidates if not c.name.endswith("_qc.json")]
            if not candidates:
                raise WbmsExecutionError(
                    f"Correct.py exited 0 but produced no FUSED_*.json under "
                    f"{workspace / 'fusion'}"
                )
            fused_path = candidates[0]
        try:
            return json.loads(fused_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WbmsExecutionError(f"cannot parse fused output {fused_path}: {exc}") from exc


def _parse_day(raw: str) -> Date | None:
    text = raw.strip()
    if not text:
        return None
    from datetime import datetime

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_float(raw: str) -> float | None:
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):
        return None
    return value if value == value else None
