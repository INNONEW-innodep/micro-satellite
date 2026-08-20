#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""wbms_chain_runner.py — WBMS ②③④⑤ CPU 체인 러너 (docker wbms:0.13).

체인:  ② detect_water (씬별) → ③ calc_wlwa (센서×지점 집계) → ④ Correct (지점별 융합)
       → ⑤ validate (센서별, 선택 단계 — 기본 steps 에 미포함)

출력 규약 (다른 에이전트와의 계약, <out-root> 기본 data/wbms_runs):
  ② SAR   → <out-root>/wb/2020/<date>/WB_Busan_ICEYE_<stamp>.tif (+_meta/_qc)
  ② 광학  → <out-root>/wb_planet/2020/<date>/WB_Busan_PlanetScope_<stamp>.tif
  ③       → <out-root>/wlwa/<sensor>/<station>/WLWA_Busan_<Sensor>_<loc>_<s>_<e>.json
            (④ 의 지점 대조 가드 때문에 지점별 폴더 분리가 필수다 — 한 폴더에 두 지점의
             WLWA 가 있으면 ④ 가 AOI_INVALID 로 정당 실패한다)
  ④       → <out-root>/fused/<station>/FUSED_Busan_<station>.json
  ⑤       → <out-root>/validate/<sensor>/VALREPORT_*.json
  로그    → <out-root>/logs/<task_id>.stdout.jsonl / .stderr.log
  매니페스트 → <out-root>/chain_manifest.json (태스크별 exit code·소요시간·산출 경로 누적)

멱등성:
  · ② 는 러너가 완료 마커(WB_*_qc.json status=COMPLETE)를 보고 파일 수준에서 스킵한다
    (씬당 CPU ~610 s 라 컨테이너 기동 자체를 아낀다). --force 로 무시.
  · ③④⑤ 는 항상 모듈을 호출하고 모듈 자체의 SKIP(입력 지문·모델 sha 대조)에 맡긴다
    — 입력 집합이 바뀌면(예: WB 1씬 → 4씬) 마커가 있어도 모듈이 재계산한다.
    파일 존재만으로 러너가 스킵하면 이 재계산이 막히므로 하지 않는다. 호출 비용은 수 초다.

사용례:
  python3 scripts/wbms_chain_runner.py                       # 전체 8씬, wb→wlwa→fused
  python3 scripts/wbms_chain_runner.py --steps wlwa,fused --stations gupo
  python3 scripts/wbms_chain_runner.py --scenes 20200416 --steps wb
  python3 scripts/wbms_chain_runner.py --dry-run             # 커맨드만 출력

주의: ③ 은 --gpu 인자가 없다(딥러닝 미사용, 매뉴얼 §2.6). ②④⑤ 는 --gpu <장치>.
장치 선택: 러너 --gpu {cpu,gpu,auto} (기본 cpu) — gpu/auto 는 nvidia-smi 여유 VRAM
프리플라이트 후 docker run 에 --gpus device=0 + TF_FORCE_GPU_ALLOW_GROWTH 를 전달한다.
이 서버 GPU 는 vucatcher 상주 서비스와 공유이므로 auto 권장.
기존 코드 무수정 · 호스트 파이썬 표준 라이브러리만 사용.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys
import time

# ── 고정 경로 ────────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(REPO, "data", "incoming", "handover")          # 인수 패키지
IMAGE = "wbms:0.13"
TB_CODE = "Busan"

SENSORS = {
    "iceye": {
        "sensor_name": "ICEYE",
        "pre_dir": os.path.join(H, "05_l1_pre", "iceye_pre"),
        "weights": "/model/WBMS_SAR_ICEYE.h5",
        # SAR 필수 — 없으면 detect_water 가 정당 실패한다
        "mask_tpl": "/aux/Masks/Img_DEM_3class_utm/busan_{date}_mask3_utm.tif",
        "mask_host_tpl": os.path.join(
            H, "06_aux", "Masks", "Img_DEM_3class_utm", "busan_{date}_mask3_utm.tif"),
        "wb_subdir": "wb",
        # ③ DEM: 날짜별 격자 정렬본 디렉터리 — calc_wlwa 가 <date>_DEM_32652.tif 를 고른다
        "dem": "/aux/DEM_busan_ngii3m/icegrid",
        "gt_dir": "/aux/Labels_GT",
        "mask_dir": "/aux/Masks/Img_DEM_3class_utm",
    },
    "planet": {
        "sensor_name": "PlanetScope",
        "pre_dir": os.path.join(H, "05_l1_pre", "planet_pre"),
        "weights": "/model/WBMS_Optic_PlanetScope.h5",
        "mask_tpl": None,                 # 광학은 마스크 불필요
        "mask_host_tpl": None,
        "wb_subdir": "wb_planet",
        # ③ DEM: 광학 Pre 는 NGII 3 m 단일 격자 — 단일 파일을 그대로 준다
        "dem": "/aux/DEM_busan_ngii3m/ngii_busan_32652_3m_topofilled.tif",
        "gt_dir": "/aux/Labels_GT_planet",
        "mask_dir": None,
    },
}
STATIONS_ALL = ["jeongcheon", "hupo", "gimhae", "gupo"]   # aoi_busan.geojson + 번들 loc_index
STEPS_ALL = ["wb", "wlwa", "fused", "validate"]
TIMEOUTS = {"wb": 5400, "wlwa": 900, "fused": 1800, "validate": 21600}


# ── 유틸 ────────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def discover_scenes() -> list[dict]:
    """05_l1_pre 에서 씬 목록을 만든다: [{sensor, date, stamp, input_host}]"""
    scenes = []
    for sensor, sc in SENSORS.items():
        pat = os.path.join(sc["pre_dir"], "*", "*",
                           f"Pre_{TB_CODE}_{sc['sensor_name']}_*.tif")
        for p in sorted(glob.glob(pat)):
            base = os.path.basename(p)
            if base.endswith("_lsmap.tif"):
                continue
            m = re.match(rf"Pre_{TB_CODE}_{sc['sensor_name']}_(\d{{8}}(?:T\d{{6}})?)\.tif$",
                         base)
            if not m:
                continue
            stamp = m.group(1)
            scenes.append({"sensor": sensor, "date": stamp[:8], "stamp": stamp,
                           "input_host": p})
    return scenes


def load_manifest(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            backup = path + ".corrupt_" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            os.replace(path, backup)
            print(f"[warn] 매니페스트 파싱 실패 — {backup} 로 치우고 새로 만든다",
                  file=sys.stderr)
    return {"created": now_iso(), "image": IMAGE, "tasks": {}}


def save_manifest(path: str, man: dict) -> None:
    man["updated"] = now_iso()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def record(man_path: str, man: dict, task_id: str, rec: dict) -> None:
    """디스크 상태를 다시 읽어 병합 후 저장 — 동시 실행자의 기록을 지우지 않는다.
    save_manifest 의 os.replace 가 inode 를 바꾸므로 매니페스트 자체가 아니라
    고정 사이드카 파일을 flock 한다."""
    import fcntl
    with open(man_path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        disk = load_manifest(man_path)
        ent = disk["tasks"].setdefault(task_id, {"runs": []})
        ent["runs"].append(rec)
        ent["last_status"] = rec["status"]
        ent["last_ts"] = rec["ts"]
        save_manifest(man_path, disk)
    man["tasks"][task_id] = ent  # 호출자 인메모리 뷰 동기화


def rm_container(name: str) -> None:
    """이전 잔여 컨테이너(동명) 정리 — 없는 경우 조용히 지나간다."""
    subprocess.run(["docker", "rm", "-f", name],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_events(stdout_path: str) -> tuple[dict, list[str]]:
    """모듈 stdout JSONL 에서 done 이벤트와 artifact 경로(컨테이너 경로)를 뽑는다."""
    done, artifacts = {}, []
    try:
        with open(stdout_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") == "done":
                    done = ev
                elif ev.get("type") == "artifact":
                    artifacts.append(ev.get("path", ""))
    except OSError:
        pass
    return done, artifacts


def map_out(paths: list[str], host_out: str) -> list[str]:
    """컨테이너 /out 경로를 호스트 경로로 치환한다."""
    out = []
    for p in paths:
        if p.startswith("/out/"):
            out.append(os.path.join(host_out, p[len("/out/"):]))
        elif p == "/out":
            out.append(host_out)
        else:
            out.append(p)
    return out


def run_task(*, step: str, task_id: str, cname: str, cmd: list[str],
             host_out: str, log_dir: str, man_path: str, man: dict,
             dry: bool) -> dict:
    """docker run 1회 실행 + 매니페스트 기록. 반환 = 기록 레코드."""
    cmd_str = " ".join(cmd)
    if dry:
        print(f"[dry-run] {task_id}\n  {cmd_str}")
        return {"status": "dry_run"}

    os.makedirs(log_dir, exist_ok=True)
    so = os.path.join(log_dir, f"{task_id}.stdout.jsonl")
    se = os.path.join(log_dir, f"{task_id}.stderr.log")
    rm_container(cname)
    t0 = time.monotonic()
    ts = now_iso()
    timeout = TIMEOUTS.get(step, 3600)
    status, exit_code = "failed", None
    try:
        with open(so, "w") as fo, open(se, "w") as fe:
            proc = subprocess.run(cmd, stdout=fo, stderr=fe, timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        rm_container(cname)
        status = "timeout"
    elapsed = round(time.monotonic() - t0, 1)

    done, artifacts = parse_events(so)
    counts = done.get("counts", {})
    if exit_code == 0:
        if counts.get("skipped"):
            status = "module_skip"
        elif done.get("status") == "success" or counts.get("complete"):
            status = "success"
        else:
            status = "success_no_output"    # 예: calc_wlwa NO_INPUT (exit 0, 실패 아님)
    rec = {"ts": ts, "step": step, "status": status, "exit_code": exit_code,
           "elapsed_s": elapsed, "container": cname, "cmd": cmd_str,
           "counts": counts, "outputs": map_out(artifacts, host_out),
           "stdout": so, "stderr": se}
    record(man_path, man, task_id, rec)
    tag = {"success": "ok", "module_skip": "skip(module)"}.get(status, status)
    print(f"[{tag}] {task_id} exit={exit_code} {elapsed}s "
          + (f"→ {rec['outputs']}" if rec["outputs"] else ""))
    return rec


# ── 단계별 태스크 구성 ──────────────────────────────────────────────────────
RUN_DEVICE = "cpu"  # main() 의 --gpu 프리플라이트 결과로 설정된다


def gpu_free_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def resolve_device(mode: str, min_free_mib: int) -> str:
    """gpu/auto 요청 시 여유 VRAM 프리플라이트. 이 서버 GPU 는 vucatcher
    상주 서비스(triton·DeepStream)와 공유라 여유분이 수시로 변한다."""
    if mode == "cpu":
        return "cpu"
    free = gpu_free_mib()
    if free is not None and free >= min_free_mib:
        print(f"[gpu] 여유 VRAM {free} MiB ≥ {min_free_mib} — GPU 사용")
        return "gpu"
    msg = (f"[gpu] 여유 VRAM 부족(free={free} MiB < {min_free_mib} MiB) — "
           "상주 서비스 조정 후 재시도 필요")
    if mode == "gpu":
        raise SystemExit(msg + " · --gpu gpu 강제 모드라 중단")
    print(msg + " · auto 모드라 CPU 폴백")
    return "cpu"


def oom_in_log(stdout_path: str) -> bool:
    """모듈 stdout JSONL 에 GPU OOM(RESOURCE_EXHAUSTED) 흔적이 있는지."""
    try:
        with open(stdout_path, encoding="utf-8") as f:
            return "RESOURCE_EXHAUSTED" in f.read()
    except OSError:
        return False


def docker_base(cname: str, mounts: list[tuple[str, str, bool]],
                device: str | None = None) -> list[str]:
    dev = device or RUN_DEVICE
    cmd = ["docker", "run", "--rm", "--name", cname]
    # 산출물이 root 소유로 떨어지면 실행 계정이 재실행·정리를 못 한다
    # (backend docker_prefix 와 동일한 이유 — 서버 실측에서 확인된 결함)
    cmd += ["--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp"]
    if dev == "gpu":
        # TF 기본은 가용 VRAM 전부 선점 — growth 로 필요한 만큼만 할당해
        # 동거 서비스와의 공존을 지킨다. cuda_malloc_async 는 단편화 완화용.
        cmd += ["--gpus", "device=0",
                "-e", "TF_FORCE_GPU_ALLOW_GROWTH=true",
                "-e", "TF_GPU_ALLOCATOR=cuda_malloc_async"]
    cmd += ["--entrypoint", "python3"]
    for host, cont, ro in mounts:
        cmd += ["-v", f"{host}:{cont}" + (":ro" if ro else "")]
    cmd += [IMAGE]
    return cmd


def wb_expected_qc(out_root: str, sc: dict, scene: dict) -> str:
    return os.path.join(out_root, sc["wb_subdir"], scene["date"][:4], scene["date"],
                        f"WB_{TB_CODE}_{sc['sensor_name']}_{scene['stamp']}_qc.json")


def qc_complete(path: str) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("status") == "COMPLETE"
    except Exception:
        return False


def wb_complete_any(out_root: str, sensor: str) -> bool:
    sub = SENSORS[sensor]["wb_subdir"]
    for qc in glob.glob(os.path.join(out_root, sub, "**", "WB_*_qc.json"),
                        recursive=True):
        if qc_complete(qc):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="WBMS ②③④⑤ CPU 체인 러너 (docker wbms:0.13)")
    ap.add_argument("--scenes", default="",
                    help="쉼표구분 YYYYMMDD 목록 (기본: 05_l1_pre 의 전체 8씬)")
    ap.add_argument("--steps", default="wb,wlwa,fused",
                    help=f"쉼표구분 {STEPS_ALL} (기본 wb,wlwa,fused)")
    ap.add_argument("--stations", default=",".join(STATIONS_ALL),
                    help="③④ 지점 목록 (기본 전체 4지점)")
    ap.add_argument("--out-root", default=os.path.join(REPO, "data", "wbms_runs"),
                    help="산출 루트 (기본 data/wbms_runs)")
    ap.add_argument("--force", action="store_true",
                    help="러너 스킵 무시 + 모듈에 --force 전달(완료 마커 무시)")
    ap.add_argument("--gpu", choices=["cpu", "gpu", "auto"], default="cpu",
                    help="추론 장치 (기본 cpu). gpu=여유 VRAM 부족 시 중단, "
                         "auto=부족 시 CPU 폴백")
    ap.add_argument("--gpu-min-free-mib", type=int, default=3000,
                    help="GPU 사용에 필요한 최소 여유 VRAM (기본 3000 MiB)")
    ap.add_argument("--dry-run", action="store_true", help="커맨드만 출력")
    a = ap.parse_args()

    global RUN_DEVICE
    RUN_DEVICE = resolve_device(a.gpu, a.gpu_min_free_mib)

    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS_ALL]
    if bad:
        print(f"알 수 없는 단계 {bad} — 가능: {STEPS_ALL}", file=sys.stderr)
        return 1
    stations = [s.strip().lower() for s in a.stations.split(",") if s.strip()]
    bad = [s for s in stations if s not in STATIONS_ALL]
    if bad:
        print(f"알 수 없는 지점 {bad} — 가능: {STATIONS_ALL}", file=sys.stderr)
        return 1

    out_root = os.path.abspath(a.out_root)
    log_dir = os.path.join(out_root, "logs")
    man_path = os.path.join(out_root, "chain_manifest.json")
    os.makedirs(out_root, exist_ok=True)
    man = load_manifest(man_path)

    all_scenes = discover_scenes()  # ③ 범위 스템은 항상 전체 인벤토리 기준
    scenes = all_scenes
    if a.scenes:
        want = {s.strip() for s in a.scenes.split(",") if s.strip()}
        unknown = want - {s["date"] for s in scenes}
        if unknown:
            print(f"05_l1_pre 에 없는 씬 날짜: {sorted(unknown)}", file=sys.stderr)
            return 1
        scenes = [s for s in scenes if s["date"] in want]
    if not scenes:
        print("대상 씬이 없다", file=sys.stderr)
        return 1
    sel_sensors = sorted({s["sensor"] for s in scenes})
    print(f"대상: {len(scenes)}씬 {sel_sensors} · 단계 {steps} · 지점 {stations}"
          f" · out={out_root}")

    failed = []

    # ── ② detect_water — 씬별 ────────────────────────────────────────────
    if "wb" in steps:
        for scene in sorted(scenes, key=lambda s: (s["sensor"], s["date"])):
            sc = SENSORS[scene["sensor"]]
            task_id = f"wb_{scene['sensor']}_{scene['date']}"
            qc = wb_expected_qc(out_root, sc, scene)
            if not a.force and qc_complete(qc):
                print(f"[skip(runner)] {task_id} — 완료 마커 존재: {qc}")
                if not a.dry_run:
                    record(man_path, man, task_id,
                           {"ts": now_iso(), "step": "wb", "status": "runner_skip",
                            "exit_code": 0, "elapsed_s": 0.0, "container": None,
                            "cmd": None, "counts": {}, "outputs": [qc],
                            "stdout": None, "stderr": None})
                continue
            if sc["mask_host_tpl"]:
                mh = sc["mask_host_tpl"].format(date=scene["date"])
                if not os.path.exists(mh):
                    print(f"[failed] {task_id} — SAR 마스크 없음: {mh}", file=sys.stderr)
                    failed.append(task_id)
                    continue
            host_out = os.path.join(out_root, sc["wb_subdir"])
            os.makedirs(host_out, exist_ok=True)
            cname = f"wbms_chain_wb_{scene['date']}"
            rel = os.path.relpath(scene["input_host"], H)
            def build_wb_cmd(device: str) -> list[str]:
                c = docker_base(cname, [
                    (os.path.join(H, "03_model"), "/model", True),
                    (H, "/data", True),
                    (os.path.join(H, "06_aux"), "/aux", True),
                    (host_out, "/out", False),
                ], device=device) + ["/WBMS/wbms_modules/detect_water.py",
                      "--input", f"/data/{rel}",
                      "--sensor", scene["sensor"], "--testbed", "busan",
                      "--image_date", scene["date"],
                      "--weights", sc["weights"]]
                if sc["mask_tpl"]:
                    c += ["--mask", sc["mask_tpl"].format(date=scene["date"])]
                c += ["--gpu", device, "--output_dir", "/out"]
                if a.force:
                    c += ["--force"]
                return c

            rec = run_task(step="wb", task_id=task_id, cname=cname,
                           cmd=build_wb_cmd(RUN_DEVICE), host_out=host_out,
                           log_dir=log_dir, man_path=man_path, man=man,
                           dry=a.dry_run)
            if (rec.get("status") == "failed" and RUN_DEVICE == "gpu"
                    and a.gpu == "auto"
                    and oom_in_log(os.path.join(log_dir,
                                                f"{task_id}.stdout.jsonl"))):
                # 공유 GPU 여유가 이 모델의 배치에 부족 — 이 태스크만 CPU 재시도
                print(f"[gpu] {task_id} VRAM OOM — CPU 로 재시도")
                rec = run_task(step="wb", task_id=task_id, cname=cname,
                               cmd=build_wb_cmd("cpu"), host_out=host_out,
                               log_dir=log_dir, man_path=man_path, man=man,
                               dry=a.dry_run)
            if rec.get("status") in ("failed", "timeout", "success_no_output"):
                failed.append(task_id)

    # ── ③ calc_wlwa — 센서×지점 (③ 은 --gpu 인자가 없다) ─────────────────
    if "wlwa" in steps:
        for sensor in sel_sensors:
            sc = SENSORS[sensor]
            # --scenes 부분 실행이어도 범위 스템이 갈라지지 않도록 전체 인벤토리 기준
            dates = sorted(s["date"] for s in all_scenes if s["sensor"] == sensor)
            if not a.dry_run and not wb_complete_any(out_root, sensor):
                print(f"[no_input] wlwa_{sensor} — 완료 WB 없음 "
                      f"({os.path.join(out_root, sc['wb_subdir'])})")
                continue
            wb_host = os.path.join(out_root, sc["wb_subdir"])
            for loc in stations:
                task_id = f"wlwa_{sensor}_{loc}"
                host_out = os.path.join(out_root, "wlwa", sensor, loc)
                os.makedirs(host_out, exist_ok=True)
                # 범위가 다른 잔존 WLWA 스템은 ④가 잘못 병합하지 않게 격리
                expected = f"_{dates[0]}_{dates[-1]}"
                for old_qc in sorted(glob.glob(os.path.join(host_out, "WLWA_*_qc.json"))):
                    if expected not in os.path.basename(old_qc):
                        stem = os.path.basename(old_qc)[: -len("_qc.json")]
                        sup = os.path.join(host_out, "_superseded")
                        os.makedirs(sup, exist_ok=True)
                        for p in sorted(glob.glob(os.path.join(host_out, stem + "*"))):
                            os.replace(p, os.path.join(sup, os.path.basename(p)))
                        print(f"[sweep] 범위 불일치 WLWA 스템 격리: {stem}")
                cname = f"wbms_chain_wlwa_{sensor}_{loc}"
                cmd = docker_base(cname, [
                    (os.path.join(H, "06_aux"), "/aux", True),
                    (wb_host, "/wb", True),
                    (host_out, "/out", False),
                ]) + ["/WBMS/wbms_modules/calc_wlwa.py",
                      "--wb_result_dir", "/wb",
                      "--sensor", sensor, "--testbed", "busan",
                      "--loc_id", loc,
                      "--start_date", dates[0], "--end_date", dates[-1],
                      "--dem", sc["dem"],
                      "--output_dir", "/out"]
                if a.force:
                    cmd += ["--force"]
                rec = run_task(step="wlwa", task_id=task_id, cname=cname, cmd=cmd,
                               host_out=host_out, log_dir=log_dir,
                               man_path=man_path, man=man, dry=a.dry_run)
                if rec.get("status") in ("failed", "timeout"):
                    failed.append(task_id)

    # ── ④ Correct — 지점별 융합 (모델 번들은 이미지 동봉본 자동 선택) ─────
    if "fused" in steps:
        for loc in stations:
            task_id = f"fused_{loc}"
            sar_host = os.path.join(out_root, "wlwa", "iceye", loc)
            optic_host = os.path.join(out_root, "wlwa", "planet", loc)
            os.makedirs(sar_host, exist_ok=True)
            os.makedirs(optic_host, exist_ok=True)
            have = (glob.glob(os.path.join(sar_host, "WLWA_*_qc.json"))
                    + glob.glob(os.path.join(optic_host, "WLWA_*_qc.json")))
            if not a.dry_run and not have:
                print(f"[no_input] {task_id} — WLWA 산출 없음 ({sar_host} / {optic_host})")
                continue
            host_out = os.path.join(out_root, "fused", loc)
            os.makedirs(host_out, exist_ok=True)
            cname = f"wbms_chain_fused_{loc}"
            cmd = docker_base(cname, [
                (os.path.join(H, "06_aux"), "/aux", True),
                (sar_host, "/sar", True),
                (optic_host, "/optic", True),
                (host_out, "/out", False),
            ]) + ["/WBMS/wbms_modules/Correct.py",
                  "--sar_result_dir", "/sar",
                  "--optic_result_dir", "/optic",
                  "--aws_csv", "/aux/busan_aws_2019_202004.csv",
                  "--testbed", "busan", "--loc_id", loc,
                  "--gpu", RUN_DEVICE, "--output_dir", "/out"]
            if a.force:
                cmd += ["--force"]
            rec = run_task(step="fused", task_id=task_id, cname=cname, cmd=cmd,
                           host_out=host_out, log_dir=log_dir,
                           man_path=man_path, man=man, dry=a.dry_run)
            if rec.get("status") in ("failed", "timeout"):
                failed.append(task_id)

    # ── ⑤ validate — 센서별 (선택 단계) ───────────────────────────────────
    if "validate" in steps:
        for sensor in sel_sensors:
            sc = SENSORS[sensor]
            task_id = f"validate_{sensor}"
            host_out = os.path.join(out_root, "validate", sensor)
            os.makedirs(host_out, exist_ok=True)
            cname = f"wbms_chain_validate_{sensor}"
            rel_pre = os.path.relpath(sc["pre_dir"], H)
            cmd = docker_base(cname, [
                (os.path.join(H, "03_model"), "/model", True),
                (H, "/data", True),
                (os.path.join(H, "06_aux"), "/aux", True),
                (host_out, "/out", False),
            ]) + ["/WBMS/wbms_modules/validate.py",
                  "--input_dir", f"/data/{rel_pre}",
                  "--gt_dir", sc["gt_dir"]]
            if sc["mask_dir"]:
                cmd += ["--mask_dir", sc["mask_dir"]]
            cmd += ["--weights", sc["weights"],
                    "--sensor", sensor, "--testbed", "busan",
                    "--gpu", RUN_DEVICE, "--output_dir", "/out"]
            if a.force:
                cmd += ["--force"]
            rec = run_task(step="validate", task_id=task_id, cname=cname, cmd=cmd,
                           host_out=host_out, log_dir=log_dir,
                           man_path=man_path, man=man, dry=a.dry_run)
            if rec.get("status") in ("failed", "timeout"):
                failed.append(task_id)

    if failed:
        print(f"\n실패 태스크 {len(failed)}: {failed}", file=sys.stderr)
        print(f"매니페스트: {man_path}", file=sys.stderr)
        return 2
    print(f"\n전 태스크 정상 종료 — 매니페스트: {man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
