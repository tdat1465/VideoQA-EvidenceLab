"""MoLab driver, launched from its terminal or a Python notebook cell.

Notebook Python 3.13 only orchestrates; inference uses isolated Python 3.12.14,
Torch 2.8 cu128 and the existing pinned backbone requirements. No Slurm spoofing.
"""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from evidencelab.config import source_fingerprint
from evidencelab.longvideo import validate_ram_root
from evidencelab.longvideo_config import DATASET_REVISION, LongVideoConfig
from evidencelab.longvideo_data import cgroup_memory, check_space
from evidencelab.notebook_backup import backup_run, restore_run
from evidencelab.store import atomic_json, run_lock


def environment(root, runtime=None):
    env = os.environ.copy()
    for name in ("TRANSFORMERS_CACHE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    paths = {
        "HF_HOME": "hf", "HF_HUB_CACHE": "hf/hub", "HUGGINGFACE_HUB_CACHE": "hf/hub",
        "HF_MODULES_CACHE": "hf/modules", "HF_DATASETS_CACHE": "hf/datasets", "HF_XET_CACHE": "hf/xet",
        "XDG_CACHE_HOME": "cache", "TMPDIR": "tmp", "PIP_CACHE_DIR": "pip", "UV_CACHE_DIR": "uv",
        "UV_PYTHON_INSTALL_DIR": "python", "UV_PYTHON_BIN_DIR": "bin", "TORCH_HOME": "torch",
        "PYTORCH_KERNEL_CACHE_PATH": "kernel", "TORCHINDUCTOR_CACHE_DIR": "inductor",
        "TORCH_EXTENSIONS_DIR": "extensions", "CUDA_CACHE_PATH": "cuda", "TRITON_CACHE_DIR": "triton",
        "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib",
    }
    for key, subdir in paths.items():
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    if runtime is not None:
        # /dev/shm is often noexec in notebook containers. Keep executable
        # Python/venv in ephemeral /tmp, with weights/video/cache still in RAM.
        for key, subdir in (("UV_PYTHON_INSTALL_DIR", "python"), ("UV_PYTHON_BIN_DIR", "bin")):
            path = runtime / subdir
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    env.update(EVIDENCELAB_EXECUTION="molab", PYTHONPATH=str(REPO / "src"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_CONFIG_FILE="/dev/null",
               HF_HUB_DISABLE_XET="1", HF_HUB_DOWNLOAD_TIMEOUT="120", OMP_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", UV_PYTHON_PREFERENCE="only-managed", UV_LINK_MODE="copy")
    return env


def child(command, env, stop_file, deadline, inference=False, stdout=None):
    if stop_file.exists():
        raise InterruptedError("Stop requested")
    check_space(Path(env["TMPDIR"]), 0)
    print("Running: " + " ".join(str(x) for x in command), flush=True)
    proc = subprocess.Popen([str(x) for x in command], env=env, cwd=REPO,
                            start_new_session=True, stdout=stdout)
    def send(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
    old = {}
    def stop(signum, frame):
        stop_file.touch()
    for name in (signal.SIGINT, signal.SIGTERM):
        old[name] = signal.signal(name, stop)
    signalled = None
    try:
        while proc.poll() is None:
            memory = cgroup_memory()
            if not memory or memory["current_bytes"] >= min(memory["limit_bytes"], 90 * 1024**3):
                send(signal.SIGKILL)
                proc.wait()
                raise MemoryError("MoLab RAM watchdog stopped subprocess; committed results remain in SQLite")
            now = time.monotonic()
            if (stop_file.exists() or now >= deadline) and signalled is None:
                send(signal.SIGUSR1 if inference else signal.SIGTERM)
                signalled = now
            if signalled is not None and now - signalled > 180:
                send(signal.SIGKILL)
            time.sleep(1)
        if proc.returncode not in ((0, 75) if inference else (0,)):
            raise subprocess.CalledProcessError(proc.returncode, command)
        return proc.returncode
    finally:
        if proc.poll() is None:
            send(signal.SIGKILL)
            proc.wait()
        for number, handler in old.items():
            signal.signal(number, handler)


def session(workspace, backend):
    state = workspace / f"session-{backend}.json"
    fingerprint = source_fingerprint()
    if state.exists():
        info = json.loads(state.read_text())
        root = Path(info["ram_root"])
        runtime = Path(info["runtime_root"])
        if root.is_symlink() or root.parent != Path("/dev/shm") or not root.name.startswith("videoqa.molab."):
            raise ValueError("Invalid session RAM path")
        if runtime.is_symlink() or runtime.parent != Path("/tmp") or not runtime.name.startswith("videoqa.molab.runtime."):
            raise ValueError("Invalid session runtime path")
        if root.is_dir() and runtime.is_dir():
            if info["source"] != fingerprint:
                raise ValueError("Checkout changed; use a new workspace and new runs")
            return root, state, info
    root = Path(tempfile.mkdtemp(prefix="videoqa.molab.", dir="/dev/shm"))
    runtime = Path(tempfile.mkdtemp(prefix="videoqa.molab.runtime.", dir="/tmp"))
    info = {"ram_root": str(root), "runtime_root": str(runtime), "source": fingerprint,
            "ready": False, "backend": backend}
    atomic_json(state, info)
    return root, state, info


def setup(workspace, backend, config):
    root, state, info = session(workspace, backend)
    validate_ram_root(root)
    runtime = Path(info["runtime_root"])
    env = environment(root, runtime)
    stop_file = workspace / "stop.request"
    deadline = time.monotonic() + 3 * 3600
    py = runtime / "venv/bin/python"
    if info["ready"]:
        child([py, "-m", "evidencelab.longvideo", "doctor", "--config", config], env, stop_file, deadline)
        return root, env, py
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("MoLab setup requires uv on PATH")
    if not py.is_file():
        child([uv, "venv", "--python", "3.12.14", "--seed", runtime / "venv"], env, stop_file, deadline)
    pip = [py, "-m", "pip", "install", "--no-cache-dir"]
    child(pip + ["pip==25.2", "setuptools==80.9.0", "wheel==0.45.1",
                 "requests==2.32.5", "huggingface-hub==0.35.3"], env, stop_file, deadline)
    # Dataset authorization and Range support first, before large package/model downloads.
    child([py, "-m", "evidencelab.longvideo", "prepare", "--config", config,
           "--ram-root", root / "data"], env, stop_file, deadline)
    child(pip + ["torch==2.8.0", "torchvision==0.23.0", "--index-url",
                 "https://download.pytorch.org/whl/cu128"], env, stop_file, deadline)
    requirements = "requirements-longvideo-molmo.txt" if backend == "molmo2" else "requirements-longvideo-llava.txt"
    child(pip + ["-r", REPO / requirements], env, stop_file, deadline)
    child(pip + ["--no-build-isolation", "--no-deps", "-e", REPO], env, stop_file, deadline)
    if backend == "llava_video":
        if not (root / "llava/pyproject.toml").exists():
            child([py, REPO / "scripts/fetch_llava_source.py", "--target", root / "llava"], env, stop_file, deadline)
        child(pip + ["--no-build-isolation", "--no-deps", "-e", root / "llava"], env, stop_file, deadline)
        child([py, REPO / "scripts/smoke_llava_cpu.py"], env, stop_file, deadline)
    child([py, "-m", "pip", "check"], env, stop_file, deadline)
    child([py, "-m", "evidencelab.longvideo", "doctor", "--config", config], env, stop_file, deadline)
    info["ready"] = True
    atomic_json(state, info)
    print("Setup complete; environment and downloads can be reused in this notebook session", flush=True)
    return root, env, py


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["setup", "run", "stop", "backup", "restore"])
    parser.add_argument("--workspace", type=Path, default=Path("/marimo/videoqa-molab"))
    parser.add_argument("--config", type=Path, default=REPO / "configs/lvb_molmo2_lens64.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-samples", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=10800)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    index = workspace / "metadata" / DATASET_REVISION / "index.json"
    if args.command == "stop":
        (workspace / "stop.request").touch()
        print("Stop requested. Wait for process exit and backup before shutting down MoLab.")
        return 0
    if args.command in {"backup", "restore"}:
        if args.output is None or args.archive is None:
            parser.error("backup/restore require --output and --archive")
        if args.command == "backup":
            chosen_index = args.output / "archive-index.json"
            count = backup_run(args.output, args.archive, chosen_index if chosen_index.exists() else index)
        else:
            count = restore_run(args.archive, args.output)
        print(f"{args.command}: {count} committed questions; {args.archive}")
        return 0
    if sys.platform != "linux" or os.environ.get("SLURM_JOB_ID"):
        parser.error("MoLab profile requires a Linux notebook, outside Slurm")
    if not os.environ.get("HF_TOKEN"):
        parser.error("Add HF_TOKEN to MoLab Secrets, then restart/reload the notebook environment")
    if not 1 <= args.max_seconds <= 36000 or args.max_new_samples < 0:
        parser.error("Use 1..36000 seconds and nonnegative max-new-samples")
    os.environ["EVIDENCELAB_EXECUTION"] = "molab"
    config_path = args.config.resolve()
    config = LongVideoConfig.load(config_path)
    if config.backend not in {"molmo2", "llava_video"}:
        parser.error("Use a real LongVideoBench backbone")
    if args.command == "run":
        if args.output is None:
            parser.error("run requires --output")
        exists = (args.output / "results.sqlite3").exists()
        if exists != args.resume:
            parser.error("Existing journal requires --resume; a new run must omit --resume")
    with run_lock(workspace / "gpu-session"):
        stop_file = workspace / "stop.request"
        stop_file.unlink(missing_ok=True)
        if args.command == "run":
            args.output.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output / "session-status.json", {"state": "starting", "exit_code": None})
        root, env, py = setup(workspace, config.backend, config_path)
        if args.command == "setup":
            return 0
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        if output.is_relative_to(root):
            raise ValueError("Results must be outside the temporary RAM directory")
        status = 1
        try:
            child([py, "-m", "evidencelab.longvideo", "prepare", "--config", config_path,
                   "--ram-root", root / "data"], env, stop_file, time.monotonic() + 300)
            with (output / "environment.latest.txt").open("w") as f:
                child([py, "-m", "pip", "freeze"], env, stop_file, time.monotonic()+60, stdout=f)
            chosen_index = output / "archive-index.json"
            if not chosen_index.exists():
                chosen_index = index
            command = [py, "-m", "evidencelab.longvideo", "run", "--config", config_path,
                       "--annotations", root / "data/lvb_val.json", "--ram-root", root / "data",
                       "--output", output, "--index", chosen_index,
                       "--max-seconds", str(args.max_seconds), "--max-new-samples", str(args.max_new_samples)]
            if args.resume:
                command.append("--resume")
            status = child(command, env, stop_file, time.monotonic()+args.max_seconds, inference=True)
            return status
        finally:
            for name in ("active-video.mp4", "active-video.partial"):
                (root / "data" / name).unlink(missing_ok=True)
            archive = workspace / "backups" / f"{output.name}.zip"
            if (output / "results.sqlite3").exists():
                count = backup_run(output, archive, chosen_index if 'chosen_index' in locals() else index)
                print(f"BACKUP {count} questions: {archive}\nDOWNLOAD THIS FILE before ending the notebook session.", flush=True)
            atomic_json(output / "session-status.json", {"state": "stopped", "exit_code": status, "backup": str(archive)})


if __name__ == "__main__":
    raise SystemExit(main())
