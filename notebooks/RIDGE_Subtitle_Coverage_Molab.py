import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import marimo as mo
    return Path, json, mo, os, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # RIDGE với phụ đề và bảo vệ độ phủ thời gian

    So sánh bốn nhánh trên Molmo2-4B và LongVideoBench validation.
    `ridge_sigma2`, `eventcover_sigma2`, `multi_anchor_v1`, `protected_v2`.

    V2 giữ đại diện của 16 mốc trải đều và frame đầu/cuối của tập nền.
    Mỗi frame phụ đề mới thay một frame gần nó trong cùng vùng thời gian.
    Nếu vùng đó hết frame có thể thay, ưu tiên frame có điểm BLIP thấp
    ở vùng còn nhiều bằng chứng. Tất cả nhánh được trả lời lại trong run này.

    Cần một GPU CUDA, môi trường của repo, file annotation và `anchors.json`
    từ thí nghiệm trước, hoặc archive `subtitles.tar` chính thức.
    Đặt `HF_TOKEN` trong Secrets/environment của Molab với quyền đọc dataset.
    Dữ liệu video và pixel dùng `/dev/shm`. Model dùng cache Hugging Face của phiên.
    """)
    return


@app.cell
def _(Path, mo):
    _default = Path("/marimo/ridge-canonical/repo-fresh")
    repo_path = mo.ui.text(value=str(_default), label="Checkout repo có code v2", full_width=True)
    annotation_path = mo.ui.text(value="/marimo/ridge-canonical/subtitle-pilot/lvb_val.json",
                                 label="lvb_val.json", full_width=True)
    anchors_path = mo.ui.text(value="/marimo/ridge-canonical/subtitle-full-v1/anchors.json",
                              label="anchors.json cũ, để trống nếu dùng subtitles.tar", full_width=True)
    subtitles_path = mo.ui.text(value="/marimo/ridge-canonical/subtitle-pilot/subtitles.tar",
                                label="Archive phụ đề khi không có anchors.json", full_width=True)
    output_path = mo.ui.text(value="/marimo/ridge-canonical/subtitle-protected-v2",
                             label="Thư mục kết quả v2 để resume", full_width=True)
    mode = mo.ui.dropdown(options=["pilot20", "full1337"], value="pilot20", label="Chế độ")
    duration = mo.ui.number(start=10, stop=2880, step=10, value=120, label="Phút mỗi phiên")
    precision = mo.ui.dropdown(options=["auto", "bfloat16", "float16"], value="auto", label="Precision")
    mo.vstack([repo_path, annotation_path, anchors_path, subtitles_path, output_path,
               mo.hstack([mode, duration, precision])])
    return annotation_path, anchors_path, duration, mode, output_path, precision, repo_path, subtitles_path


@app.cell
def _(Path, annotation_path, anchors_path, duration, mode, output_path, precision, repo_path, subtitles_path, sys):
    worker = Path(repo_path.value).expanduser() / "experiments/subtitle_coverage_v2/run.py"
    command = [sys.executable, "-u", str(worker), "--annotation", annotation_path.value,
               "--output", output_path.value, "--dtype", precision.value,
               "--max-new-questions", "20" if mode.value == "pilot20" else "0",
               "--session-minutes", str(int(duration.value))]
    if anchors_path.value.strip():
        command.extend(["--anchors", anchors_path.value])
    else:
        command.extend(["--subtitles", subtitles_path.value])
    return command, worker


@app.cell
def _(Path, os, subprocess):
    def execute_worker(arguments, output_dir):
        directory = Path(output_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        with (directory / "notebook-session.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=environment)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
        if code not in (0, 75):
            raise RuntimeError(f"Worker dừng với mã {code}. Xem notebook-session.log.")
        return code
    return (execute_worker,)


@app.cell
def _(mo):
    preflight = mo.ui.run_button(label="1. Kiểm tra dữ liệu đầu vào")
    preflight
    return (preflight,)


@app.cell
def _(command, execute_worker, mo, output_path, preflight, worker):
    mo.stop(not preflight.value)
    if not worker.is_file():
        raise FileNotFoundError(f"Chưa có worker v2 tại {worker}. Chép code mới vào checkout trước.")
    execute_worker(command + ["--preflight"], output_path.value)
    return


@app.cell
def _(mo):
    launch = mo.ui.run_button(label="2. Chạy hoặc tiếp tục phiên GPU", kind="success")
    mo.vstack([launch, mo.md("""
    Pilot xử lý tối đa 20 câu chưa xong, ưu tiên câu khớp phụ đề để kiểm tra thay frame.
    Accuracy pilot phụ thuộc cách chọn này. Full tiếp tục cùng journal đến 1.337 câu.
    Mã thoát 75 là dừng có checkpoint. Bấm lại để tiếp tục.
    Thời lượng phiên được kiểm tra giữa các công đoạn, một forward đang chạy có thể kết thúc muộn hơn.
    """)])
    return (launch,)


@app.cell
def _(command, execute_worker, launch, mo, output_path, worker):
    mo.stop(not launch.value)
    if not worker.is_file():
        raise FileNotFoundError(worker)
    exit_code = execute_worker(command, output_path.value)
    mo.md("Đã hoàn thành toàn bộ run." if exit_code == 0 else "Đã lưu tiến độ. Có thể tiếp tục cùng thư mục.")
    return (exit_code,)


@app.cell
def _(mo):
    refresh = mo.ui.run_button(label="3. Đọc bảng kết quả đã lưu")
    refresh
    return (refresh,)


@app.cell
def _(Path, json, mo, output_path, refresh):
    mo.stop(not refresh.value)
    _path = Path(output_path.value) / "summary.json"
    mo.stop(not _path.is_file(), mo.md("Chưa có summary. Chờ worker kết thúc phiên."))
    _result = json.loads(_path.read_text())
    _rows = [{"Nhánh": arm, **values} for arm, values in _result["arms"].items()]
    _comparisons = [{"V2 so với": arm, **values}
                    for arm, values in _result["protected_v2_vs"].items()]
    mo.vstack([mo.md(f"Đã xong **{_result['completed_questions']}/1337 câu**."),
               mo.ui.table(_rows), mo.ui.table(_comparisons)])
    return


if __name__ == "__main__":
    app.run()
