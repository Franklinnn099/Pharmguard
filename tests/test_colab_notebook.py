"""Static checks on the Colab notebook's Option 2 cells (camera server on the GPU + Cloudflare tunnel).
The cells only run on Colab, so this checks they compile, run in the required order and keep the safety rules."""

import json
import os
import re

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "PharmaGuard_Colab.ipynb")


def _option2_cells():
    cells = json.load(open(NB, encoding="utf-8"))["cells"]
    start = next(i for i, c in enumerate(cells) if "<!-- pg-option2 -->" in "".join(c["source"]))
    code = []
    for c in cells[start + 1:]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and src.startswith("# O2-"):
            code.append(src)
        elif c["cell_type"] == "markdown":
            break
    return code


def test_option2_cells_compile_and_run_in_order():
    cells = _option2_cells()
    for src in cells:
        compile(src, "<cell>", "exec")
    steps = [int(re.match(r"# O2-(\d+)", s).group(1)) for s in cells]
    assert steps == list(range(1, 13))
    # What each step must do, in the required order.
    expect = ["torch.cuda.is_available()", "drive.mount", "git", "requirements.txt", "PG_ZONES_DIR", "PG_ACCESS_KEY",
              "\nstart_camera_server()", "trycloudflare", "qrcode.make", "gradio.live", "api(\"/api/cameras\")",
              "stop_camera_server()"]
    for i, marker in enumerate(expect):
        assert marker in cells[i], (i + 1, marker)


def test_measure_cells_go_through_the_tunnel_with_the_key():
    cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
    m = [s for s in cells if s.startswith("# M-")]
    assert len(m) == 2
    for src in m:
        compile(src, "<cell>", "exec")
    assert 'TUNNEL_URL.replace("https://", "wss://")' in m[0] and '"--key", PG_ACCESS_KEY' in m[0]
    assert 'for items in ("1", "0")' in m[0] and 'os.environ["PG_ITEMS"] = "1"' in m[0]   # ends on the demo setting


def test_option2_keeps_safety_rules():
    cells = _option2_cells()
    joined = "\n".join(cells)
    assert "raise SystemExit" in cells[0]                                   # no GPU = clear failure
    assert "-c\", KEEP_TORCH" in joined                                     # never reinstall Colab's torch
    assert 'PG_HOST="127.0.0.1"' in joined                                  # only reachable through the tunnel
    assert '"cuda" not in str(info["model"])' in joined                     # refuse to run off the GPU
    data_dir = re.search(r'PHARMAGUARD_DATA_DIR="([^"]+)"', joined).group(1)
    assert not data_dir.startswith("/content/drive")                       # SQLite never opened on Drive
    assert 'mirror_dir"] = f"{DRIVE}/camera"' in joined                     # DB copied to Drive via backup API
    assert "http.extraHeader" in joined and "x-access-token" in joined      # token never stored in .git/config
    assert "secrets.token_urlsafe" in joined                                # random key per session
    assert "key={K}" in joined                                              # links carry the key
    assert "PG_CORS_ORIGINS" in joined and "DASHBOARD_URL" in joined
