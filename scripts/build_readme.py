"""Synchronize README tables, local badges, and the offline leaderboard from public data."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "docs/leaderboard/results.json"


def validate(data):
    models = data["models"]
    if not models or len({m["id"] for m in models}) != len(models):
        raise ValueError("Model IDs must be unique and the baseline must not be empty")
    keys = [metric["key"] for metric in data["metrics"]]
    if data["default_ranking"] not in keys:
        raise ValueError("Unknown default ranking metric")
    for model in models:
        for key in keys:
            value = model[key]
            if not isinstance(value, (int, float)) or not 0 <= value <= 100:
                raise ValueError(f"Invalid percentage: {model['id']}/{key}")
        if model["jacc"] > min(model["cacc"], model["gacc_joint"]):
            raise ValueError("Joint correctness cannot exceed either same-set marginal")
        if abs(model["paired"]["gt"] - model["paired"]["base"] - model["paired"]["delta_pp"]) > 0.011:
            raise ValueError("Paired delta exceeds the independently rounded tolerance")
    for key in ("arxiv", "dataset", "project"):
        if not data["links"][key].startswith("https://"):
            raise ValueError("Project links must use HTTPS")


def ranked(data):
    key = data["default_ranking"]
    result = []
    rank, last = 0, None
    for index, model in enumerate(sorted(data["models"], key=lambda m: (-m[key], m["name"])), 1):
        if model[key] != last:
            rank, last = index, model[key]
        result.append((rank, model))
    return result


def markdown_tables(data):
    metrics = data["metrics"]
    maxima = {metric["key"]: max(m[metric["key"]] for m in data["models"]) for metric in metrics}
    lines = ["| Rank | Model | GAcc (G) ↑ | CAcc (L) ↑ | GAcc (L) ↑ | JAcc (L) ↑ | SolveAcc ↑ |",
             "| :---: | :--- | ---: | ---: | ---: | ---: | ---: |"]
    for rank, model in ranked(data):
        scores = []
        for metric in metrics:
            key = metric["key"]
            value = f"{model[key]:.2f}"
            scores.append(f"**{value}**" if model[key] == maxima[key] else value)
        lines.append("| " + " | ".join([str(rank), model["name"], *scores]) + " |")
    paired = ["| Model | Base GAcc ↑ | +GT GAcc ↑ | Δ GAcc (pp) | Parents / probes |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    for model in data["models"]:
        p = model["paired"]
        paired.append(f"| {model['name']} | {p['base']:.2f} | {p['gt']:.2f} | {p['delta_pp']:+.2f} | {p['parents']} / {p['probes']} |")
    return "\n".join(lines), "\n".join(paired)


def html_table(data):
    metrics = data["metrics"]
    maxima = {m["key"]: max(row[m["key"]] for row in data["models"]) for m in metrics}
    head = '<tr><th scope="col" class="rank-col">Rank</th><th scope="col" class="model-col">Model</th>'
    for metric in metrics:
        selected = metric["key"] == data["default_ranking"]
        head += (f'<th scope="col" class="numeric{" selected" if selected else ""}" aria-sort="{"descending" if selected else "none"}">'
                 f'<button type="button" data-sort="{metric["key"]}">{escape(metric["label"])} {"↓" if selected else "↑"}'
                 f'<small>{escape(metric["scope"])}</small></button></th>')
    rows = []
    for rank, model in ranked(data):
        access = "API" if model["access"] == "api" else "Open weights"
        row = (f'<tr><td class="rank-col"><span class="rank{" top-rank" if rank <= 3 else ""}">{rank:02d}</span></td>'
               f'<td class="model-cell"><span class="model-name">{escape(model["name"])}</span>'
               f'<span class="access {"api" if model["access"] == "api" else "open"}">{access}</span></td>')
        for metric in metrics:
            key = metric["key"]
            selected = key == data["default_ranking"]
            row += (f'<td class="numeric{" selected" if selected else ""}{" best" if model[key] == maxima[key] else ""}">'
                    f'<span class="score">{model[key]:.2f}</span>')
            if selected:
                row += f'<span class="score-bar" aria-hidden="true"><i style="width:{model[key]}%"></i></span>'
            row += "</td>"
        rows.append(row + "</tr>")
    return head + "</tr>", "\n".join(rows)


def badge(label, value, color, left, right):
    width = left + right
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="30" viewBox="0 0 {width} 30" role="img" aria-label="{escape(label)}: {escape(value)}">
  <title>{escape(label)}: {escape(value)}</title>
  <defs><clipPath id="round"><rect width="{width}" height="30" rx="6"/></clipPath></defs>
  <g clip-path="url(#round)"><rect width="{width}" height="30" fill="#293448"/><rect x="{left}" width="{right}" height="30" fill="{color}"/></g>
  <g font-family="Arial,Helvetica,sans-serif" font-size="12" text-anchor="middle"><text x="{left/2}" y="19.5" fill="#edf2fa">{escape(label)}</text><text x="{left+right/2}" y="19.5" fill="#fff" font-weight="600">{escape(value)}</text></g>
</svg>
'''


def outputs(data):
    main, paired = markdown_tables(data)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for key, table in (("MAIN_TABLE", main), ("PAIRED_TABLE", paired)):
        pattern = rf"(?s)(<!-- PHYSALIGN:{key}:START -->).*?(<!-- PHYSALIGN:{key}:END -->)"
        readme, count = re.subn(pattern, lambda match: match[1] + "\n" + table + "\n" + match[2], readme)
        if count != 1:
            raise ValueError(f"Expected one README marker pair: {key}")
    head, rows = html_table(data)
    template = (ROOT / "docs/leaderboard/template.html").read_text(encoding="utf-8")
    substitutions = {
        "DATA": json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"),
        "TABLE_HEAD": head, "TABLE_ROWS": rows,
        "ARXIV": escape(data["links"]["arxiv"], quote=True),
        "PROJECT": escape(data["links"]["project"], quote=True),
        "DATASET": escape(data["links"]["dataset"], quote=True),
    }
    for key, value in substitutions.items():
        template = template.replace("@@" + key + "@@", value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("Unresolved leaderboard template placeholder")
    return {
        ROOT / "README.md": readme,
        ROOT / "docs/leaderboard/index.html": template,
        ROOT / "docs/assets/badge-dataset.svg": badge("Hugging Face", "Dataset", "#927014", 104, 74),
        ROOT / "docs/assets/badge-project.svg": badge("Project", "Page", "#4065a8", 68, 60),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated public artifacts have drifted")
    args = parser.parse_args()
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    validate(data)
    stale = []
    for path, content in outputs(data).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if args.check:
            stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        raise SystemExit("Regenerate with python scripts/build_readme.py: " + ", ".join(stale))
    print("README tables, badges, and local leaderboard are synchronized with the paper baseline JSON.")


if __name__ == "__main__":
    main()
