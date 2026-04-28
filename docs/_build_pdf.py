"""Build a polished PDF from docs/design.md with mermaid diagrams pre-rendered."""
import base64
import re
import subprocess
import sys
import urllib.request
import zlib
from pathlib import Path
import markdown

DOCS = Path(__file__).parent
SRC = DOCS / "design.md"
HTML = DOCS / "design.html"
PDF = DOCS / "design.pdf"
IMG_DIR = DOCS / "diagrams"
IMG_DIR.mkdir(exist_ok=True)

text = SRC.read_text(encoding="utf-8")


def render_mermaid_to_png(source: str, out_path: Path) -> bool:
    """Render via kroki.io (returns PNG bytes for mermaid source)."""
    try:
        compressed = zlib.compress(source.encode("utf-8"), 9)
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        url = f"https://kroki.io/mermaid/png/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  kroki failed: {e}", file=sys.stderr)
    # fallback: mermaid.ink (plain base64)
    try:
        encoded = base64.urlsafe_b64encode(source.encode("utf-8")).decode("ascii")
        url = f"https://mermaid.ink/img/{encoded}?type=png"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        out_path.write_bytes(data)
        return True
    except Exception as e:
        print(f"  mermaid.ink failed: {e}", file=sys.stderr)
        return False


# Extract mermaid blocks from raw markdown and replace with image references.
mermaid_pat = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)
counter = {"i": 0}

def replace_mermaid(m):
    counter["i"] += 1
    idx = counter["i"]
    src = m.group(1)
    img_path = IMG_DIR / f"diagram_{idx}.png"
    print(f"rendering diagram {idx} ({len(src)} chars) -> {img_path.name}")
    if not render_mermaid_to_png(src, img_path):
        # fall back to leaving the code block in place
        return m.group(0)
    rel = img_path.relative_to(DOCS).as_posix()
    return f'\n<p class="diagram"><img src="{rel}" alt="diagram {idx}"></p>\n'

text_with_imgs = mermaid_pat.sub(replace_mermaid, text)

md_html = markdown.markdown(
    text_with_imgs,
    extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists"],
    extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
)

html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Git Explainer Agent — Design Document</title>
<style>
  @page {{ size: A4; margin: 18mm 16mm; }}
  html, body {{ background: white; }}
  body {{
    font-family: "Segoe UI", -apple-system, Helvetica, Arial, sans-serif;
    font-size: 11pt; line-height: 1.55; color: #1f2328;
    max-width: 760px; margin: 0 auto;
  }}
  h1 {{ font-size: 22pt; border-bottom: 2px solid #d0d7de; padding-bottom: 6px; }}
  h2 {{ font-size: 16pt; margin-top: 1.6em; border-bottom: 1px solid #d0d7de;
        padding-bottom: 4px; page-break-after: avoid; }}
  h3 {{ font-size: 13pt; margin-top: 1.3em; page-break-after: avoid; }}
  p, li {{ orphans: 3; widows: 3; }}
  code {{ font-family: "Cascadia Mono", "Consolas", monospace; font-size: 9.5pt;
          background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }}
  pre {{ background: #f6f8fa; padding: 10px 12px; border-radius: 6px;
         overflow-x: auto; font-size: 9pt; line-height: 1.4;
         page-break-inside: avoid; white-space: pre-wrap; word-wrap: break-word;
         border: 1px solid #d0d7de; }}
  pre code {{ background: transparent; padding: 0; }}
  table {{ border-collapse: collapse; margin: 1em 0; width: 100%; font-size: 10pt;
           page-break-inside: avoid; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 10px; text-align: left;
            vertical-align: top; }}
  th {{ background: #f6f8fa; }}
  blockquote {{ border-left: 3px solid #d0d7de; padding-left: 12px; color: #57606a; }}
  a {{ color: #0969da; text-decoration: none; }}
  p.diagram {{ text-align: center; margin: 1.4em 0; page-break-inside: avoid; }}
  p.diagram img {{ max-width: 100%; max-height: 230mm; height: auto;
                   border: 1px solid #eaecef; border-radius: 4px; padding: 4px;
                   background: white; }}
  hr {{ border: none; border-top: 1px solid #d0d7de; margin: 2em 0; }}
</style>
</head>
<body>
{md_html}
</body></html>
"""
HTML.write_text(html_doc, encoding="utf-8")
print(f"wrote {HTML}")

edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
url = HTML.resolve().as_uri()
cmd = [
    edge,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=8000",
    f"--print-to-pdf={PDF}",
    url,
]
print("printing to PDF...")
res = subprocess.run(cmd, capture_output=True, text=True)
sys.stdout.write(res.stdout)
sys.stderr.write(res.stderr)
print(f"PDF size: {PDF.stat().st_size if PDF.exists() else 'MISSING'}")
