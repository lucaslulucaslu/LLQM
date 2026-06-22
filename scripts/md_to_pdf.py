from __future__ import annotations

import argparse
import re
from pathlib import Path

from markdown import markdown
from xhtml2pdf import pisa

BASE_CSS = """
@page {
  size: A4;
  margin: 0.72in;
}
body {
  font-family: Helvetica, Arial, sans-serif;
  font-size: 10.3pt;
  line-height: 1.32;
  color: #1a1a1a;
}
h1, h2, h3 {
  color: #0f172a;
  margin-top: 14px;
  margin-bottom: 6px;
}
h1 { font-size: 18pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h2 { font-size: 14pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; }
h3 { font-size: 11.5pt; }
p { margin: 4px 0 6px 0; }
hr { margin: 10px 0; border: none; border-top: 1px solid #e2e8f0; }
code {
  font-family: Courier, monospace;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 0 3px;
  font-size: 9.2pt;
}
pre {
  font-family: Courier, monospace;
  font-size: 8.7pt;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  padding: 5px 6px;
  white-space: normal;
}
.codeblock {
  line-height: 1.08;
  margin-top: 5px;
  margin-bottom: 6px;
  padding: 5px 6px 1px 6px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin: 6px 0 9px 0;
  table-layout: fixed;
  font-size: 9.5pt;
}
th, td {
  border: 1px solid #d1d5db;
  padding: 3px 5px;
  vertical-align: top;
  line-height: 1.2;
}
th {
  background: #f1f5f9;
  font-weight: bold;
}
.results-summary { table-layout: auto; }
.results-summary th:nth-child(4), .results-summary td:nth-child(4) { text-align: center; }
.results-summary th:nth-child(5), .results-summary td:nth-child(5) { text-align: center; }
ul, ol {
  margin-top: 3px;
  margin-bottom: 6px;
  padding-left: 18px;
}
li {
  margin: 1px 0;
}
blockquote {
  border-left: 4px solid #cbd5e1;
  margin: 5px 0;
  padding-left: 8px;
  color: #334155;
}
"""

_PRE_CODE_RE = re.compile(
    r"<pre><code(?: class=\"[^\"]*\")?>(.*?)</code></pre>", re.DOTALL
)
_RESULTS_SUMMARY_TABLE_RE = re.compile(
    r"<table>\s*<thead>\s*<tr>\s*"
    r"<th>Query</th>\s*<th>Ground Truth</th>\s*<th>System Verdict</th>\s*"
    r"<th>Confidence</th>\s*<th>Status</th>\s*"
    r"</tr>\s*</thead>",
    re.DOTALL,
)
_RESULTS_SUMMARY_BLOCK_RE = re.compile(
    r"(<table class=\"results-summary\">.*?</table>)", re.DOTALL
)


def _preserve_code_block_line_breaks(html_body: str) -> str:
    """Convert code block newlines to explicit <br/> for xhtml2pdf rendering.

    xhtml2pdf can ignore/preprocess white-space rules inconsistently for <pre><code>.
    We keep code escaped but convert line breaks and leading indentation explicitly.
    """

    def _replace(match: re.Match[str]) -> str:
        code_text = match.group(1)
        lines = code_text.splitlines()

        # Markdown often wraps fenced code content with a leading/trailing
        # newline inside <pre><code>; trim only those wrapper blanks.
        if lines and lines[0] == "":
            lines = lines[1:]
        if lines and lines[-1] == "":
            lines = lines[:-1]

        rendered_lines: list[str] = []
        for line in lines:
            leading_spaces = len(line) - len(line.lstrip(" "))
            rendered_lines.append("&nbsp;" * leading_spaces + line[leading_spaces:])
        return f"<pre class=\"codeblock\">{'<br/>'.join(rendered_lines)}</pre>"

    return _PRE_CODE_RE.sub(_replace, html_body)


def _mark_results_summary_table(html_body: str) -> str:
    """Attach class for custom widths on the Results Summary table only."""

    match = _RESULTS_SUMMARY_TABLE_RE.search(html_body)
    if not match:
        return html_body

    start, end = match.span()
    table_open = match.group(0).replace(
        "<table>",
        (
            '<table class="results-summary">'
            "<colgroup>"
            '<col style="width: 40%;"/>'
            '<col style="width: 18%;"/>'
            '<col style="width: 22%;"/>'
            '<col style="width: 13%;"/>'
            '<col style="width: 7%;"/>'
            "</colgroup>"
        ),
        1,
    )
    return html_body[:start] + table_open + html_body[end:]


def _force_results_summary_cell_widths(html_body: str) -> str:
    """Apply explicit width attributes that xhtml2pdf reliably honors."""

    match = _RESULTS_SUMMARY_BLOCK_RE.search(html_body)
    if not match:
        return html_body

    block = match.group(1)
    block = block.replace("<th>Query</th>", '<th width="40%">Query</th>')
    block = block.replace("<th>Ground Truth</th>", '<th width="18%">Ground Truth</th>')
    block = block.replace(
        "<th>System Verdict</th>", '<th width="22%">System Verdict</th>'
    )
    block = block.replace("<th>Confidence</th>", '<th width="13%">Confidence</th>')
    block = block.replace("<th>Status</th>", '<th width="7%">Status</th>')

    row_re = re.compile(
        r"<tr>\s*"
        r"<td>(.*?)</td>\s*"
        r"<td>(.*?)</td>\s*"
        r"<td>(.*?)</td>\s*"
        r"<td>(.*?)</td>\s*"
        r"<td>(.*?)</td>\s*"
        r"</tr>",
        re.DOTALL,
    )

    def _row_replace(row_match: re.Match[str]) -> str:
        c1, c2, c3, c4, c5 = row_match.groups()
        return (
            "<tr>"
            f'<td width="40%">{c1}</td>'
            f'<td width="18%">{c2}</td>'
            f'<td width="22%">{c3}</td>'
            f'<td width="13%">{c4}</td>'
            f'<td width="7%">{c5}</td>'
            "</tr>"
        )

    block = row_re.sub(_row_replace, block)
    return html_body[: match.start()] + block + html_body[match.end() :]


def convert_markdown_to_pdf(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")
    html_body = markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )
    html_body = _preserve_code_block_line_breaks(html_body)
    html_body = _mark_results_summary_table(html_body)
    html_body = _force_results_summary_cell_widths(html_body)

    html = f"""
<html>
  <head>
    <meta charset=\"utf-8\" />
    <style>{BASE_CSS}</style>
  </head>
  <body>
    {html_body}
  </body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(src=html, dest=pdf_file, encoding="utf-8")

    if result.err:
        raise RuntimeError("PDF generation failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Markdown to styled PDF")
    parser.add_argument("input", help="Path to input markdown file")
    parser.add_argument("output", nargs="?", help="Path to output PDF file")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_suffix(".pdf")

    convert_markdown_to_pdf(input_path, output_path)
    print(f"PDF written to: {output_path}")


if __name__ == "__main__":
    main()
