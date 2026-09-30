"""临时脚本：把 docx 抽取为带结构的纯文本（供阅读参考文档风格），避免控制台编码问题。"""
import re
import sys
import zipfile

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
import xml.etree.ElementTree as ET  # noqa: E402

src = sys.argv[1]
dst = sys.argv[2]

z = zipfile.ZipFile(src)
root = ET.fromstring(z.read("word/document.xml"))
body = root.find(W + "body")

lines = []


def para_text(p):
    return "".join(t.text or "" for t in p.iter(W + "t"))


def walk(el, depth=0):
    for child in el:
        tag = child.tag.replace(W, "")
        if tag == "p":
            text = para_text(child).strip()
            if not text:
                continue
            style = ""
            ppr = child.find(W + "pPr")
            if ppr is not None:
                st = ppr.find(W + "pStyle")
                if st is not None:
                    style = st.get(W + "val") or ""
                if ppr.find(W + "numPr") is not None:
                    style = (style + " LIST").strip()
            prefix = f"[{style}] " if style else ""
            lines.append("  " * depth + prefix + text)
        elif tag == "tbl":
            lines.append("  " * depth + "<TABLE>")
            for tr in child.findall(W + "tr"):
                cells = []
                for tc in tr.findall(W + "tc"):
                    cells.append(" ".join(para_text(p).strip() for p in tc.findall(W + "p")).strip())
                lines.append("  " * depth + " | " + " | ".join(cells) + " |")
            lines.append("  " * depth + "</TABLE>")
        elif tag == "sdt":
            walk(child, depth)
        else:
            walk(child, depth)


walk(body)
with open(dst, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("lines:", len(lines))
