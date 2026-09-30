"""临时脚本：抽取已有 docx 的标题大纲，仅用于对齐新文档覆盖范围。"""
import re
import sys
import zipfile

path = sys.argv[1]
z = zipfile.ZipFile(path)
xml = z.read("word/document.xml").decode("utf-8")
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
out = []
for p in paras:
    text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
    style = re.search(r'w:pStyle w:val="([^"]+)"', p)
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
            .replace("&gt;", ">").replace("&quot;", '"'))
    if text.strip():
        out.append((style.group(1) if style else "", text.strip()))
print("paragraphs:", len(out))
for s, t in out:
    prefix = "[" + s + "] " if s else ""
    print(prefix + t[:120])
