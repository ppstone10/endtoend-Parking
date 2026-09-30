"""临时脚本：校验生成的 docx 是否为合法 OOXML 包，并输出结构统计。"""
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

path = sys.argv[1]
z = zipfile.ZipFile(path)
bad = z.testzip()
print("zip integrity:", "OK" if bad is None else f"BAD {bad}")

required = [
    "[Content_Types].xml",
    "_rels/.rels",
    "word/document.xml",
    "word/styles.xml",
    "word/footer1.xml",
    "word/_rels/document.xml.rels",
    "docProps/core.xml",
    "docProps/app.xml",
]
missing = [n for n in required if n not in z.namelist()]
print("missing parts:", missing or "none")

for name in z.namelist():
    if name.endswith(".xml") or name.endswith(".rels"):
        try:
            ET.fromstring(z.read(name))
        except Exception as exc:  # noqa: BLE001
            print("XML ERROR", name, exc)
print("xml parse: done")

root = ET.fromstring(z.read("word/document.xml"))
body = root.find(W + "body")
paras = body.findall(W + "p")
tables = body.findall(W + "tbl")
print("body paragraphs:", len(paras), "tables:", len(tables))

# 关系解析检查：每个 relationship 的 Target 必须真实存在
R = "{http://schemas.openxmlformats.org/package/2006/relationships}"
names = set(z.namelist())
for rels_name in [n for n in names if n.endswith(".rels")]:
    base = rels_name.rsplit("_rels/", 1)[0]
    rels_root = ET.fromstring(z.read(rels_name))
    for rel in rels_root.findall(R + "Relationship"):
        if rel.get("TargetMode") == "External":
            continue
        target = base + rel.get("Target")
        if target not in names:
            print("DANGLING RELATIONSHIP", rels_name, rel.get("Target"))
print("relationship targets: checked")

# 表格结构一致性：gridCol 数应与每行 tc 数一致
for i, t in enumerate(tables):
    ncol = len(t.find(W + "tblGrid").findall(W + "gridCol"))
    for j, tr in enumerate(t.findall(W + "tr")):
        ncell = len(tr.findall(W + "tc"))
        if ncell != ncol:
            print(f"TABLE {i} row {j}: cells {ncell} != grid {ncol}")
print("table grid consistency: checked")

# pPr / tblPr 子元素顺序（对照 OOXML 序列）抽查
PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "numPr", "spacing",
             "ind", "jc", "outlineLvl"]
TPR_ORDER = ["tblStyle", "tblW", "jc", "tblBorders", "shd", "tblLayout", "tblCellMar"]
for tag, order, root_el in (("pPr", PPR_ORDER, body), ("tblPr", TPR_ORDER, body)):
    for holder in root_el.iter(W + tag):
        seen = [c.tag.replace(W, "") for c in holder if c.tag.replace(W, "") in order]
        idx = [order.index(s) for s in seen]
        if idx != sorted(idx):
            print(f"ORDER VIOLATION in {tag}: {seen}")
print("element order: checked")

heads = []
for p in paras:
    ppr = p.find(W + "pPr")
    style = None
    if ppr is not None:
        st = ppr.find(W + "pStyle")
        if st is not None:
            style = st.get(W + "val")
        if ppr.find(W + "outlineLvl") is not None:
            pass
    if style and style.startswith("Heading"):
        text = "".join(t.text or "" for t in p.iter(W + "t"))
        heads.append((style, text))
print("headings:", len(heads))
for style, text in heads:
    level = int(style[-1])
    print("  " * (level - 1) + f"[H{level}] {text}")

rows = sum(len(t.findall(W + "tr")) for t in tables)
print("total table rows:", rows)
chars = sum(len(t.text or "") for t in body.iter(W + "t"))
print("total characters:", chars)
