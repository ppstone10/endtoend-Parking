"""最小 OOXML(.docx) 生成器：只用标准库，避免给 conda 环境装新依赖。

支持：标题/正文/项目符号/公式行/表格/分页，中文字体与页脚页码。
"""
from __future__ import annotations

import re
import zipfile
from xml.sax.saxutils import escape

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

CONTENT_WIDTH = 9026  # A4 去掉左右各 1440 twips 页边距后的可用宽度

_BODY_RPR = (
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
    'w:eastAsia="宋体" w:cs="Times New Roman"/><w:sz w:val="21"/><w:szCs w:val="21"/>'
)
_HEAD_FONT = '<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="黑体"/>'

_HEAD_RPR = {
    1: _HEAD_FONT + '<w:b/><w:color w:val="1F3864"/><w:sz w:val="30"/><w:szCs w:val="30"/>',
    2: _HEAD_FONT + '<w:b/><w:color w:val="2E5496"/><w:sz w:val="26"/><w:szCs w:val="26"/>',
    3: _HEAD_FONT + '<w:b/><w:color w:val="2E5496"/><w:sz w:val="22"/><w:szCs w:val="22"/>',
    4: _HEAD_FONT + '<w:b/><w:color w:val="404040"/><w:sz w:val="21"/><w:szCs w:val="21"/>',
}

_INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`)")


def _runs(text: str, rpr: str = _BODY_RPR, extra_bold: bool = False) -> str:
    out = []
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            style = rpr + "<w:b/>"
            body = part[2:-2]
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            style = (
                '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="宋体"/>'
                '<w:sz w:val="20"/><w:szCs w:val="20"/>'
            )
            body = part[1:-1]
        else:
            style = rpr + ("<w:b/>" if extra_bold else "")
            body = part
        out.append(
            '<w:r><w:rPr>' + style + '</w:rPr><w:t xml:space="preserve">'
            + escape(body) + "</w:t></w:r>"
        )
    return "".join(out)


class Doc:
    def __init__(self) -> None:
        self.body: list[str] = []

    # ---------- 基础块 ----------
    def title(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr><w:pStyle w:val="Title"/>'
            '<w:spacing w:before="240" w:after="120" w:line="360" w:lineRule="auto"/>'
            '<w:jc w:val="center"/>'
            '</w:pPr>'
            + _runs(text, _HEAD_FONT + '<w:b/><w:color w:val="1F3864"/><w:sz w:val="40"/><w:szCs w:val="40"/>')
            + "</w:p>"
        )

    def subtitle(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr>'
            '<w:spacing w:before="0" w:after="240" w:line="360" w:lineRule="auto"/>'
            '<w:jc w:val="center"/></w:pPr>'
            + _runs(text, _BODY_RPR + '<w:color w:val="595959"/><w:sz w:val="22"/>')
            + "</w:p>"
        )

    def h(self, level: int, text: str) -> None:
        self.body.append(
            f'<w:p><w:pPr><w:pStyle w:val="Heading{level}"/>'
            f'<w:keepNext/><w:spacing w:before="200" w:after="100" w:line="320" w:lineRule="auto"/>'
            f'</w:pPr>' + _runs(text, _HEAD_RPR[level], extra_bold=True) + "</w:p>"
        )

    def p(self, text: str, align: str = "both", indent: int = 0) -> None:
        self.body.append(
            f'<w:p><w:pPr><w:spacing w:before="0" w:after="100" w:line="340" w:lineRule="auto"/>'
            f'<w:jc w:val="{align}"/>'
            + (f'<w:ind w:left="{indent}"/>' if indent else "")
            + "</w:pPr>" + _runs(text) + "</w:p>"
        )

    def bullet(self, text: str, level: int = 0) -> None:
        mark = "• " if level == 0 else "– "
        left = 420 + level * 420
        self.body.append(
            f'<w:p><w:pPr><w:spacing w:before="0" w:after="60" w:line="330" w:lineRule="auto"/>'
            f'<w:ind w:left="{left}" w:hanging="210"/><w:jc w:val="both"/></w:pPr>'
            + _runs(mark + text) + "</w:p>"
        )

    def formula(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr>'
            '<w:spacing w:before="80" w:after="80" w:line="300" w:lineRule="auto"/>'
            '<w:jc w:val="center"/></w:pPr>'
            + _runs(
                text,
                '<w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:eastAsia="宋体"/>'
                '<w:i/><w:sz w:val="21"/><w:szCs w:val="21"/>',
            )
            + "</w:p>"
        )

    def flow(self, lines: list[str]) -> None:
        """居中流程图（箭头连线），用于讲清数据流与状态流。"""
        for i, line in enumerate(lines):
            after = "140" if i == len(lines) - 1 else "20"
            self.body.append(
                '<w:p><w:pPr>'
                f'<w:spacing w:before="20" w:after="{after}" w:line="280" w:lineRule="auto"/>'
                '<w:ind w:left="360"/><w:jc w:val="center"/></w:pPr>'
                + _runs(
                    line,
                    '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="黑体"/>'
                    '<w:sz w:val="20"/><w:szCs w:val="20"/><w:color w:val="1F3864"/>',
                    extra_bold=True,
                )
                + "</w:p>"
            )

    def callout(self, title: str, text: str = "") -> None:
        """带底纹与左侧竖线的提示块，用于“可以这样讲”等话术。"""
        body = f"**{title}**" + (f"　{text}" if text else "")
        self.body.append(
            '<w:p><w:pPr>'
            '<w:pBdr><w:left w:val="single" w:sz="18" w:space="8" w:color="2E5496"/></w:pBdr>'
            '<w:shd w:val="clear" w:color="auto" w:fill="F2F6FB"/>'
            '<w:spacing w:before="100" w:after="160" w:line="330" w:lineRule="auto"/>'
            '<w:ind w:left="220" w:right="140"/>'
            '<w:jc w:val="both"/></w:pPr>'
            + _runs(body)
            + "</w:p>"
        )

    def caption(self, text: str) -> None:
        self.body.append(
            '<w:p><w:pPr>'
            '<w:spacing w:before="40" w:after="140" w:line="280" w:lineRule="auto"/>'
            '<w:jc w:val="center"/></w:pPr>'
            + _runs(text, _BODY_RPR + '<w:color w:val="595959"/><w:sz w:val="19"/>')
            + "</w:p>"
        )

    def page_break(self) -> None:
        self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows: list[list[str]], widths: list[int] | None = None, header: bool = True) -> None:
        ncol = max(len(r) for r in rows)
        if widths is None:
            widths = [CONTENT_WIDTH // ncol] * ncol
        total = sum(widths)
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        borders = (
            "<w:tblBorders>"
            + "".join(
                f'<w:{edge} w:val="single" w:sz="6" w:space="0" w:color="9DA9BF"/>'
                for edge in ("top", "left", "bottom", "right", "insideH", "insideV")
            )
            + "</w:tblBorders>"
        )
        tbl = (
            '<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/>'
            f'{borders}'
            '<w:tblLayout w:type="fixed"/>'
            '<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
            '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
            f"</w:tblPr><w:tblGrid>{grid}</w:tblGrid>"
        )
        for i, row in enumerate(rows):
            is_head = header and i == 0
            cells = []
            for j in range(ncol):
                text = row[j] if j < len(row) else ""
                shade = '<w:shd w:val="clear" w:color="auto" w:fill="DEEAF6"/>' if is_head else ""
                rpr = _BODY_RPR + ('<w:b/><w:sz w:val="20"/><w:szCs w:val="20"/>' if is_head else '<w:sz w:val="20"/><w:szCs w:val="20"/>')
                cells.append(
                    f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{shade}'
                    '<w:vAlign w:val="center"/></w:tcPr>'
                    '<w:p><w:pPr><w:spacing w:before="20" w:after="20" w:line="260" w:lineRule="auto"/>'
                    '<w:jc w:val="left"/></w:pPr>'
                    + _runs(text, rpr) + "</w:p></w:tc>"
                )
            trpr = '<w:trPr><w:tblHeader/></w:trPr>' if is_head else ""
            tbl += f"<w:tr>{trpr}" + "".join(cells) + "</w:tr>"
        tbl += "</w:tbl>"
        self.body.append(tbl)
        # 表格与后文之间留白
        self.body.append('<w:p><w:pPr><w:spacing w:before="0" w:after="60" w:line="240" w:lineRule="auto"/></w:pPr></w:p>')

    # ---------- 打包 ----------
    def save(self, path: str, core_title: str) -> None:
        sect = (
            '<w:sectPr><w:footerReference w:type="default" r:id="rId3"/>'
            '<w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="851" w:footer="992" w:gutter="0"/>'
            '<w:cols w:space="425"/><w:docGrid w:type="lines" w:linePitch="312"/></w:sectPr>'
        )
        document = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:document xmlns:w="{W}" xmlns:r="{R}"><w:body>'
            + "".join(self.body)
            + sect
            + "</w:body></w:document>"
        )
        styles = _styles_xml()
        footer = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:ftr xmlns:w="{W}"><w:p><w:pPr>'
            '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
            '<w:jc w:val="center"/></w:pPr>'
            + _runs("第 ", _BODY_RPR + '<w:color w:val="808080"/><w:sz w:val="18"/>')
            + '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + _runs("1", _BODY_RPR + '<w:color w:val="808080"/><w:sz w:val="18"/>')
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
            + _runs(" 页", _BODY_RPR + '<w:color w:val="808080"/><w:sz w:val="18"/>')
            + "</w:p></w:ftr>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        )
        root_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        )
        doc_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
            "</Relationships>"
        )
        core = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f"<dc:title>{escape(core_title)}</dc:title>"
            "<dc:subject>无人矿卡端到端自动泊车系统</dc:subject>"
            "<cp:keywords>自动泊车;BEV;模仿学习;Hybrid A*;MPC</cp:keywords>"
            "</cp:coreProperties>"
        )
        app = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>DSH OOXML Builder</Application><DocSecurity>0</DocSecurity>"
            "</Properties>"
        )
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", content_types)
            z.writestr("_rels/.rels", root_rels)
            z.writestr("word/document.xml", document)
            z.writestr("word/styles.xml", styles)
            z.writestr("word/footer1.xml", footer)
            z.writestr("word/_rels/document.xml.rels", doc_rels)
            z.writestr("docProps/core.xml", core)
            z.writestr("docProps/app.xml", app)


def _styles_xml() -> str:
    def head(sid: str, name: str, level: int, size: int) -> str:
        return (
            f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/>'
            '<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
            f'<w:pPr><w:keepNext/><w:outlineLvl w:val="{level}"/>'
            '<w:spacing w:before="200" w:after="100" w:line="320" w:lineRule="auto"/></w:pPr>'
            f'<w:rPr>{_HEAD_FONT}<w:b/><w:color w:val="2E5496"/>'
            f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr></w:style>'
        )

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles xmlns:w="{W}">'
        "<w:docDefaults><w:rPrDefault><w:rPr>" + _BODY_RPR + "</w:rPr></w:rPrDefault>"
        '<w:pPrDefault><w:pPr><w:spacing w:line="340" w:lineRule="auto" w:after="100"/></w:pPr></w:pPrDefault>'
        "</w:docDefaults>"
        '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
        '<w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/>'
        f'<w:rPr>{_HEAD_FONT}<w:b/><w:sz w:val="40"/></w:rPr></w:style>'
        + head("Heading1", "heading 1", 0, 30)
        + head("Heading2", "heading 2", 1, 26)
        + head("Heading3", "heading 3", 2, 22)
        + head("Heading4", "heading 4", 3, 21)
        + '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/></w:style>'
        + "</w:styles>"
    )
