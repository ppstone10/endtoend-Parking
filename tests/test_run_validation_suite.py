"""闭环验证套件入口的协议解析测试。

回归背景：`--output` 指向的目录若与协议目录是**兄弟关系**，`_load_protocol` 会向上
找到父目录里那份**旧的** `protocol.json`；此时 `--indices-file` 给的是新协议的索引，
两者套在一起会让新协议索引落到旧协议的数据集上，样本因"无法复原"被静默丢弃——
产出一份条数变少、场景缺项、却看起来完全合理的报告（实测：300 条 / 7 场景
变成 183 条 / 4 场景，S3/S5/S8 整段消失）。因此索引与数据必须同源。
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_validation_suite import _load_indices, _load_protocol


class TestLoadProtocolPrecedence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.stale = self.root / "protocol.json"
        self.fresh = self.root / "v34" / "protocol.json"
        self._write(self.stale, {"data": "old-dataset.npz", "indices": [1, 2, 3]})
        self._write(
            self.fresh, {"data": "new-dataset.npz", "indices": list(range(300))}
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _write(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_output_dir_is_sibling_of_protocol_dir(self):
        output = self.root / "pilot"
        output.mkdir()

        # 不给 --indices-file：沿用"向上一级复用"的冒烟语义（这里就是旧协议）
        self.assertEqual(_load_protocol(output)["data"], "old-dataset.npz")

        # 给了 --indices-file：必须以该协议为准，索引与数据集同源
        loaded = _load_protocol(output, str(self.fresh))
        self.assertEqual(loaded["data"], "new-dataset.npz")
        self.assertEqual(len(loaded["indices"]), 300)

    def test_protocol_inside_output_wins(self) -> None:
        output = self.root / "v34"
        self.assertEqual(
            _load_protocol(output, str(self.fresh))["data"], "new-dataset.npz"
        )

    def test_missing_protocol_raises(self) -> None:
        # 只向上找两级：比协议所在目录再深两层时应当明确报错，而不是随便捡一份
        with self.assertRaisesRegex(ValueError, "找不到 protocol.json"):
            _load_protocol(self.root / "a" / "b" / "c")


class TestLoadIndices(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, payload) -> str:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_accepts_three_key_names_and_bare_list(self) -> None:
        for key in ("indices", "selected_indices", "evaluated_indices"):
            path = self._write(f"{key}.json", {key: [3, 1, 2]})
            self.assertEqual(_load_indices(path), [3, 1, 2])
        self.assertEqual(_load_indices(self._write("bare.json", [0, 5])), [0, 5])

    def test_rejects_malformed_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少"):
            _load_indices(self._write("nokey.json", {"data": "x"}))
        with self.assertRaisesRegex(ValueError, "整数索引"):
            _load_indices(self._write("float.json", [1.5]))
        with self.assertRaisesRegex(ValueError, "为空"):
            _load_indices(self._write("empty.json", []))


if __name__ == "__main__":
    unittest.main()
