import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.checks.check_i18n import check  # noqa: E402


class PipelineI18nTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "assets" / "interface" / "tasks").mkdir(parents=True)
        (self.root / "assets" / "resource" / "base").mkdir(parents=True)
        (self.root / "assets" / "i18n").mkdir(parents=True)
        (self.root / "agent" / "agent_config" / "i18n").mkdir(parents=True)

        self.interface = {
            "languages": {"zh_cn": "i18n/zh_cn.json"},
            "import": ["interface/tasks/tasks.json"],
        }
        self.task = {
            "option": {
                "Example": {
                    "pipeline_override": {
                        "OverrideNode": {
                            "focus": {
                                "Node.Action.Succeeded": {
                                    "content": "$interface.task.option.Example.pipeline_override.OverrideNode.focus.Node.Action.Succeeded",
                                    "display": "log",
                                }
                            }
                        }
                    }
                }
            }
        }
        self.pipeline = {
            "PipelineNode": {
                "focus": {
                    "Node.Action.Succeeded": "$pipeline.PipelineNode.focus.Node.Action.Succeeded"
                }
            }
        }
        self.translations = {
            "interface.task.option.Example.pipeline_override.OverrideNode.focus.Node.Action.Succeeded": "覆盖完成",
            "pipeline.PipelineNode.focus.Node.Action.Succeeded": "节点完成",
        }
        self._write_json("assets/interface.json", self.interface)
        self._write_json("assets/interface/tasks/tasks.json", self.task)
        self._write_json("assets/resource/base/pipeline.json", self.pipeline)
        self._write_json("assets/i18n/zh_cn.json", self.translations)
        self._write_json("agent/agent_config/i18n/zh_cn.json", {})

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_json(self, relative_path, value):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_scans_normal_pipeline_and_override_focus_templates(self):
        self.assertEqual(check(self.root), [])

    def test_reports_hardcoded_chinese_focus_content(self):
        self.task["option"]["Example"]["pipeline_override"]["OverrideNode"]["focus"][
            "Node.Action.Succeeded"
        ]["content"] = "硬编码文案"
        self._write_json("assets/interface/tasks/tasks.json", self.task)

        findings = check(self.root)

        self.assertTrue(
            any("hardcoded Chinese in a Pipeline focus message" in item.message for item in findings)
        )

    def test_ignores_language_specific_pipeline_files(self):
        self._write_json(
            "assets/interface/tasks/tasks_en.json",
            {
                "option": {
                    "Localized": {
                        "pipeline_override": {
                            "Node": {"focus": {"Node.Action.Succeeded": "硬编码文案"}}
                        }
                    }
                }
            },
        )
        self.interface["import"].append("interface/tasks/tasks_en.json")
        self._write_json("assets/interface.json", self.interface)
        self._write_json(
            "assets/resource/en/pipeline_en.json",
            {"LocalizedNode": {"focus": {"Node.Action.Succeeded": "硬编码文案"}}},
        )

        self.assertEqual(check(self.root), [])

    def test_resolves_symlinked_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            alias = Path(temp_dir) / "project"
            try:
                alias.symlink_to(self.root, target_is_directory=True)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")

            self.assertEqual(check(alias), [])

    def _write_agent_fixture(self, source, translations):
        source_path = self.root / "agent" / "custom" / "example.py"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
        self._write_json("agent/agent_config/i18n/zh_cn.json", translations)

    def test_scans_agent_log_keys_and_ignores_debug(self):
        self._write_agent_fixture(
            """
logger.info("$agent.example.ready")
logger.warning("$agent.example.warning")
self._handle_fatal_error(context, "$agent.example.failed")
messages = {"stop": "$agent.example.stopped"}
logger.error(messages[reason])
label = translate("$agent.example.label")
logger.debug("未处理的 Debug 文案")
""",
            {
                "agent.example.ready": "已就绪",
                "agent.example.warning": "注意",
                "agent.example.failed": "失败",
                "agent.example.stopped": "已停止",
                "agent.example.label": "选项",
            },
        )

        self.assertEqual(check(self.root), [])

    def test_reports_hardcoded_agent_log_and_translation_coverage(self):
        self._write_agent_fixture(
            """
logger.info("硬编码日志")
logger.error("$agent.example.missing")
""",
            {
                "agent.example.unused": "未使用",
            },
        )

        messages = [item.message for item in check(self.root)]

        self.assertIn("hardcoded Chinese in an Agent info/warning/error log", messages)
        self.assertIn(
            "missing translation for locale zh_cn: agent.example.missing",
            messages,
        )
        self.assertIn(
            "unused translation for locale zh_cn: agent.example.unused",
            messages,
        )


if __name__ == "__main__":
    unittest.main()
