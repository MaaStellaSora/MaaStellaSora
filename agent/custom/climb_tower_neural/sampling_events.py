"""Maa's supported Agent event sinks. Callbacks only record; they issue no RPCs."""
from maa.context import ContextEventSink
from maa.tasker import TaskerEventSink

from . import TASK_ENTRY


class TowerContextSink(ContextEventSink):
    def __init__(self, collector):
        self.collector = collector

    def on_raw_notification(self, context, msg, details):
        collector = self.collector
        name = details.get("name", "")
        if not collector.enabled or not collector.run_confirmed or not name.startswith("星塔_") or name.startswith("星塔_神经网络_"):
            return
        if msg == "Node.Action.Succeeded":
            if name == "星塔_节点_进行对话选择_agent":
                collector.event_clicked(name)
            elif name == "星塔_保存记录_agent":
                collector.settlement_seen = True
        if msg != "Node.PipelineNode.Succeeded":
            return
        kind = {
            "星塔_节点_商店_商店主流程_agent": "shop_action_completed",
            "星塔_节点_商店_离开商店_agent": "shop_leave",
            "星塔_记录界面_读取数值_agent": "settlement_page",
            "星塔_保存记录_agent": "settlement_save",
        }.get(name)
        if kind:
            collector.emit("lifecycle_observed", lifecycle_type=kind)


class TowerTaskSink(TaskerEventSink):
    def __init__(self, collector, notes=None, runtime=None):
        self.collector = collector
        self.task_id = None
        self.notes = notes
        self.runtime = runtime

    def on_raw_notification(self, tasker, msg, details):
        if msg == "Tasker.Task.Starting":
            if self.collector.upload and self.collector.upload.consent_enabled:
                self.collector.upload.set_consent(False)
            if self.runtime and self.runtime.task_open:
                if self.notes:
                    self.notes.finish("next_task_started")
                self.collector.finish_at_boundary("next_task_started")
                self.runtime.finish_task()
                self.collector.enabled = False
            self.task_id = details["task_id"] if details.get("entry") == TASK_ENTRY else None
            return
        if (msg not in ("Tasker.Task.Succeeded", "Tasker.Task.Failed") or self.task_id is None
                or details.get("task_id") != self.task_id):
            return
        if self.collector.enabled:
            if self.notes:
                self.notes.finish("task_end")
            self.collector.finish_at_boundary("task_ended_without_settlement")
            self.collector.emit("task_end", status="succeeded" if msg.endswith("Succeeded") else "failed_or_stopped")
            self.collector.enabled = False
        if self.runtime:
            self.runtime.finish_task()
        self.task_id = None
