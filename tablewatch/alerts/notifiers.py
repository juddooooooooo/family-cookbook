"""Where alerts go: console, webhook, Slack, WhatsApp (stub) or your own class.

Notifiers run on a background thread so a slow webhook never stalls video
processing. Add a channel by subclassing `Notifier` and naming it in
config.yaml as `type: "my_package.module:MyNotifier"`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import queue
import threading
import urllib.request
from abc import ABC, abstractmethod

from tablewatch.alerts.rules import Alert
from tablewatch.config import NotifierConfig
from tablewatch.core import fmt_duration

log = logging.getLogger(__name__)

RAISED, REPEATED, RESOLVED = "raised", "repeated", "resolved"


def describe(alert: Alert, event: str) -> str:
    if event == RESOLVED:
        return f"RESOLVED {alert.table_id} {alert.rule} (after {fmt_duration(alert.value_s)})"
    prefix = "ALERT" if event == RAISED else f"ALERT x{alert.notify_count}"
    return f"{prefix} [{alert.severity}] {alert.message}"


class Notifier(ABC):
    def __init__(self, cfg: NotifierConfig):
        self.cfg = cfg

    @abstractmethod
    def send(self, alert: Alert, event: str) -> None:
        """Deliver one alert event ('raised', 'repeated' or 'resolved')."""


class ConsoleNotifier(Notifier):
    def send(self, alert: Alert, event: str) -> None:
        print(f"[{fmt_duration(alert.resolved_ts or alert.last_notified_ts or alert.ts)}] {describe(alert, event)}", flush=True)


class WebhookNotifier(Notifier):
    """POSTs the alert as JSON to `url`. Works with Zapier, n8n, Make, custom APIs."""

    def payload(self, alert: Alert, event: str) -> dict:
        return {"event": event, "text": describe(alert, event), "alert": dataclasses.asdict(alert)}

    def send(self, alert: Alert, event: str) -> None:
        if not self.cfg.url:
            raise ValueError(f"{type(self).__name__} needs a url")
        body = json.dumps(self.payload(alert, event)).encode()
        req = urllib.request.Request(self.cfg.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            resp.read()


class SlackNotifier(WebhookNotifier):
    """Slack incoming webhook (https://api.slack.com/messaging/webhooks)."""

    def payload(self, alert: Alert, event: str) -> dict:
        icon = ":white_check_mark:" if event == RESOLVED else ":rotating_light:"
        return {"text": f"{icon} {describe(alert, event)}"}


class WhatsAppNotifier(Notifier):
    """Stub. To wire up, implement `send` with e.g. the Twilio API
    (POST https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json with
    From=whatsapp:+..., To=whatsapp:{cfg.to}, Body=describe(...)) or the Meta
    WhatsApp Cloud API. Put credentials in `options` or environment variables.
    """

    def send(self, alert: Alert, event: str) -> None:
        log.info("WhatsApp (stub) to %s: %s", self.cfg.to or "<unset>", describe(alert, event))


BUILTIN = {
    "console": ConsoleNotifier,
    "webhook": WebhookNotifier,
    "slack": SlackNotifier,
    "whatsapp": WhatsAppNotifier,
}


def build_notifier(cfg: NotifierConfig) -> Notifier:
    cls = BUILTIN.get(cfg.type)
    if cls is None:
        module_name, _, class_name = cfg.type.partition(":")
        if not class_name:
            raise ValueError(f"Unknown notifier type {cfg.type!r}; use one of {sorted(BUILTIN)} or 'module:Class'")
        cls = getattr(importlib.import_module(module_name), class_name)
    return cls(cfg)


class NotificationDispatcher:
    """Fans alert events out to notifiers on a background thread."""

    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers
        self._queue: queue.Queue[tuple[Alert, str] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="tablewatch-notify", daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, configs: list[NotifierConfig]) -> NotificationDispatcher:
        return cls([build_notifier(c) for c in configs if c.enabled])

    def dispatch(self, alert: Alert, event: str) -> None:
        # Snapshot the alert so later engine updates don't change what gets sent.
        self._queue.put((dataclasses.replace(alert), event))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            alert, event = item
            for notifier in self.notifiers:
                try:
                    notifier.send(alert, event)
                except Exception as exc:  # a broken channel must not stop the others
                    log.warning("%s failed: %s", type(notifier).__name__, exc)

    def close(self, timeout: float = 10.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
