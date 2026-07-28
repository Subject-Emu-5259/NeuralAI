#!/usr/bin/env python3
"""
NeuralAI reminder daemon.

Polls the `reminders` table in the NeuralAI SQLite DB and fires due reminders
through Zo's messaging REST API (SMS / email / Telegram). Channel is chosen per
row; defaults to SMS. Runs forever, checking every 15s.

Wired to the same DB as webui_service.py:
  /home/workspace/Projects/NeuralAI/data/neuralai.db

Delivery auth: ZO_CLIENT_IDENTITY_TOKEN (set in the service env by Zo).
If the token is missing, reminders are still marked delivered (so they don't
loop forever) but a warning is logged.
"""
import os
import time
import sqlite3
import logging
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [reminder-daemon] %(levelname)s %(message)s")
log = logging.getLogger("reminder-daemon")

DB_PATH = "/home/workspace/Projects/NeuralAI/data/neuralai.db"
POLL_SECONDS = 15
TOKEN = os.environ.get("ZO_CLIENT_IDENTITY_TOKEN", "")
API = "https://api.zo.computer"

CHANNEL_ENDPOINTS = {
    "sms": ("/sms/send", lambda m: {"message": m}),
    "email": ("/email/send", lambda m: {"subject": "NeuralAI Reminder", "markdown_body": m}),
    "telegram": ("/telegram/send", lambda m: {"message": m}),
}


def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def deliver(channel: str, message: str) -> bool:
    if not TOKEN:
        log.warning("ZO_CLIENT_IDENTITY_TOKEN not set; cannot deliver reminder")
        return False
    endpoint, mk = CHANNEL_ENDPOINTS.get(channel, CHANNEL_ENDPOINTS["sms"])
    try:
        r = requests.post(
            API + endpoint,
            headers={"authorization": TOKEN, "content-type": "application/json"},
            json=mk(message),
            timeout=30,
        )
        if r.status_code < 300:
            log.info("Delivered %s reminder", channel)
            return True
        log.error("Delivery failed %s: %s %s", channel, r.status_code, r.text[:200])
        return False
    except Exception as e:
        log.error("Delivery error %s: %s", channel, e)
        return False


def main():
    log.info("Reminder daemon starting (DB=%s, token=%s)", DB_PATH, "present" if TOKEN else "MISSING")
    while True:
        try:
            c = db()
            due = c.execute(
                "SELECT * FROM reminders WHERE done = 0 AND fire_at <= ? ORDER BY fire_at LIMIT 20",
                (time.time(),),
            ).fetchall()
            for row in due:
                msg = f"⏰ Reminder: {row['message']}"
                ok = deliver(row["channel"] or "sms", msg)
                c.execute(
                    "UPDATE reminders SET done = 1, delivered_at = ? WHERE id = ?",
                    (time.time(), row["id"]),
                )
                c.commit()
                if not ok:
                    log.warning("Reminder %s not delivered (token issue?) but marked done", row["id"])
            c.close()
        except Exception as e:
            log.error("Poll error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
