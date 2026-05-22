"""Send a webhook alert when a simulated LLM cost threshold is exceeded."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from langfuse import get_client, observe

load_dotenv()

langfuse = get_client()


@dataclass
class CostAlert:
    """Payload sent to the alert webhook."""

    text: str
    timestamp: str
    alert_type: str
    details: dict[str, str]


def get_required_webhook_url() -> str:
    """Read the webhook URL from the environment instead of hardcoding secrets."""

    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        raise RuntimeError(
            "Set ALERT_WEBHOOK_URL to a webhook.site, Slack, PagerDuty, "
            "or local ngrok webhook endpoint."
        )
    return webhook_url


def build_cost_alert(hourly_cost: float, threshold: float) -> CostAlert:
    """Create a structured alert payload that any webhook receiver can parse."""

    overage = hourly_cost - threshold
    return CostAlert(
        text=f"Cost Alert: ${hourly_cost:.2f} spent in the last hour",
        timestamp=datetime.now(timezone.utc).isoformat(),
        alert_type="cost_spike",
        details={
            "hourly_cost": f"${hourly_cost:.2f}",
            "threshold": f"${threshold:.2f}",
            "overage": f"${overage:.2f}",
            "dashboard": os.getenv("LANGFUSE_DASHBOARD_URL", "https://cloud.langfuse.com"),
        },
    )


@observe(name="send_alert_webhook", as_type="span")
def send_alert_webhook(alert: CostAlert, webhook_url: str) -> requests.Response:
    """Send the alert and record the delivery status in Langfuse."""

    response = requests.post(webhook_url, json=asdict(alert), timeout=10)

    langfuse.update_current_span(
        input={"webhook_url_configured": bool(webhook_url), "alert": asdict(alert)},
        output={"status_code": response.status_code, "ok": response.ok},
        metadata={
            "alert_type": alert.alert_type,
            "delivery_target": webhook_url.split("?")[0],
        },
    )

    response.raise_for_status()
    return response


@observe(name="check_costs_and_alert", as_type="span")
def check_costs_and_alert(
    hourly_cost: float | None = None,
    threshold: float | None = None,
    webhook_url: str | None = None,
) -> bool:
    """Monitor LLM costs and send an alert when the threshold is exceeded."""

    hourly_cost = hourly_cost or float(os.getenv("SIMULATED_HOURLY_COST", "15.50"))
    threshold = threshold or float(os.getenv("ALERT_COST_THRESHOLD", "10.00"))

    print(f"Current hourly cost: ${hourly_cost:.2f}")
    print(f"Threshold: ${threshold:.2f}")

    should_alert = hourly_cost > threshold

    langfuse.update_current_span(
        input={"hourly_cost": hourly_cost, "threshold": threshold},
        metadata={
            "alert_type": "cost_spike",
            "should_alert": should_alert,
            "overage": max(hourly_cost - threshold, 0.0),
        },
    )

    if not should_alert:
        print("Costs are within budget. No alert needed.")
        langfuse.update_current_span(output={"alert_sent": False})
        return False

    print("Cost spike detected. Sending alert.")
    alert = build_cost_alert(hourly_cost, threshold)
    response = send_alert_webhook(alert, webhook_url or get_required_webhook_url())

    print(f"Alert sent. Webhook status: {response.status_code}")
    langfuse.update_current_span(output={"alert_sent": True, "status_code": response.status_code})
    return True


if __name__ == "__main__":
    try:
        check_costs_and_alert()
    finally:
        langfuse.flush()
