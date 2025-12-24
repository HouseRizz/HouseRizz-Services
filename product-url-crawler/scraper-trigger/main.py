import json
import requests
from datetime import datetime
from google.cloud import firestore
import functions_framework

db = firestore.Client()

@functions_framework.http
def trigger_scrapper_job(request):
    if request.method == "OPTIONS":
        return "", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

    headers = {"Access-Control-Allow-Origin": "*"}

    body = request.get_json(silent=True) or {}
    web_log_id = body.get("webServiceLogId")
    if not web_log_id:
        return (
            json.dumps({"error": "Missing 'webServiceLogId' in request body"}),
            400,
            headers,
        )

    web_ref = db.collection("webServiceLogs").document(web_log_id)
    web_doc = web_ref.get()
    if not web_doc.exists:
        return (
            json.dumps({"error": f"No webServiceLogs doc with ID '{web_log_id}'"}),
            404,
            headers,
        )
    web_data = web_doc.to_dict()

    scr_id = web_data.get("scrapperJobsToBeDoneId")
    if not scr_id:
        return (
            json.dumps({
                "error": "Field 'scrapperJobsToBeDoneId' not found on webServiceLogs doc"
            }),
            400,
            headers,
        )

    scr_ref = db.collection("scrapperJobsToBeDone").document(scr_id)
    scr_doc = scr_ref.get()
    if not scr_doc.exists:
        return (
            json.dumps({
                "error": f"No scrapperJobsToBeDone doc with ID '{scr_id}'"
            }),
            404,
            headers,
        )
    scr_data = scr_doc.to_dict()
    payload = scr_data.get("payload")
    if not payload:
        scr_ref.update({"status": "failed", "error": "Missing payload"})
        return (json.dumps({"error": "No payload in scrapperJobsToBeDone"}), 400, headers)

    try:
        # Call the furniture scraper service
        resp = requests.post(
            "https://furniture-scraper-wmggns3mvq-uc.a.run.app/"
            "scrape_furniture_csv",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        err = str(e)
        scr_ref.update({"status": "failed", "error": err})
        return (json.dumps({"error": err}), 500, headers)

    external = resp.json()
    now = datetime.utcnow().isoformat() + "Z"

    scr_ref.update(
        {
            "status": "triggered",
            "externalResponse": external,
            "dateTriggered": now,
        }
    )

    web_ref.update({"status": "Enriched"})

    return (
        json.dumps({"success": True, "response": external}),
        200,
        headers,
    )