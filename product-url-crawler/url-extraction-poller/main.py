import os
import time
import math
import io
import json
import csv
import requests
from datetime import datetime
from google.cloud import storage, firestore, secretmanager
import functions_framework

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secret Manager at runtime (no caching)."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "houserizz-481012")
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

db = firestore.Client()

@functions_framework.http
def poll_and_generate_csv(request):
    if request.method == "OPTIONS":
        return "", 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

    headers = {"Access-Control-Allow-Origin": "*"}

    # Fetch API key dynamically from Secret Manager
    try:
        api_key = get_secret("HYPERBROWSER_API_KEY")
    except Exception as e:
        return (json.dumps({"error": f"Failed to get API key: {str(e)}"}),
                500, headers)

    payload = request.get_json(silent=True)
    if not payload:
        return (json.dumps({"error": "Invalid JSON"}), 400, headers)

    job_id = payload.get("jobId")
    manufacturer = payload.get("manufacturer")
    tenant_id = payload.get("tenantId")
    filter_type = payload.get("filterType")

    if not job_id or not manufacturer or not tenant_id:
        return (
            json.dumps({
                "error": "Missing 'jobId', 'manufacturer', or 'tenantId'"
            }),
            400,
            headers,
        )

    max_polls = payload.get("maxPolls", 30)
    interval = payload.get("pollInterval", 10)

    query = db.collection("webServiceLogs") \
              .where("jobId", "==", job_id)
    docs = query.get()
    if not docs:
        return (
            json.dumps({"error": f"No document found with jobId: {job_id}"}),
            404,
            headers,
        )
    job_doc_ref = docs[0].reference

    def update_job_status(status, extra=None):
        now = datetime.utcnow().isoformat() + "Z"
        upd = {"status": status, "dateEdited": now}
        if filter_type:
            upd["filterType"] = filter_type
        if status in ("completed", "failed"):
            upd["dateEnded"] = now
        if extra:
            upd.update(extra)
        try:
            job_doc_ref.update(upd)
        except Exception as e:
            print(f"Firestore update failed: {e}")

    update_job_status("in_progress")

    for _ in range(max_polls):
        try:
            resp = requests.get(
                f"https://app.hyperbrowser.ai/api/extract/{job_id}",
                headers={"x-api-key": api_key},
                timeout=30,
            )
            resp.raise_for_status()
        except Exception as e:
            update_job_status("failed", {"error": str(e)})
            return (
                json.dumps({"error": "Status request failed", "details": str(e)}),
                500,
                headers,
            )

        data = resp.json()
        status = data.get("status")

        if status == "completed":
            urls = data.get("data", {}).get("productUrls", [])
            if not urls:
                update_job_status("failed", {"error": "No productUrls"})
                return (json.dumps({"error": "No productUrls"}), 400, headers)

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["product_name", "url"])
            for u in urls:
                writer.writerow([manufacturer, u])
            csv_content = buf.getvalue()
            buf.close()

            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket("houserizz-scraper-items")
                blob = bucket.blob(f"{job_id}.csv")
                blob.upload_from_string(csv_content,
                                        content_type="text/csv")
                gcs_uri = f"gs://houserizz-scraper-items/{job_id}.csv"
            except Exception as e:
                update_job_status("failed", {"error": str(e)})
                return (
                    json.dumps({"error": "GCS upload failed", "details": str(e)}),
                    500,
                    headers,
                )

            batch_size = max(1, math.ceil(len(urls) / 3))
            batch_payload = {
                "csv_url": gcs_uri,
                "batch_size": batch_size,
                "tenantId": tenant_id,
                "jobId": job_id,
                "manufacturer": manufacturer,
            }
            if filter_type:
                batch_payload["filterType"] = filter_type

            try:
                now = datetime.utcnow().isoformat() + "Z"
                doc_ref = db.collection("scrapperJobsToBeDone").add({
                    "payload": batch_payload,
                    "status": "pending",
                    "dateCreated": now,
                })
                scr_id = doc_ref[1].id
                job_doc_ref.update({"scrapperJobsToBeDoneId": scr_id})
            except Exception as e:
                update_job_status("failed", {"error": str(e)})
                return (
                    json.dumps({
                        "error": "Saving job-to-be-done failed",
                        "details": str(e)
                    }),
                    500,
                    headers,
                )

            # Auto-trigger the scrapper job to complete the pipeline
            try:
                trigger_resp = requests.post(
                    "https://trigger-scrapper-job-wmggns3mvq-uc.a.run.app",
                    json={"webServiceLogId": job_id},
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                trigger_result = trigger_resp.json() if trigger_resp.ok else {"error": trigger_resp.text}
            except Exception as e:
                trigger_result = {"error": str(e)}

            update_job_status("completed", {
                "productCount": len(urls),
                "csvUri": gcs_uri,
                "batchSize": batch_size,
                "scrapperJobsToBeDoneId": scr_id,
                "triggerResult": trigger_result,
            })

            return (
                json.dumps({
                    "success": True,
                    "scrapperJobsToBeDoneId": scr_id,
                    "triggerResult": trigger_result,
                }),
                200,
                headers,
            )

        if status in ("pending", "running"):
            update_job_status("in_progress", {
                "lastPolled": datetime.utcnow().isoformat() + "Z"
            })
            time.sleep(interval)
            continue

        update_job_status("failed", {"error": status, "details": data})
        return (json.dumps({"error": status, "details": data}),
                400, headers)

    update_job_status("failed", {"error": "Polling timeout"})
    return (json.dumps({"error": "Polling timeout", "jobId": job_id}),
            408, headers)
