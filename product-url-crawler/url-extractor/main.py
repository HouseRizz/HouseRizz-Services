import os
import json
import requests
import uuid
from datetime import datetime
from google.cloud import tasks_v2, firestore, secretmanager
import functions_framework

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secret Manager at runtime (no caching)."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "houserizz-481012")
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

@functions_framework.http
def start_extract_job(request):
    if request.method == "OPTIONS":
        return (
            "",
            204,
            {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "3600",
            },
        )
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        payload = request.get_json(silent=True)
        if not payload:
            return (json.dumps({"error": "Invalid JSON"}), 400, headers)

        url = payload.get("url")
        manufacturer = payload.get("manufacturer")
        tenant_id = payload.get("tenantId")
        is_trigger = payload.get("isTrigger", True)
        document_id = payload.get("documentId")
        filter_type = payload.get("filterType")
        db = firestore.Client()
        current_time = datetime.utcnow().isoformat() + "Z"

        if not is_trigger:
            if not url or not manufacturer or not tenant_id:
                return (
                    json.dumps(
                        {"error": "Missing 'url', 'manufacturer', or 'tenantId' in payload"}
                    ),
                    400,
                    headers,
                )

            try:
                job_id = str(uuid.uuid4())

                crawl_url = url if url.endswith("/*") else url.rstrip("/") + "/*"

                job_doc = {
                    "tenantId": tenant_id,
                    "type": "crawl",
                    "status": "pending",
                    "dateCreated": current_time,
                    "dateEdited": current_time,
                    "dateEnded": None,
                    "jobId": job_id,
                    "manufacturer": manufacturer,
                    "url": url,
                    "crawlUrl": crawl_url
                }

                if filter_type:
                    job_doc["filterType"] = filter_type

                db.collection("webServiceLogs").document(job_id).set(job_doc)

                return (
                    json.dumps(
                        {
                            "success": True,
                            "message": "Job document created without triggering",
                            "documentId": job_id,
                            "tenantId": tenant_id,
                            "filterType": filter_type,
                        }
                    ),
                    200,
                    headers,
                )

            except Exception as e:
                return (
                    json.dumps(
                        {"error": "Failed to save job metadata to Firestore", "details": str(e)}
                    ),
                    500,
                    headers,
                )

        elif is_trigger and document_id:
            try:
                doc_ref = db.collection("webServiceLogs").document(document_id)
                doc = doc_ref.get()

                if not doc.exists:
                    return (
                        json.dumps(
                            {"error": f"Document with ID '{document_id}' not found in Firestore"}
                        ),
                        404,
                        headers,
                    )

                job_data = doc.to_dict()
                url = job_data.get("url")
                manufacturer = job_data.get("manufacturer")
                tenant_id = job_data.get("tenantId")
                crawl_url = job_data.get("crawlUrl")
                filter_type = job_data.get("filterType")
                if not all([url, manufacturer, tenant_id]):
                    return (
                        json.dumps(
                            {"error": "Document missing required fields: url, manufacturer, or tenantId"}
                        ),
                        400,
                        headers,
                    )

            except Exception as e:
                return (
                    json.dumps(
                        {"error": "Failed to fetch document from Firestore", "details": str(e)}
                    ),
                    500,
                    headers,
                )

        else:
            if not url or not manufacturer or not tenant_id:
                return (
                    json.dumps(
                        {"error": "Missing 'url', 'manufacturer', or 'tenantId' in payload"}
                    ),
                    400,
                    headers,
                )

            crawl_url = url if url.endswith("/*") else url.rstrip("/") + "/*"
            document_id = None
        
        # Fetch API key dynamically from Secret Manager
        try:
            api_key = get_secret("HYPERBROWSER_API_KEY")
        except Exception as e:
            return (
                json.dumps({"error": f"Failed to get API key: {str(e)}"}),
                500,
                headers,
            )

        extract_body = {
            "urls": [crawl_url],
            "prompt": (
                "I want urls of each individual products in the url, "
                "the url is like a collection of products"
            ),
            "schema": {
                "type": "object",
                "properties": {
                    "productUrls": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["productUrls"],
            },
            "maxLinks": 50,
        }

        resp = requests.post(
            "https://app.hyperbrowser.ai/api/extract",
            headers={"Content-Type": "application/json", "x-api-key": api_key},
            json=extract_body,
            timeout=30,
        )
        if resp.status_code != 200:
            return (
                json.dumps(
                    {
                        "error": "Hyperbrowser API error",
                        "status": resp.status_code,
                        "details": resp.text,
                    }
                ),
                resp.status_code,
                headers,
            )

        job_id = resp.json().get("jobId")

        try:
            if document_id:
                doc_ref = db.collection("webServiceLogs").document(document_id)
                doc_ref.update({
                    "status": "started",
                    "dateEdited": current_time,
                    "jobId": job_id
                })
                final_document_id = document_id
            else:
                job_doc = {
                    "tenantId": tenant_id,
                    "type": "crawl",
                    "status": "started",
                    "dateCreated": current_time,
                    "dateEdited": current_time,
                    "dateEnded": None,
                    "jobId": job_id,
                    "manufacturer": manufacturer,
                    "url": url,
                    "crawlUrl": crawl_url
                }

                if filter_type:
                    job_doc["filterType"] = filter_type

                db.collection("webServiceLogs").document(job_id).set(job_doc)
                final_document_id = job_id

        except Exception as e:
            return (
                json.dumps(
                    {"error": "Failed to save/update job metadata in Firestore", "details": str(e)}
                ),
                500,
                headers,
            )

        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("TASK_LOCATION", "us-central1")
        queue = os.getenv("TASK_QUEUE", "products-scraping")
        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(project, location, queue)

        task_payload = {
            "jobId": job_id,
            "manufacturer": manufacturer,
            "tenantId": tenant_id
        }

        if filter_type:
            task_payload["filterType"] = filter_type
        task = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": (
                    "https://hyperbrowser-jobv2-wmggns3mvq-uc.a.run.app"
                ),
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(task_payload).encode(),
            }
        }
        task_resp = client.create_task(request={"parent": parent, "task": task})

        return (
            json.dumps(
                {
                    "success": True,
                    "jobId": job_id,
                    "documentId": final_document_id,
                    "tenantId": tenant_id,
                    "filterType": filter_type,
                    "pollerTask": task_resp.name,
                }
            ),
            200,
            headers,
        )

    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, headers)
