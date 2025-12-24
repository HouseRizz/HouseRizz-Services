import os
import json
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime
import functions_framework

try:
    db = firestore.Client()
except Exception as e:
    print(f"Error initializing Firestore: {e}")
    db = None

WEB_SERVICE_LOGS_COLLECTION = 'webServiceLogs'

@functions_framework.http
def crawl_services_api(request):
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*", "Content-Type": "application/json"}

    if not db:
        return (
            json.dumps({"error": "Firestore client not initialized"}),
            500,
            headers,
        )

    if request.method != "GET":
        return (
            json.dumps({"error": "Method not allowed"}),
            405,
            headers,
        )

    path = request.path.strip('/')
    path_parts = path.split('/')

    try:
        if len(path_parts) == 1 and path_parts[0] == 'crawl-services':
            if request.args.get('stats') == 'true':
                return get_crawl_services_stats(request, headers)
            else:
                return list_crawl_services(request, headers)
        elif len(path_parts) == 2 and path_parts[0] == 'crawl-services':
            if path_parts[1] == 'stats':
                return get_crawl_services_stats(request, headers)
            else:
                return get_crawl_service(request, headers, path_parts[1])
        else:
            return list_crawl_services(request, headers)

    except Exception as e:
        print(f"Error in crawl_services_api: {e}")
        return (
            json.dumps({"error": "An internal server error occurred"}),
            500,
            headers,
        )

def list_crawl_services(request, headers):
    try:
        tenant_id = request.args.get('tenantId')
        if not tenant_id:
            return (
                json.dumps({"error": "tenantId query parameter is required"}),
                400,
                headers,
            )

        status_filter = request.args.get('status')
        manufacturer_filter = request.args.get('manufacturer')
        filter_type_filter = request.args.get('filterType')

        page_size_str = request.args.get('pageSize', '20')
        page_token = request.args.get('pageToken')

        try:
            page_size = int(page_size_str)
            if page_size <= 0:
                page_size = 20
            elif page_size > 100:
                page_size = 100
        except ValueError:
            page_size = 20

        query = db.collection(WEB_SERVICE_LOGS_COLLECTION)

        if tenant_id != 'all':
            query = query.where(filter=FieldFilter('tenantId', '==', tenant_id))

        if status_filter:
            query = query.where(filter=FieldFilter('status', '==', status_filter))

        if manufacturer_filter:
            query = query.where(filter=FieldFilter('manufacturer', '==', manufacturer_filter))

        if filter_type_filter:
            query = query.where(filter=FieldFilter('filterType', '==', filter_type_filter))

        query = query.order_by('dateCreated', direction=firestore.Query.DESCENDING)

        if page_token:
            try:
                last_doc_ref = db.collection(WEB_SERVICE_LOGS_COLLECTION).document(page_token)
                last_doc_snapshot = last_doc_ref.get()
                if last_doc_snapshot.exists:
                    query = query.start_after(last_doc_snapshot)
                else:
                    return (
                        json.dumps({"error": "Invalid pageToken: specified document not found"}),
                        400,
                        headers,
                    )
            except Exception as e:
                return (
                    json.dumps({"error": f"Invalid pageToken: {str(e)}"}),
                    400,
                    headers,
                )

        query_snapshot = query.limit(page_size + 1).stream()
        docs = list(query_snapshot)

        has_next_page = len(docs) > page_size
        if has_next_page:
            docs = docs[:page_size]

        services = []
        for doc in docs:
            service_data = doc.to_dict()
            if service_data:
                service_data['id'] = doc.id
                services.append(service_data)

        distinct_statuses = set()
        distinct_manufacturers = set()
        distinct_filter_types = set()
        for service in services:
            if isinstance(service, dict):
                if 'status' in service and service['status']:
                    distinct_statuses.add(service['status'])
                if 'manufacturer' in service and service['manufacturer']:
                    distinct_manufacturers.add(service['manufacturer'])
                if 'filterType' in service and service['filterType']:
                    distinct_filter_types.add(service['filterType'])

        next_page_token = None
        if has_next_page and docs:
            next_page_token = docs[-1].id

        response_data = {
            "services": services,
            "nextPageToken": next_page_token,
            "pageSize": page_size,
            "totalCount": len(services),
            "hasNextPage": has_next_page,
            "distinctStatuses": sorted(list(distinct_statuses)),
            "distinctManufacturers": sorted(list(distinct_manufacturers)),
            "distinctFilterTypes": sorted(list(distinct_filter_types))
        }

        return (json.dumps(response_data), 200, headers)

    except Exception as e:
        print(f"Error listing crawl services for tenant {tenant_id}: {e}")
        return (
            json.dumps({"error": "An internal server error occurred"}),
            500,
            headers,
        )

def get_crawl_service(request, headers, service_id):
    try:
        service_doc_ref = db.collection(WEB_SERVICE_LOGS_COLLECTION).document(service_id)
        service_doc = service_doc_ref.get()

        if not service_doc.exists:
            return (
                json.dumps({"error": "Crawl service not found"}),
                404,
                headers,
            )

        service_data = service_doc.to_dict()
        service_data['id'] = service_doc.id

        return (json.dumps(service_data), 200, headers)

    except Exception as e:
        print(f"Error getting crawl service {service_id}: {e}")
        return (
            json.dumps({"error": "An internal server error occurred"}),
            500,
            headers,
        )

def get_crawl_services_stats(request, headers):
    try:
        tenant_id = request.args.get('tenantId')
        if not tenant_id:
            return (
                json.dumps({"error": "tenantId query parameter is required"}),
                400,
                headers,
            )

        filter_type_filter = request.args.get('filterType')

        query = db.collection(WEB_SERVICE_LOGS_COLLECTION)

        if tenant_id != 'all':
            query = query.where(filter=FieldFilter('tenantId', '==', tenant_id))

        if filter_type_filter:
            query = query.where(filter=FieldFilter('filterType', '==', filter_type_filter))

        docs = query.stream()

        stats = {
            'total': 0,
            'pending': 0,
            'in_progress': 0,
            'completed': 0,
            'failed': 0,
            'manufacturers': set(),
            'filterTypes': set(),
            'recent_jobs': []
        }

        recent_jobs_limit = 5
        jobs = []

        for doc in docs:
            data = doc.to_dict()
            if data:
                stats['total'] += 1

                status = data.get('status', 'unknown')
                if status in stats:
                    stats[status] += 1

                manufacturer = data.get('manufacturer')
                if manufacturer:
                    stats['manufacturers'].add(manufacturer)

                filter_type = data.get('filterType')
                if filter_type:
                    stats['filterTypes'].add(filter_type)

                jobs.append({
                    'id': doc.id,
                    'jobId': data.get('jobId'),
                    'status': status,
                    'manufacturer': manufacturer,
                    'filterType': filter_type,
                    'dateCreated': data.get('dateCreated'),
                    'dateEdited': data.get('dateEdited')
                })

        jobs.sort(key=lambda x: x.get('dateCreated', ''), reverse=True)
        stats['recent_jobs'] = jobs[:recent_jobs_limit]

        stats['manufacturers'] = sorted(list(stats['manufacturers']))
        stats['filterTypes'] = sorted(list(stats['filterTypes']))

        return (json.dumps(stats), 200, headers)

    except Exception as e:
        print(f"Error getting crawl services stats for tenant {tenant_id}: {e}")
        return (
            json.dumps({"error": "An internal server error occurred"}),
            500,
            headers,
        )
