import os
import json
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize Firebase with default credentials (for Cloud Run)
if not firebase_admin._apps:
    firebase_admin.initialize_app()

db = firestore.client()

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "furniture")

def _get_tenant_id_arg() -> str | None:
    """Return ?tenantId= or ?tenant_id= value, else None."""
    return request.args.get("tenantId") or request.args.get("tenant_id")

@app.route('/status/<process_id>', methods=['GET'])
def get_process_status(process_id):
    """
    Get the status and details of a specific process by its ID.

    Args:
        process_id: The unique identifier for the process

    Returns:
        JSON response with process status and details
    """
    try:
        process_ref = db.collection('webServiceLogs').document(process_id)
        process_doc = process_ref.get()

        if not process_doc.exists:
            return jsonify({
                'status': 'error',
                'message': f'Process with ID {process_id} not found'
            }), 404

        process_data = process_doc.to_dict()

        response = {
            'process_id': process_id,
            'status': process_data.get('status', 'unknown'),
            'start_time': process_data.get('start_time'),
            'tenant_id': process_data.get('tenant_id'),
            'end_time': process_data.get('end_time'),
            'last_updated': process_data.get('last_updated'),
            'csv_url': process_data.get('csv_url'),
            'products_processed': process_data.get('products_processed', 0),
            'products_failed': process_data.get('products_failed', 0),
        }

        if 'error' in process_data:
            response['error'] = process_data['error']

        include_products = request.args.get('include_products', 'false').lower() == 'true'
        if include_products and 'products' in process_data:
            response['products'] = process_data['products']

        return jsonify(response)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving process status: {str(e)}',
            'details': error_details
        }), 500

@app.route('/status', methods=['GET'])
def list_processes():
    """
    List all processes with their basic status information.

    Query parameters:
        limit: Maximum number of processes to return (default: 10)
        status: Filter by status (optional)
        start_date: Filter by start date (ISO format, optional)
        end_date: Filter by end date (ISO format, optional)

    Returns:
        JSON response with list of processes
    """
    try:
        limit = int(request.args.get('limit', 10))
        status_filter = request.args.get('status')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        tenant_id = _get_tenant_id_arg()

        query = db.collection('webServiceLogs')

        if status_filter:
            query = query.where('status', '==', status_filter)

        if start_date:
            try:
                start_datetime = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.where('start_time', '>=', start_datetime.isoformat() + 'Z')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid start_date format: {start_date}. Use ISO format (YYYY-MM-DDTHH:MM:SSZ).'
                }), 400

        if end_date:
            try:
                end_datetime = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.where('start_time', '<=', end_datetime.isoformat() + 'Z')
            except ValueError:
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid end_date format: {end_date}. Use ISO format (YYYY-MM-DDTHH:MM:SSZ).'
                }), 400

        if tenant_id:
            query = query.where('tenant_id', '==', tenant_id)

        query = query.order_by('start_time', direction=firestore.Query.DESCENDING).limit(limit)

        processes = []
        for doc in query.stream():
            process_data = doc.to_dict()
            processes.append({
                'process_id': doc.id,
                'status': process_data.get('status', 'unknown'),
                'start_time': process_data.get('start_time'),
                'tenant_id': process_data.get('tenant_id'),
                'end_time': process_data.get('end_time'),
                'last_updated': process_data.get('last_updated'),
                'products_processed': process_data.get('products_processed', 0),
                'products_failed': process_data.get('products_failed', 0),
            })

        return jsonify({
            'processes': processes,
            'count': len(processes)
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error listing processes: {str(e)}',
            'details': error_details
        }), 500

@app.route('/', methods=['GET'])
def health_check():
    """
    Simple health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'service': 'process-status-service',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

# ------------------------------------------------------------------
# Tenant-specific quick list
# ------------------------------------------------------------------
@app.route("/tenant/<tenant_id>/processes", methods=["GET"])
def tenant_processes(tenant_id):
    try:
        limit = int(request.args.get("limit", 50))
        docs = (
            db.collection("webServiceLogs")
            .where("tenantId", "==", tenant_id)
            .order_by("start_time", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return jsonify(
            {
                "tenantId": tenant_id,
                "processes": [d.to_dict() for d in docs],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

@app.route('/master_task_status/<master_process_id>', methods=['GET'])
def get_master_task_status(master_process_id):
    """API endpoint to get the status of a master scraping task and all its batches."""
    try:
        tenant_filter = _get_tenant_id_arg()
        if not db:
            return jsonify({'error': 'Firestore not initialized'}), 500

        master_doc_ref = db.collection('webServiceLogs').document(master_process_id)
        master_doc = master_doc_ref.get()

        if not master_doc.exists:
            return jsonify({'error': 'Master process not found'}), 404

        master_data = master_doc.to_dict()


        if master_data.get('type') != 'master':
            return jsonify({'error': 'Specified ID is not a master process'}), 400

        if tenant_filter and data.get("tenantId") != tenant_filter:
            return jsonify({"error": "Forbidden"}), 403
        batch_processes = master_data.get('batch_processes', [])
        detailed_batches = []

        for batch in batch_processes:
            batch_id = batch.get('process_id')
            if batch_id:
                batch_doc_ref = db.collection('webServiceLogs').document(batch_id)
                batch_doc = batch_doc_ref.get()

                if batch_doc.exists:
                    batch_data = batch_doc.to_dict()

                    if 'products' in batch_data and len(batch_data['products']) > 10:
                        batch_data['products'] = batch_data['products'][:10]
                        batch_data['products_truncated'] = True

                    detailed_batches.append(batch_data)
                else:

                    detailed_batches.append(batch)


        master_data['detailed_batches'] = detailed_batches


        if master_data.get('total_products', 0) > 0:
            processed = master_data.get('products_processed', 0)
            failed = master_data.get('products_failed', 0)
            total = master_data.get('total_products', 0)
            progress = ((processed + failed) / total) * 100
            master_data['progress_percentage'] = round(progress, 2)
        else:
            master_data['progress_percentage'] = 0

        return jsonify({
            'status': 'success',
            'master_process': master_data
        })
    except Exception as e:
        error_msg = f"Exception in get_master_task_status: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/active_batch_processes', methods=['GET'])
def get_active_batch_processes():
    """API endpoint to get a list of all active master batch processes."""
    try:
        if not db:
            return jsonify({'error': 'Firestore not initialized'}), 500

        tenant_filter = _get_tenant_id_arg()

        q = db.collection('webServiceLogs').where('type', '==', 'master').where('status', 'in', ['created', 'processing'])

        if tenant_filter:
            q = q.where("tenantId", "==", tenant_filter)
        master_processes_ref = q.get()
        active_processes = []
        for doc in master_processes_ref:
            process_data = doc.to_dict()

            active_processes.append({
                'process_id': process_data.get('process_id'),
                'tenantId': process_data.get('tenantId'),
                'status': process_data.get('status'),
                'csv_url': process_data.get('csv_url'),
                'total_batches': process_data.get('total_batches', 0),
                'completed_batches': process_data.get('completed_batches', 0),
                'total_products': process_data.get('total_products', 0),
                'products_processed': process_data.get('products_processed', 0),
                'products_failed': process_data.get('products_failed', 0),
                'start_time': process_data.get('start_time'),
                'last_updated': process_data.get('last_updated'),
                'progress_percentage': round(
                    (process_data.get('products_processed', 0) + process_data.get('products_failed', 0)) /
                    process_data.get('total_products', 1) * 100, 2) if process_data.get('total_products', 0) > 0 else 0
            })


        one_day_ago = (datetime.utcnow() - timedelta(days=1)).replace(microsecond=0).isoformat() + 'Z'
        completed_processes_ref = db.collection('webServiceLogs').where('type', '==', 'master').where('status', '==', 'completed').where('last_updated', '>=', one_day_ago).stream()

        completed_processes = []
        for doc in completed_processes_ref:
            process_data = doc.to_dict()
            completed_processes.append({
                'process_id': process_data.get('process_id'),
                'status': process_data.get('status'),
                'csv_url': process_data.get('csv_url'),
                'total_batches': process_data.get('total_batches', 0),
                'completed_batches': process_data.get('completed_batches', 0),
                'total_products': process_data.get('total_products', 0),
                'products_processed': process_data.get('products_processed', 0),
                'products_failed': process_data.get('products_failed', 0),
                'start_time': process_data.get('start_time'),
                'end_time': process_data.get('end_time'),
                'csv_updated': process_data.get('csv_updated', False)
            })

        return jsonify({
            'status': 'success',
            'active_processes': active_processes,
            'recent_completed_processes': completed_processes
        })
    except Exception as e:
        error_msg = f"Exception in get_active_batch_processes: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/cancel_batch_process/<master_process_id>', methods=['POST'])
def cancel_batch_process(master_process_id):
    """API endpoint to cancel a batch processing operation."""
    try:
        if not db:
            return jsonify({'error': 'Firestore not initialized'}), 500

        master_doc_ref = db.collection('webServiceLogs').document(master_process_id)
        master_doc = master_doc_ref.get()

        if not master_doc.exists:
            return jsonify({'error': 'Master process not found'}), 404

        master_data = master_doc.to_dict()


        if master_data.get('type') != 'master':
            return jsonify({'error': 'Specified ID is not a master process'}), 400


        if master_data.get('status') in ['completed', 'failed', 'canceled']:
            return jsonify({
                'status': 'warning',
                'message': f"Process already in {master_data.get('status')} state"
            })

        current_time = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


        master_doc_ref.update({
            'status': 'canceled',
            'end_time': current_time,
            'last_updated': current_time,
            'cancel_reason': 'User requested cancellation'
        })


        batch_processes = master_data.get('batch_processes', [])
        for batch in batch_processes:
            if batch.get('status') in ['created', 'processing']:
                batch_id = batch.get('process_id')
                if batch_id:
                    batch_doc_ref = db.collection('webServiceLogs').document(batch_id)
                    batch_doc_ref.update({
                        'status': 'canceled',
                        'end_time': current_time,
                        'last_updated': current_time,
                        'cancel_reason': 'Master process was canceled'
                    })

        return jsonify({
            'status': 'success',
            'message': f"Successfully canceled batch process {master_process_id}"
        })
    except Exception as e:
        error_msg = f"Exception in cancel_batch_process: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500
