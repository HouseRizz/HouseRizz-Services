import functions_framework
import firebase_admin
from firebase_admin import credentials, firestore
from openai import OpenAI
from google.cloud import secretmanager
import os
import json
import uuid

firebase_admin.initialize_app()

db = firestore.client()

def get_secret(secret_id: str) -> str:
    """Fetch secret from Secret Manager at runtime."""
    client = secretmanager.SecretManagerServiceClient()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "houserizz-481012")
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Initialize OpenAI client with dynamic secret fetch
client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))

@functions_framework.http
def merge_products(request):
    if not client.api_key:
        return 'OPENAI_API_KEY environment variable not set.', 500

    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload.', 400

    master_product_id = request_json.get('master_product_id')
    secondary_product_id = request_json.get('secondary_product_id')

    if not master_product_id or not secondary_product_id:
        return 'Missing master_product_id or secondary_product_id in request.', 400

    try:
        master_doc_ref = db.collection('products').document(master_product_id)
        master_doc = master_doc_ref.get()
        if not master_doc.exists:
            return f'Master product with ID {master_product_id} not found.', 404
        master_data = master_doc.to_dict()

        secondary_doc_ref = db.collection('products').document(secondary_product_id)
        secondary_doc = secondary_doc_ref.get()
        if not secondary_doc.exists:
            return f'Secondary product with ID {secondary_product_id} not found.', 404
        secondary_data = secondary_doc.to_dict()

        prompt = (
            f"You are an intelligent assistant tasked with merging two product descriptions into a single, comprehensive, and accurate product record. "
            f"Prioritize information from the 'master product' when conflicts arise. The goal is to create a new, consolidated product entry. "
            f"Avoid redundancy and ensure all key information is present. Output the merged product data as a JSON object.\n\n"
            f"Master Product (ID: {master_product_id}):\n{json.dumps(master_data, indent=2)}\n\n"
            f"Secondary Product (ID: {secondary_product_id}):\n{json.dumps(secondary_data, indent=2)}\n\n"
            f"Please provide the merged product data as a JSON object:"
        )

        response = client.responses.create(
            model="o4-mini",
            input=[
                {"role": "user", "content": prompt}
            ],
            reasoning={"effort": "low"}
        )

        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            print("Ran out of tokens")
            if response.output_text:
                print("Partial output:", response.output_text)
                merged_product_str = response.output_text
            else:
                print("Ran out of tokens during reasoning")
                return "Failed to generate complete response due to token limit", 500
        else:
            merged_product_str = response.output_text
        
        try:
            merged_product_str = merged_product_str.strip()
            if merged_product_str.startswith('```json'):
                merged_product_str = merged_product_str[7:]
            if merged_product_str.endswith('```'):
                merged_product_str = merged_product_str[:-3]
            merged_product_str = merged_product_str.strip()
            
            merged_product_data = json.loads(merged_product_str)
            if 'rawScrappedData' in merged_product_data:
                del merged_product_data['rawScrappedData']
            if 'id' in merged_product_data:
                del merged_product_data['id']
            
            merged_product_data['source'] = {
                'master_product_id': master_product_id,
                'secondary_product_id': secondary_product_id
            }

            new_product_id = str(uuid.uuid4()).upper()
            new_product_ref = db.collection('products').document(new_product_id)
            new_product_ref.set(merged_product_data)

            master_doc_ref.update({'status': f'merged_master_{new_product_id}'})
            secondary_doc_ref.update({'status': f'merged_secondary_{new_product_id}'})

            return {
                'message': 'Products merged successfully and statuses updated.',
                'new_product_id': new_product_id,
                'merged_data': merged_product_data
            }, 200
        except json.JSONDecodeError:
            print(f"Error decoding OpenAI response: {merged_product_str}")
            return f'Failed to parse merged product data from LLM. Response was: {merged_product_str}', 500

        merged_product_data['status'] = 'merged'

        new_product_ref = db.collection('products').document()
        new_product_ref.set(merged_product_data)
        new_merged_product_id = new_product_ref.id

        master_doc_ref.update({'status': f'merged_master_{new_merged_product_id}'})

        secondary_doc_ref.update({'status': f'merged_secondary_{new_merged_product_id}'})

        return {
            'message': 'Products merged successfully and statuses updated.',
            'new_product_id': new_merged_product_id,
            'merged_data': merged_product_data
        }, 200

    except firebase_admin.FirebaseError as e:
        print(f"Firebase error: {e}")
        return f'An error occurred with Firestore: {str(e)}', 500
    except client.error.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return f'An error occurred with OpenAI API: {str(e)}', 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f'An unexpected error occurred: {str(e)}', 500
