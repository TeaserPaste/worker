import os
import logging
import time
import json
import re
import sys
from collections import Counter
# SỬA LỖI IMPORT: Sửa lại import từ priority_rules để lấy đúng đối tượng
# Giả sử hàm chính trong priority_rules là 'calculate_priority'
# và nó trả về một dictionary.
from priority_rules import calculate_priority
# SỬA LỖI IMPORT: loại bỏ Timestamp khỏi google.cloud.firestore_v1.
# Chúng ta sẽ sử dụng firebase_admin.firestore.Timestamp thay thế khi cần.
import firebase_admin
from firebase_admin import credentials, firestore
# Import WriteBatch từ google.cloud.firestore_v1 vì nó vẫn cần thiết cho firestore.client().batch()
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.firestore_v1 import WriteBatch 
from opensearchpy import OpenSearch, helpers, exceptions as os_exceptions 
from dotenv import load_dotenv 
import datetime
import requests
import boto3
import uuid
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# --- Cấu hình logging ---
# Fix lỗi: Nếu LOG_LEVEL rỗng (cronjob không set), ta phải gán giá trị mặc định 'INFO'.
log_level_env = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = log_level_env if log_level_env else 'INFO'

# Set logging level cho module priority_rules
logging.getLogger("priority_rules").setLevel(logging.DEBUG if log_level == 'DEBUG' else logging.INFO)
# Giữ nguyên các cấu hình logging khác
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("opensearch").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING) 
logging.getLogger("botocore").setLevel(logging.WARNING) 
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Load .env 
load_dotenv()

# --- Khởi tạo Firebase Admin SDK ---
db = None
try:
    if not firebase_admin._apps:
        # Thay thế \n thành ký tự xuống dòng thực tế
        private_key = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        client_email = os.getenv("FIREBASE_CLIENT_EMAIL")

        if not all([project_id, client_email, private_key]):
             logging.warning("Critical Firebase credentials (project_id, client_email, private_key) are missing.")
             raise ValueError("Critical Firebase credentials (project_id, client_email, private_key) are missing.")

        cred_obj = {
            "type": "service_account", "project_id": project_id,
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""), "private_key": private_key,
            "client_email": client_email, "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
            "auth_uri": os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth"),
            "token_uri": os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token"),
            "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL", "https://www.googleapis.com/oauth2/v1/certs"),
        }
        client_cert_url_env = os.getenv("FIREBASE_CLIENT_CERT_URL")
        if client_cert_url_env:
             cred_obj["client_x509_cert_url"] = client_cert_url_env


        cred = credentials.Certificate(cred_obj)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase Admin SDK initialized using specific environment variables (GitHub Secrets).")
    else:
        logging.info("Firebase Admin SDK already initialized.")

    db = firestore.client()
except Exception as e:
    logging.error(f"Failed to initialize Firebase Admin SDK or get Firestore client: {e}", exc_info=True)
    db = None 

# --- Initialize OpenSearch Client ---
os_client = None
opensearch_host = os.getenv("OPENSEARCH_HOST")
opensearch_port = int(os.getenv("OPENSEARCH_PORT", 9200))
opensearch_user = os.getenv("OPENSEARCH_USER")
opensearch_password = os.getenv("OPENSEARCH_PASSWORD")
# Mặc định an toàn là https
opensearch_scheme = os.getenv("OPENSEARCH_SCHEME", "https") 
opensearch_index = os.getenv("OPENSEARCH_INDEX", "snippets")

if not opensearch_host:
    logging.error("OPENSEARCH_HOST environment variable not set.")
else:
    auth = (opensearch_user, opensearch_password) if opensearch_user and opensearch_password else None
    try:
        # Bỏ verify_certs=True nếu scheme là http, nhưng tốt nhất nên dùng https
        verify_certs_val = opensearch_scheme == "https"
        os_client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}], http_auth=auth,
            use_ssl=opensearch_scheme == "https", 
            verify_certs=verify_certs_val, # Chỉ kiểm tra certs nếu dùng HTTPS
            ssl_assert_hostname=False, 
            ssl_show_warn=False,
            timeout=90, retry_on_timeout=True, max_retries=2
        )
        logging.info(f"OpenSearch client initialized for host: {opensearch_host}")
        if not os_client.ping():
             logging.warning("OpenSearch cluster ping failed.")
        else:
             logging.info("OpenSearch cluster ping successful.")
    except Exception as e:
        logging.error(f"Failed to initialize or ping OpenSearch client: {e}")
        os_client = None

# --- Initialize Cloudflare R2 Client ---
r2_endpoint_url = os.getenv("R2_ENDPOINT_URL")
r2_access_key_id = os.getenv("R2_ACCESS_KEY_ID")
r2_secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
r2_bucket_name = os.getenv("R2_BUCKET_NAME")
r2_recovery_prefix = os.getenv("R2_RECOVERY_PREFIX", "deleted_snippets/")

s3_client = None
if r2_endpoint_url and r2_access_key_id and r2_secret_access_key and r2_bucket_name:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            config=Config(signature_version='s3v4'),
            region_name='auto' 
        )
        logging.info("Cloudflare R2 (S3 compatible) client initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize R2 client: {e}")
        s3_client = None
else:
    logging.warning("R2 environment variables not fully set. Snippet recovery backup will be disabled.")


# --- Configure D1 Logging API ---
D1_LOG_API_URL = os.getenv("D1_LOG_API_URL")
session_id = None 

# --- D1 Logging Functions ---
def open_log_session(app_name="os_sync_worker"):
    """Opens a new logging session with the D1 API."""
    global session_id
    session_id = None 
    if not D1_LOG_API_URL:
        logging.warning("D1_LOG_API_URL not set, cannot open log session.")
        return False
    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/open", json={"app_name": app_name}, timeout=10)
        response.raise_for_status() 
        data = response.json()
        session_id = data.get("session_id")
        if session_id:
            logging.info(f"Opened D1 log session: {session_id}")
            return True
        else:
            logging.error(f"Failed to get session_id from D1 API response: {data}")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to open D1 log session due to RequestException: {e}")
        session_id = None 
        return False
    except Exception as e:
        logging.error(f"Unexpected error opening D1 log session: {e}")
        session_id = None
        return False

def log_event(event_type, details=None, status="info", snippet_id=None):
    """Logs an event to the currently open D1 session."""
    if not D1_LOG_API_URL or not session_id:
        log_level_map = {"INFO": logging.INFO, "WARN": logging.WARNING, "ERROR": logging.ERROR, "DEBUG": logging.DEBUG}
        log_func = logging.log
        level = log_level_map.get(status.upper(), logging.INFO)
        log_message = f"D1_LOG_SKIP ({status.upper()}): {event_type}"
        if snippet_id: log_message += f" [Snippet: {snippet_id}]"
        log_details_str = ""
        if details:
            try: log_details_str = json.dumps(details)[:500] 
            except TypeError: log_details_str = "{Non-serializable details}"
            log_message += f" Details: {log_details_str}"
        log_func(level, log_message)
        return

    log_details = details if isinstance(details, dict) else {}
    if snippet_id: log_details["snippet_id"] = snippet_id

    serializable_details = {}
    for key, value in log_details.items():
        if isinstance(value, str) and len(value) > 250: 
            serializable_details[key] = value[:250] + "..."
        elif isinstance(value, Exception):
            try: serializable_details[key] = str(value)
            except Exception: serializable_details[key] = f"Unrepresentable Exception: {type(value).__name__}"
        elif isinstance(value, (datetime.datetime, datetime.date)):
            serializable_details[key] = value.isoformat()
        elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
             serializable_details[key] = value 
        else:
            try:
                repr_str = repr(value)
                serializable_details[key] = (repr_str[:250] + "...") if len(repr_str) > 250 else repr_str
            except Exception:
                serializable_details[key] = f"Non-representable type: {type(value).__name__}"


    log_entry = {
        "message": str(event_type),       
        "level": str(status).upper(),    
        "details": serializable_details 
    }

    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/{session_id}/log", json=log_entry, timeout=5)
        if response.status_code == 400:
            logging.error(f"D1 Log Error 400 (Bad Request). Session: {session_id}. Payload: {json.dumps(log_entry)}. Response: {response.text[:500]}")
        elif response.status_code >= 400: 
            logging.error(f"D1 Log HTTP Error {response.status_code}. Session: {session_id}. Payload (truncated): {json.dumps(log_entry)[:500]}. Response: {response.text[:500]}")

        response.raise_for_status() 
        logging.debug(f"D1 log successful (status: {response.status_code})") 

    except requests.exceptions.RequestException as e:
        truncated_payload = json.dumps(log_entry)[:500] 
        logging.error(f"Failed to log event to D1 session {session_id} due to RequestException: {e}. Payload (truncated): {truncated_payload}")
    except Exception as e:
        truncated_payload = json.dumps(log_entry)[:500]
        logging.error(f"Unexpected error logging event to D1 session {session_id}: {e}. Payload (truncated): {truncated_payload}", exc_info=True)


def close_log_session():
    """Closes the current D1 logging session."""
    global session_id
    if not D1_LOG_API_URL or not session_id:
        return 
    current_session = session_id 
    session_id = None 
    try:
        response = requests.post(f"{D1_LOG_API_URL}/api/session/{current_session}/close", timeout=10)
        response.raise_for_status()
        logging.info(f"Closed D1 log session: {current_session}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to close D1 log session {current_session} due to RequestException: {e}")
    except Exception as e:
        logging.error(f"Unexpected error closing D1 log session {current_session}: {e}")


# --- Discord Notification Function ---
def send_discord_notification(message=None, embeds=None, level="info"):
    """Sends a notification to the configured Discord webhook."""
    discord_webhook_url_local = os.getenv("DISCORD_WEBHOOK_URL")
    if not discord_webhook_url_local:
        if not getattr(send_discord_notification, 'logged_missing_url', False):
            logging.warning("DISCORD_WEBHOOK_URL not set. Discord notifications disabled.")
            setattr(send_discord_notification, 'logged_missing_url', True)
        return
    if getattr(send_discord_notification, 'logged_missing_url', False):
        setattr(send_discord_notification, 'logged_missing_url', False)

    payload = {}
    color_map = {"info": 0x3498db, "success": 0x2ecc71, "warning": 0xf1c40f, "error": 0xe74c3c}

    if message:
        payload['content'] = str(message)[:2000] 

    if embeds:
        if not isinstance(embeds, list): embeds = [embeds]
        payload['embeds'] = embeds[:10]
        if level:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for embed in payload['embeds']:
                if isinstance(embed, dict):
                    embed['color'] = color_map.get(str(level).lower(), 0x3498db)
                    if 'timestamp' not in embed: 
                        embed['timestamp'] = timestamp
    try:
        response = requests.post(discord_webhook_url_local, json=payload, timeout=10)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Discord notification: {e}")
    except Exception as e: 
        logging.error(f"Unexpected error sending Discord notification: {e}")


# --- Get Last Processed Timestamp from OpenSearch ---
def get_last_processed_timestamp():
    """Fetches the latest 'processed_at' timestamp from the OpenSearch index."""
    if not os_client:
        logging.error("OpenSearch client not available.")
        return None
    try:
        query = {
            "size": 0, 
            "aggs": {
                "max_processed_at": {
                    "max": {
                        "field": "processed_at",
                        "format": "strict_date_optional_time_nanos"
                    }
                }
            }
        }
        res = os_client.search(index=opensearch_index, body=query, request_timeout=60)
        max_ts_value = res['aggregations']['max_processed_at'].get('value')

        if max_ts_value:
            last_processed_dt = datetime.datetime.fromtimestamp(max_ts_value / 1000.0, tz=datetime.timezone.utc)
            logging.info(f"Last processed timestamp found in OpenSearch: {last_processed_dt.isoformat()}")
            return last_processed_dt
        else:
            logging.info("No 'processed_at' timestamp found in OpenSearch index. Performing full sync.")
            return None
    except os_exceptions.NotFoundError:
        logging.info(f"OpenSearch index '{opensearch_index}' not found. Performing full sync.")
        return None
    except os_exceptions.RequestError as e:
        logging.error(f"OpenSearch request error getting max timestamp: Status {e.status_code}, Info: {e.info}, Error: {e.error}")
        return None
    except Exception as e:
        logging.error(f"Error getting last processed timestamp from OpenSearch: {e}", exc_info=True)
        return None

# --- Phase 2: Purge Deleted Snippets ---
def purge_deleted_snippets(stats_counter: Counter):
    """
    Queries Firestore for 'deleted' snippets, backs them up to R2,
    and permanently deletes them from Firestore.
    """
    if not db:
        logging.error("Firestore client unavailable for purging.")
        return 0, 0, 0, 0

    purge_start_time = time.time()
    if not open_log_session(app_name="purge_worker"):
        logging.warning("Proceeding with purge without D1 logging session.")

    purged_count = 0
    query_count = 0
    backup_failed_count = 0
    backup_skipped_count = 0
    batch_limit = 400
    total_batches = 0

    try:
        logging.info("Starting Purge Phase: Querying for 'deleted' snippets...")
        log_event("purge_started", status="INFO")

        snippets_ref = db.collection('snippets')
        query = snippets_ref.where(filter=FieldFilter('visibility', '==', 'deleted'))

        docs_stream = query.stream()

        batch = db.batch()
        current_batch_size = 0

        for doc in docs_stream:
            query_count += 1
            snippet_id = doc.id
            logging.debug(f"Processing snippet {snippet_id} for purge.")
            log_event("snippet_queued_for_purge", snippet_id=snippet_id, status="DEBUG")

            # --- R2 Recovery Backup ---
            if s3_client and r2_bucket_name:
                try:
                    snippet_data = doc.to_dict()

                    # Check allowBackup (default to True if missing)
                    if snippet_data.get('allowBackup') is False:
                        logging.info(f"Skipping R2 backup for snippet {snippet_id} (allowBackup=False).")
                    else:
                        recovery_json = json.dumps(snippet_data, default=str)

                        today_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
                        unique_filename = f"{today_str}_{uuid.uuid4()}.json"
                        prefix = r2_recovery_prefix if r2_recovery_prefix.endswith('/') else f"{r2_recovery_prefix}/"
                        object_key = f"{prefix}{snippet_id}/{unique_filename}"

                        s3_client.put_object(
                            Bucket=r2_bucket_name,
                            Key=object_key,
                            Body=recovery_json.encode('utf-8'),
                            ContentType='application/json'
                        )
                        log_event("recovery_backup_success_r2", snippet_id=snippet_id, details={"key": object_key}, status="INFO")
                        logging.info(f"Successfully backed up snippet {snippet_id} to R2: {object_key}")

                except ClientError as ce:
                    backup_failed_count += 1
                    stats_counter['recovery_backup_failed_r2'] += 1
                    log_event("recovery_backup_failed_r2", details={"error": str(ce)}, status="ERROR", snippet_id=snippet_id)
                    logging.error(f"R2 ClientError backing up snippet {snippet_id}: {ce}")
                except Exception as recovery_err:
                    backup_failed_count += 1
                    stats_counter['recovery_backup_failed_r2'] += 1
                    log_event("recovery_backup_failed_r2", details={"error": str(recovery_err)}, status="ERROR", snippet_id=snippet_id)
                    logging.error(f"Failed to back up snippet {snippet_id} to R2: {recovery_err}", exc_info=True)
            else:
                backup_skipped_count += 1
                logging.debug(f"R2 client not configured. Skipping recovery backup for {snippet_id}.")
                if query_count == 1:
                    log_event("recovery_backup_skipped_r2_config", status="WARN")
            # --- End R2 Recovery Backup ---

            batch.delete(doc.reference)
            current_batch_size += 1

            if current_batch_size >= batch_limit:
                logging.info(f"Committing purge batch of {current_batch_size} snippets...")
                batch.commit()
                purged_count += current_batch_size
                total_batches += 1
                logging.info(f"Committed batch {total_batches}. Total purged so far: {purged_count}")
                batch = db.batch()
                current_batch_size = 0
                time.sleep(0.5)

        if current_batch_size > 0:
            logging.info(f"Committing final purge batch of {current_batch_size} snippets...")
            batch.commit()
            purged_count += current_batch_size
            total_batches += 1
            logging.info(f"Committed final batch {total_batches}. Total purged: {purged_count}")

        purge_duration = round(time.time() - purge_start_time, 2)
        logging.info(f"Purge Phase Completed. Found {query_count} deleted snippets, purged {purged_count} in {total_batches} batches. Backups Failed: {backup_failed_count}, Skipped: {backup_skipped_count}. Duration: {purge_duration}s")
        log_event("purge_finished", details={
            "found": query_count,
            "purged": purged_count,
            "batches": total_batches,
            "backup_failed_r2": backup_failed_count,
            "backup_skipped_r2": backup_skipped_count,
            "duration_sec": purge_duration
        }, status="INFO" if backup_failed_count == 0 else "WARN")

        return query_count, purged_count, backup_failed_count, backup_skipped_count

    except Exception as e:
        purge_duration = round(time.time() - purge_start_time, 2)
        error_message = f"Critical error during Purge Phase after {purge_duration}s: {e}"
        logging.error(error_message, exc_info=True)
        log_event("critical_purge_error", details={"error": str(e)[:500], "duration_sec": purge_duration}, status="ERROR")
        send_discord_notification(embeds=[{"title": f"❌ Purge Phase Failed Critically", "description": error_message[:1000]}], level="error")
        return query_count, purged_count, backup_failed_count, backup_skipped_count
    finally:
        close_log_session()


def run_sync():
    """Main function to run Index and Purge phases. (Not an endpoint)"""
    if not db:
        error_msg = "Firestore client unavailable. Aborting sync."
        logging.error(error_msg)
        send_discord_notification(embeds=[{"title": "❌ Sync Aborted", "description": error_msg}], level="error")
        sys.exit(1)
    if not os_client:
        error_msg = "OpenSearch client unavailable. Aborting sync."
        logging.error(error_msg)
        send_discord_notification(embeds=[{"title": "❌ Sync Aborted", "description": error_msg}], level="error")
        sys.exit(1)

    overall_start_time = time.time()
    sync_results = {}
    purge_results = {}
    stats_counter = Counter()

    # --- Phase 1: Indexing ---
    phase1_start_time = time.time()
    global session_id
    session_id = None
    if not open_log_session(app_name="index_worker"):
        logging.warning("Proceeding with Indexing without D1 logging session.")

    last_processed_at_dt = get_last_processed_timestamp()
    sync_type = "Incremental Index" if last_processed_at_dt else "Full Index"
    logging.info(f"Starting Phase 1: {sync_type}...")
    log_event("index_started", details={"type": sync_type, "since": last_processed_at_dt.isoformat() if last_processed_at_dt else "None"}, status="INFO")
    if last_processed_at_dt:
        send_discord_notification(message=f"🚀 Starting {sync_type} since {last_processed_at_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}...")
    else:
        send_discord_notification(message=f"🚀 Starting Full Index (no previous timestamp found)...")

    processed, indexed, index_failed, skipped_rules, expired = 0, 0, 0, 0, 0
    rule_based_analyzed = 0
    actions = []
    indexing_errors_details = []
    phase1_successful = True

    try:
        current_run_time = datetime.datetime.now(datetime.timezone.utc)
        snippets_ref = db.collection('snippets')
        query = snippets_ref.where(filter=FieldFilter('visibility', '==', 'public'))

        if last_processed_at_dt:
            query = query.where(filter=FieldFilter('updatedAt', '>', last_processed_at_dt))
            query = query.where(filter=FieldFilter('updatedAt', '<=', current_run_time))
            logging.info(f"Querying Firestore for public snippets updated in window: ({last_processed_at_dt.isoformat()}, {current_run_time.isoformat()}]")
        else:
            query = query.where(filter=FieldFilter('updatedAt', '<=', current_run_time))
            logging.info(f"Querying Firestore for ALL public snippets up to {current_run_time.isoformat()}...")

        docs_stream = query.order_by('updatedAt', direction=firestore.Query.ASCENDING).stream()

        for doc in docs_stream:
            processed += 1
            snippet_id = doc.id
            snippet_data = doc.to_dict()

            snippet_updated_at = snippet_data.get('updatedAt')
            if isinstance(snippet_updated_at, datetime.datetime):
                if snippet_updated_at.tzinfo is None:
                    snippet_updated_at = snippet_updated_at.replace(tzinfo=datetime.timezone.utc)
                if last_processed_at_dt and snippet_updated_at <= last_processed_at_dt:
                    continue
                if snippet_updated_at > current_run_time:
                    continue
            if snippet_data.get('visibility') != 'public':
                continue

            expires_at = snippet_data.get('expiresAt')
            if expires_at:
                try:
                    expiry_dt = None
                    if isinstance(expires_at, str):
                        expiry_dt = datetime.datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                    elif isinstance(expires_at, datetime.datetime):
                        expiry_dt = expires_at.replace(tzinfo=datetime.timezone.utc) if expires_at.tzinfo is None else expires_at
                    if expiry_dt and expiry_dt < current_run_time:
                        expired += 1
                        log_event("snippet_expired_indexing", snippet_id=snippet_id, status="INFO")
                        logging.info(f"Snippet {snippet_id} expired, skipping index.")
                        continue
                except Exception as e:
                    stats_counter['expiry_parse_error_indexing'] += 1
                    log_event("expiry_parse_error_indexing", {"error": str(e)}, status="WARN", snippet_id=snippet_id)

            content_to_analyze = snippet_data.get('content', '')
            snippet_lang = snippet_data.get('language', 'plaintext')
            snippet_created_at = snippet_data.get('createdAt')

            if snippet_created_at and isinstance(snippet_created_at, datetime.datetime):
                priority_score, assessment_string = calculate_priority(
                    content=content_to_analyze,
                    language=snippet_lang,
                    created_at=snippet_created_at,
                    is_verified=snippet_data.get('isVerified', False)
                )
                ai_priority = priority_score
                ai_assessment = assessment_string
                analysis_source = "rule_based"
                rule_based_analyzed += 1

                if ai_priority <= 0.1:
                    skipped_rules += 1
                    analysis_source = "rule_skip"
                    log_event("skipped_by_rule_indexing", snippet_id=snippet_id, details={"priority": ai_priority}, status="INFO")
            else:
                ai_priority = 0.1
                ai_assessment = "CRITICAL ERROR: Missing createdAt field. Priority set to minimum."
                analysis_source = "data_error"
                stats_counter['missing_created_at_error'] += 1
                log_event("missing_created_at_error", details={"priority": ai_priority}, status="ERROR", snippet_id=snippet_id)

            updated_fields = {
                'ai_priority': float(ai_priority),
                'ai_assessment': ai_assessment,
                'processed_at': current_run_time.isoformat(),
                'analysis_source': analysis_source,
            }

            # --- SỬA LỖI: Chuẩn bị doc cho upsert ---
            # Tạo một bản sao của dữ liệu gốc từ Firestore để không làm thay đổi nó
            upsert_doc = snippet_data.copy()
            # Cập nhật (hoặc thêm) các trường đã tính toán vào bản sao này
            upsert_doc.update(updated_fields)

            # Đảm bảo tất cả các trường datetime được chuyển đổi thành chuỗi ISO 8601
            # mà OpenSearch có thể hiểu được.
            for key, value in upsert_doc.items():
                if isinstance(value, datetime.datetime):
                    # Gán múi giờ UTC nếu datetime object là "naive"
                    if value.tzinfo is None:
                        upsert_doc[key] = value.replace(tzinfo=datetime.timezone.utc).isoformat()
                    else:
                        upsert_doc[key] = value.isoformat()

            action = {
                "_op_type": "update",
                "_index": opensearch_index,
                "_id": snippet_id,
                "doc": updated_fields,
                "upsert": upsert_doc  # Nếu doc không tồn tại, chèn 'upsert_doc'
            }
            actions.append(action)

        if actions:
            logging.info(f"Attempting to bulk index/update {len(actions)} snippets...")
            try:
                success_count, errors = helpers.bulk(os_client, actions, raise_on_error=False, raise_on_exception=False, request_timeout=120)
                indexed = success_count
                index_failed = len(errors)
                logging.info(f"Bulk indexing completed. Success: {indexed}, Failed: {index_failed}")
                if errors:
                    for i, error_info in enumerate(errors):
                        item_details = error_info.get('update') or {}
                        doc_id = item_details.get('_id', 'N/A')
                        err_details_obj = item_details.get('error', {})
                        err_str = json.dumps(err_details_obj)[:500] if isinstance(err_details_obj, dict) else str(err_details_obj)[:500]
                        stats_counter['indexing_error'] += 1
                        log_event("indexing_error", details={"error": err_str}, status="ERROR", snippet_id=doc_id)
                        if i < 5:
                            indexing_errors_details.append(f"ID:{doc_id} Err:{err_str[:200]}")
                    logging.error(f"First {len(indexing_errors_details)} bulk errors: {'; '.join(indexing_errors_details)}")
                    phase1_successful = False
            except os_exceptions.ConnectionTimeout as e:
                index_failed = len(actions)
                indexed = 0
                error_msg = f"OS Bulk Timeout ({e})"
                logging.error(error_msg)
                indexing_errors_details.append(error_msg)
                stats_counter['bulk_error_indexing'] += 1
                log_event("bulk_error_indexing", details={"error": error_msg}, status="ERROR")
                phase1_successful = False
            except os_exceptions.TransportError as e:
                index_failed = len(actions)
                indexed = 0
                error_msg = f"OS Bulk TransportError ({e})"
                logging.error(error_msg, exc_info=True)
                indexing_errors_details.append(error_msg[:200])
                stats_counter['bulk_error_indexing'] += 1
                log_event("bulk_error_indexing", details={"error": error_msg[:500]}, status="ERROR")
                phase1_successful = False
            except Exception as e:
                index_failed = len(actions)
                indexed = 0
                error_msg = f"OS Bulk Generic Error ({e})"
                logging.error(error_msg, exc_info=True)
                indexing_errors_details.append(error_msg[:200])
                stats_counter['bulk_error_indexing'] += 1
                log_event("bulk_error_indexing", details={"error": error_msg[:500]}, status="ERROR")
                phase1_successful = False
        else:
            logging.info("No new or updated public snippets found to index in Phase 1.")

        sync_results = {
            "phase": "Index", "type": sync_type, "processed": processed, "expired": expired,
            "skipped_rules": skipped_rules,
            "rule_based_analyzed": rule_based_analyzed,
            "indexed": indexed, "index_failed": index_failed,
            "duration_sec": round(time.time() - phase1_start_time, 2),
            "errors": indexing_errors_details,
            "stats": stats_counter
        }
        log_event("index_finished", details={k: v for k, v in sync_results.items() if k not in ['errors', 'phase', 'stats']}, status="INFO" if phase1_successful else "ERROR")

    except Exception as e:
        phase1_successful = False
        error_message = f"Critical error during Index Phase: {e}"
        logging.error(error_message, exc_info=True)
        log_event("critical_index_error", details={"error": str(e)[:500], "type": sync_type}, status="ERROR")
        sync_results = {"phase": "Index", "type": sync_type, "status": "critical_error", "error": str(e)[:500], "duration_sec": round(time.time() - phase1_start_time, 2), "stats": stats_counter}
        send_discord_notification(embeds=[{"title": f"❌ {sync_type} Failed Critically", "description": error_message[:1000]}], level="error")
        close_log_session()
        sys.exit(1)
    finally:
        close_log_session()

    # --- Phase 2: Purge ---
    phase2_successful = False
    purged_found, purged_deleted, purged_backup_failed, purged_backup_skipped = 0, 0, 0, 0
    try:
        purged_found, purged_deleted, purged_backup_failed, purged_backup_skipped = purge_deleted_snippets(stats_counter)
        purge_results = {
            "phase": "Purge",
            "found_deleted": purged_found,
            "purged": purged_deleted,
            "backup_failed_r2": purged_backup_failed,
            "backup_skipped_r2": purged_backup_skipped,
            "duration_sec": round(time.time() - (phase1_start_time + sync_results.get("duration_sec", 0)), 2)
        }
        phase2_successful = True

    except Exception as e:
        error_message = f"Critical error initiating Purge Phase: {e}"
        logging.error(error_message, exc_info=True)
        purge_results = {"phase": "Purge", "status": "critical_error", "error": str(e)[:500]}
        send_discord_notification(embeds=[{"title": f"❌ Purge Phase Failed Critically", "description": error_message[:1000]}], level="error")

    # --- Final Summary ---
    overall_duration = round(time.time() - overall_start_time, 2)
    final_status = "success" if phase1_successful and phase2_successful else "partial_failure" if phase1_successful or phase2_successful else "failed"

    summary_title_status = '✅' if final_status == 'success' else '⚠️'
    if purged_backup_failed > 0 and final_status == 'success':
        summary_title_status = '⚠️'
        final_status = 'partial_failure'
    
    summary_title = f"{summary_title_status} Sync & Purge Completed ({overall_duration}s)"
    summary_desc_parts = [f"**Index Phase ({sync_results.get('duration_sec', 'N/A')}s):**"]
    summary_desc_parts.append(f"  Type: {sync_results.get('type', 'N/A')}")
    summary_desc_parts.append(f"  Checked: {sync_results.get('processed', 'N/A')} | Expired: {sync_results.get('expired', 'N/A')} | Skipped (Rules): {sync_results.get('skipped_rules', 'N/A')}")
    summary_desc_parts.append(f"  Rule-based Analyzed: {sync_results.get('rule_based_analyzed', 'N/A')}")
    summary_desc_parts.append(f"  Indexed: {sync_results.get('indexed', 'N/A')} | Index Fail: {sync_results.get('index_failed', 'N/A')}")
    if sync_results.get('status') == 'critical_error':
        summary_desc_parts.append(f"  **Status: CRITICAL ERROR**")

    summary_desc_parts.append(f"\n**Purge Phase ({purge_results.get('duration_sec', 'N/A')}s):**")
    if purge_results.get('status') == 'critical_error':
        summary_desc_parts.append(f"  **Status: CRITICAL ERROR**")
    else:
        summary_desc_parts.append(f"  Found Deleted: {purge_results.get('found_deleted', 'N/A')} | Purged: {purge_results.get('purged', 'N/A')}")
        summary_desc_parts.append(f"  R2 Backup Fail: {purge_results.get('backup_failed_r2', 'N/A')} | R2 Backup Skip: {purge_results.get('backup_skipped_r2', 'N/A')}")

    summary_description = "\n".join(summary_desc_parts)
    summary_embed = {"title": summary_title, "description": summary_description, "fields": []}

    index_errors_sample = sync_results.get('errors')
    if index_errors_sample:
        summary_embed["fields"].append({"name": "Indexing Errors (sample)", "value": "\n".join(index_errors_sample)})

    if stats_counter:
        top_3_errors = stats_counter.most_common(3)
        error_summary_str = "\n".join([f"- {event}: {count} lần" for event, count in top_3_errors])
        if error_summary_str:
            summary_embed["fields"].append({"name": "Top 3 Warnings/Errors", "value": error_summary_str})

    discord_level = "success" if final_status == "success" else ("error" if final_status == "failed" else "warning")
    send_discord_notification(embeds=[summary_embed], level=discord_level)

    final_message = f"Worker finished. Rule-based: {sync_results.get('rule_based_analyzed', 'N/A')}. Index: {sync_results.get('indexed', 'N/A')} indexed, {sync_results.get('index_failed', 'N/A')} failed. Purge: {purge_results.get('purged', 'N/A')} purged (R2 Fail: {purged_backup_failed}). Total time: {overall_duration}s."
    
    logging.info(final_message)
    if final_status != "success":
        logging.warning(f"Worker finished with status: {final_status}")
        if not phase1_successful or not phase2_successful:
            logging.error("One or more phases failed. Exiting with error.")
            sys.exit(1)


# --- Khối thực thi chính ---
if __name__ == "__main__":
    logging.info("Starting worker script...")
    try:
        run_sync()
        logging.info("Worker run completed successfully.")
        sys.exit(0) 
    except SystemExit as e:
        # Sửa lỗi logic: Ghi log warning và raise lại lỗi SystemExit
        # để workflow biết script đã thoát với mã lỗi (nếu e.code != 0)
        logging.warning(f"Worker process exited with status {e.code}.")
        raise 
    except Exception as e:
        logging.critical(f"Worker run failed with unhandled exception: {e}", exc_info=True)
        try:
            send_discord_notification(embeds=[{"title": "❌ Worker Run Failed Critically", "description": f"Unhandled Exception: {e}"}], level="error")
        except Exception as e_discord:
             logging.error(f"Failed to send final Discord error notification: {e_discord}")
        sys.exit(1)
