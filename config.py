import os
import logging
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from opensearchpy import OpenSearch
import boto3
from botocore.client import Config

# --- Logging Configuration ---
log_level_env = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_level = log_level_env if log_level_env else 'INFO'

# Set logging level for priority_rules module
logging.getLogger("priority_rules").setLevel(logging.DEBUG if log_level == 'DEBUG' else logging.INFO)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("opensearch").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING) 
logging.getLogger("botocore").setLevel(logging.WARNING) 
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s')

# Load .env 
load_dotenv()

# --- Initialize Firebase Admin SDK ---
db = None
try:
    if not firebase_admin._apps:
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
opensearch_scheme = os.getenv("OPENSEARCH_SCHEME", "https") 
opensearch_index = os.getenv("OPENSEARCH_INDEX", "snippets")

if not opensearch_host:
    logging.error("OPENSEARCH_HOST environment variable not set.")
else:
    auth = (opensearch_user, opensearch_password) if opensearch_user and opensearch_password else None
    try:
        # Omit verify_certs=True if the scheme is http, but using https is best.
        verify_certs_val = opensearch_scheme == "https"
        os_client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}], http_auth=auth,
            use_ssl=opensearch_scheme == "https", 
            verify_certs=verify_certs_val, # Only check certificates if using HTTPS
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

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logging.warning("OPENROUTER_API_KEY is missing. AI priority scoring will be disabled and fall back to rule-based.")

