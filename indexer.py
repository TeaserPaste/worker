import time
import logging
import datetime
import json
from collections import Counter
from google.cloud.firestore_v1.base_query import FieldFilter
from opensearchpy import helpers, exceptions as os_exceptions

from config import db, os_client, opensearch_index
from d1_logger import log_event
from priority_rules import calculate_priority

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


def run_indexing(stats_counter: Counter, last_processed_at_dt, current_run_time, sync_type):
    """
    Runs the indexing phase, querying Firestore and updating OpenSearch.
    """
    processed, indexed, index_failed, skipped_rules, expired = 0, 0, 0, 0, 0
    rule_based_analyzed = 0
    actions = []
    indexing_errors_details = []
    phase1_successful = True

    try:
        snippets_ref = db.collection('snippets')
        query = snippets_ref.where(filter=FieldFilter('visibility', '==', 'public'))

        if last_processed_at_dt:
            query = query.where(filter=FieldFilter('updatedAt', '>', last_processed_at_dt))
            query = query.where(filter=FieldFilter('updatedAt', '<=', current_run_time))
            logging.info(f"Querying Firestore for public snippets updated in window: ({last_processed_at_dt.isoformat()}, {current_run_time.isoformat()}]")
        else:
            query = query.where(filter=FieldFilter('updatedAt', '<=', current_run_time))
            logging.info(f"Querying Firestore for ALL public snippets up to {current_run_time.isoformat()}...")

        def stream_snippets():
            # 1. Fetch pending snippets from OpenSearch first
            pending_ids = []
            if os_client:
                try:
                    search_query = {
                        "size": 1000,
                        "query": {
                            "match": {
                                "ai_status": "pending"
                            }
                        },
                        "_source": False
                    }
                    res = os_client.search(index=opensearch_index, body=search_query, request_timeout=60)
                    hits = res.get('hits', {}).get('hits', [])
                    pending_ids = [hit['_id'] for hit in hits]
                    if pending_ids:
                        logging.info(f"Found {len(pending_ids)} pending snippets in OpenSearch for AI retry.")
                except os_exceptions.NotFoundError:
                    logging.info(f"OpenSearch index '{opensearch_index}' not found when searching for pending snippets.")
                except Exception as e:
                    logging.error(f"Error querying pending snippets from OpenSearch: {e}", exc_info=True)

            processed_ids = set()

            # 2. Stream standard snippets
            try:
                docs_stream = query.order_by('updatedAt', direction=firestore_query_asc()).stream()
                for doc in docs_stream:
                    sid = doc.id
                    processed_ids.add(sid)
                    yield sid, doc.to_dict(), False
            except Exception as e:
                logging.error(f"Error streaming from Firestore: {e}", exc_info=True)

            # 3. Stream pending snippets that weren't in standard stream
            remaining_pending_ids = [pid for pid in pending_ids if pid not in processed_ids]
            if remaining_pending_ids:
                logging.info(f"Processing {len(remaining_pending_ids)} remaining pending snippets from queue...")
                for pid in remaining_pending_ids:
                    try:
                        doc_ref = snippets_ref.document(pid)
                        doc_snap = doc_ref.get()
                        if doc_snap.exists:
                            snippet_data = doc_snap.to_dict()
                            if snippet_data.get('visibility') == 'public':
                                processed_ids.add(pid)
                                yield pid, snippet_data, True
                    except Exception as e:
                        logging.error(f"Error fetching pending snippet {pid} from Firestore: {e}", exc_info=True)

        for snippet_id, snippet_data, is_pending_retry in stream_snippets():
            processed += 1

            snippet_updated_at = snippet_data.get('updatedAt')
            if isinstance(snippet_updated_at, datetime.datetime):
                if snippet_updated_at.tzinfo is None:
                    snippet_updated_at = snippet_updated_at.replace(tzinfo=datetime.timezone.utc)
                if not is_pending_retry:
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
                priority_score, assessment_string, ai_status = calculate_priority(
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
                ai_status = None

            updated_fields = {
                'ai_priority': float(ai_priority),
                'ai_assessment': ai_assessment,
                'processed_at': current_run_time.isoformat(),
                'analysis_source': analysis_source,
                'ai_status': ai_status,
            }

            upsert_doc = snippet_data.copy()
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

    except Exception as e:
        phase1_successful = False
        raise e

    return {
        "processed": processed,
        "expired": expired,
        "skipped_rules": skipped_rules,
        "rule_based_analyzed": rule_based_analyzed,
        "indexed": indexed,
        "index_failed": index_failed,
        "errors": indexing_errors_details,
        "phase1_successful": phase1_successful
    }

def firestore_query_asc():
    """Gets the Firestore query ascending order constant."""
    from firebase_admin import firestore
    return firestore.Query.ASCENDING

