import time
import logging
import json
import uuid
import datetime
from collections import Counter
from google.cloud.firestore_v1.base_query import FieldFilter
from botocore.exceptions import ClientError

from config import db, s3_client, r2_bucket_name, r2_recovery_prefix
from d1_logger import open_log_session, log_event, close_log_session
from discord_notifier import send_discord_notification

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

