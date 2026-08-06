import time
import logging
import sys
import datetime
from collections import Counter

from config import db, os_client
import d1_logger
from d1_logger import open_log_session, log_event, close_log_session
from discord_notifier import send_discord_notification
from indexer import get_last_processed_timestamp, run_indexing
from purge import purge_deleted_snippets

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
    d1_logger.session_id = None
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

    phase1_successful = True

    try:
        current_run_time = datetime.datetime.now(datetime.timezone.utc)
        idx_res = run_indexing(stats_counter, last_processed_at_dt, current_run_time, sync_type)
        phase1_successful = idx_res["phase1_successful"]

        sync_results = {
            "phase": "Index",
            "type": sync_type,
            "processed": idx_res["processed"],
            "expired": idx_res["expired"],
            "skipped_rules": idx_res["skipped_rules"],
            "rule_based_analyzed": idx_res["rule_based_analyzed"],
            "indexed": idx_res["indexed"],
            "index_failed": idx_res["index_failed"],
            "duration_sec": round(time.time() - phase1_start_time, 2),
            "errors": idx_res["errors"],
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
        logging.warning(f"Worker process exited with status {e.code}.")
        raise 
    except Exception as e:
        logging.critical(f"Worker run failed with unhandled exception: {e}", exc_info=True)
        try:
            send_discord_notification(embeds=[{"title": "❌ Worker Run Failed Critically", "description": f"Unhandled Exception: {e}"}], level="error")
        except Exception as e_discord:
             logging.error(f"Failed to send final Discord error notification: {e_discord}")
        sys.exit(1)

