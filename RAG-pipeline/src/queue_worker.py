import queue
import threading
import logging
import uuid
import time
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger("BackgroundQueueWorker")

class TaskStatus(Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class IngestionQueueManager:
    """
    Manages background thread tasks and tracking states for document ingestion.
    Prevents heavy text extraction pipelines from blocking active user queries.
    """
    def __init__(self, ingestion_pipeline, chunker):
        self.task_queue = queue.Queue()
        self.pipeline = ingestion_pipeline
        self.chunker = chunker
        self.task_states: Dict[str, Dict[str, Any]] = {}
        self.state_lock = threading.Lock()
        self._stop_sentinel = object()
        
        # Spawn the continuous worker daemon thread
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Background ingestion thread pool initialized successfully.")

    def submit_task(self, file_path: str) -> str:
        """Enqueues an ingestion request and immediately returns a tracker ID."""
        task_id = str(uuid.uuid4())
        
        with self.state_lock:
            self.task_states[task_id] = {
                "status": TaskStatus.PENDING.value,
                "file_path": file_path,
                "error": None
            }
            
        self.task_queue.put({"task_id": task_id, "file_path": file_path})
        logger.info("legacy_task_registered", extra={"component": "legacy_queue", "outcome": "success"})
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Thread-safe status checking for active ingestion tasks."""
        with self.state_lock:
            state = self.task_states.get(task_id)
            return state.copy() if state else None

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while deadline is None or time.monotonic() < deadline:
            state = self.get_task_status(task_id)
            if state is None or state["status"] in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value):
                return state
            time.sleep(0.05)
        return self.get_task_status(task_id)

    def shutdown(self, timeout: float = 5.0):
        self.task_queue.put(self._stop_sentinel)
        self.worker_thread.join(timeout=timeout)

    def _update_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None):
        with self.state_lock:
            if task_id in self.task_states:
                self.task_states[task_id]["status"] = status.value
                if error:
                    self.task_states[task_id]["error"] = error

    def _worker_loop(self):
        """Infinite worker loop executing tasks sequentially in the background thread."""
        while True:
            task = None
            try:
                task = self.task_queue.get()
                if task is self._stop_sentinel:
                    return
                task_id = task["task_id"]
                file_path = task["file_path"]
                
                self._update_status(task_id, TaskStatus.PROCESSING)
                logger.info("legacy_task_processing", extra={"component": "legacy_queue"})
                
                # Execute the heavy synchronous pipeline end-to-end
                self.pipeline.ingest_document(file_path, self.chunker)
                
                self._update_status(task_id, TaskStatus.COMPLETED)
                logger.info("legacy_task_completed", extra={"component": "legacy_queue", "outcome": "success"})
                
            except Exception as error:
                task_id = task.get("task_id") if isinstance(task, dict) else None
                logger.error("legacy_task_failed", extra={
                    "component": "legacy_queue", "error_code": "unexpected", "outcome": "failure",
                })
                if task_id:
                    self._update_status(task_id, TaskStatus.FAILED, error=type(error).__name__)
            finally:
                if task is not None:
                    self.task_queue.task_done()
