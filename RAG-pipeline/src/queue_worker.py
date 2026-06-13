import queue
import threading
import logging
import uuid
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
        logger.info(f"Task {task_id} successfully registered for file: {file_path}")
        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Thread-safe status checking for active ingestion tasks."""
        with self.state_lock:
            return self.task_states.get(task_id)

    def _update_status(self, task_id: str, status: TaskStatus, error: Optional[str] = None):
        with self.state_lock:
            if task_id in self.task_states:
                self.task_states[task_id]["status"] = status.value
                if error:
                    self.task_states[task_id]["error"] = error

    def _worker_loop(self):
        """Infinite worker loop executing tasks sequentially in the background thread."""
        while True:
            try:
                task = self.task_queue.get()
                task_id = task["task_id"]
                file_path = task["file_path"]
                
                self._update_status(task_id, TaskStatus.PROCESSING)
                logger.info(f"Worker thread processing active task: {task_id}")
                
                # Execute the heavy synchronous pipeline end-to-end
                self.pipeline.ingest_document(file_path, self.chunker)
                
                self._update_status(task_id, TaskStatus.COMPLETED)
                logger.info(f"Task {task_id} processed successfully.")
                
            except Exception as e:
                logger.error(f"Task processing failure on ID {task_id}: {str(e)}")
                self._update_status(task_id, TaskStatus.FAILED, error=str(e))
            finally:
                self.task_queue.task_done()