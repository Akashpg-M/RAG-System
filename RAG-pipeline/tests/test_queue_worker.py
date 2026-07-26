from src.queue_worker import IngestionQueueManager


class Pipeline:
    def __init__(self):
        self.calls = []

    def ingest_document(self, file_path, chunker):
        self.calls.append((file_path, chunker))


def test_queue_completes_and_shuts_down_cleanly():
    pipeline = Pipeline()
    manager = IngestionQueueManager(pipeline, chunker=object())
    task_id = manager.submit_task("document.md")
    state = manager.wait_for_task(task_id, timeout=2)
    assert state["status"] == "COMPLETED"
    assert pipeline.calls[0][0] == "document.md"
    manager.shutdown()
    assert not manager.worker_thread.is_alive()

