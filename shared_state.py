import threading


class SharedState:
    _instance = None
    _lock = threading.Lock()  # Add a lock for synchronization

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedState, cls).__new__(cls)
            cls._instance.citation_data = None
        return cls._instance

    def set_citation_data(self, data):
        with self._lock:  # Use lock to ensure thread-safe access
            self.citation_data = data

    def get_citation_data(self):
        with self._lock:  # Use lock to ensure thread-safe access
            return self.citation_data
