class SharedState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SharedState, cls).__new__(cls)
            cls._instance.citation_data = None
        return cls._instance

    def set_citation_data(self, data):
        self.citation_data = data

    def get_citation_data(self):
        return self.citation_data