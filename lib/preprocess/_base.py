
class Preprocessor:
    name: str

    def __init__(self, name: str):
        self.name = name
        print("Initializing Preprocessor", self.name)

    def transform(self, *args, **kwargs):
        pass
