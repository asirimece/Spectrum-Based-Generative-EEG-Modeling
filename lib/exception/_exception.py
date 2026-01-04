class NotLoadedError(Exception):

    def __init__(self, message: str = "File not yet loaded. Please first load() file before accessing raw."):
        super().__init__(message)
        pass


class DatasetLockedError(Exception):

    def __init__(self, message: str = "The dataset has already been transformed. Please actively reset the dataset "
                                      "and epochs before adding new raws."):
        super().__init__(message)
        pass
