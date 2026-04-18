from enum import Enum


class ModelType(Enum):
    RF = "rf"
    LOGREG = "logreg"


class DataKind(Enum):
    RAW = "raw"
    TRAIN = "train"
    TEST = "test"


class FraudDetectionError(Exception):
    def __init__(self, message: str, detail: str | None = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class DataNotFoundError(FraudDetectionError):
    pass


class ValidationError(FraudDetectionError):
    pass


class ModelNotLoadedError(FraudDetectionError):
    pass
