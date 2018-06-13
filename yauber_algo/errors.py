

class YaUberError(Exception):
    pass


class YaUberAlgoArgumentError(YaUberError):
    pass


class YaUberAlgoDtypeNotSupportedError(YaUberError):
    pass


class YaUberAlgoInternalError(YaUberError):
    pass


class YaUberAlgoFutureReferenceError(YaUberError):
    pass


class YaUberAlgoWindowConsistencyError(YaUberError):
    pass


class YaUberSanityCheckError(YaUberError):
    pass


class YaUberNotSupportedError(YaUberError):
    pass


class YaUberFeatureNotFoundError(YaUberError):
    pass


class YaUberModuleNotFoundError(YaUberError):
    pass


class YaUberFeatureBadSettings(YaUberError):
    pass


class YaUberModuleInitializationError(YaUberError):
    pass


class YaUberQualityCheckError(YaUberError):
    pass
