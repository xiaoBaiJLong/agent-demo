class AppError(Exception):
    def __init__(self, error_code: str, message: str, detail: str | None = None):
        self.error_code = error_code
        self.message = message
        self.detail = detail
        super().__init__(message)


class InvalidInputError(AppError):
    pass


class LLMError(AppError):
    pass


class RagError(AppError):
    pass


class ToolError(AppError):
    pass