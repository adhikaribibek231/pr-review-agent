from dataclasses import dataclass

@dataclass(frozen=True)
class Finding:
    filename: str
    line:int
    severity:str
    category:str
    message:str

@dataclass(frozen=True)
class ChangedFile:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None

@dataclass(frozen=True)
class RetrievedContext:
    filename: str
    start_line:int
    end_line:int
    content: str
    score: float
