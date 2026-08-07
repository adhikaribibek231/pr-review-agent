from dataclasses import dataclass
from email import message
from turtle import st

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
