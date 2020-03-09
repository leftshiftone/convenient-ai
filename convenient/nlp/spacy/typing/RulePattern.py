import dataclasses
from dataclasses import dataclass


@dataclass
class Pattern:
    id: str
    label: str
    pattern: str

    @property
    def asdict(self):
        return dataclasses.asdict(self)
