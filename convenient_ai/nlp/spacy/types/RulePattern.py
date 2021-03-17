import dataclasses
from dataclasses import dataclass


@dataclass
class RulePattern:
    id: str
    label: str
    pattern: str

    @property
    def as_dict(self):
        return dataclasses.asdict(self)
