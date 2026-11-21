from dataclasses import dataclass, field


@dataclass(slots=True)
class Choice:
    text: str
    consequence: str
    box: list[int] = field(default_factory=lambda: [])

@dataclass(slots=True)
class EventInfo:
    question: str
    choices: list[Choice]
