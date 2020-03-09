import re
from dataclasses import dataclass

from spacy.tokens.span import Span


@dataclass
class RegExConfig:
    label: str
    pattern: str
    minimum_length: int
    maximum_length: int


class RegExMatcher(object):
    name = "regex_matcher"

    def __init__(self, config: RegExConfig):
        self.config = config

    def __call__(self, document):
        iterator = re.finditer(self.config.pattern, document.text)
        spans = [match.span() for match in iterator]

        if spans is []:
            return document

        [self.add_new_matching_span(document, span) for span in spans]

        return document

    def add_new_matching_span(self, document, span):
        token_end_index, token_start_index = self.calculate_span_indices(document, span)
        length = span[1] - span[0]

        if self.config.minimum_length is not None and self.config.minimum_length > length:
            return
        if self.config.maximum_length is not None and self.config.maximum_length < length:
            return

        spacy_span = Span(document, token_start_index, token_end_index, self.config.label)
        document.ents = document.ents + (spacy_span,)

    @classmethod
    def calculate_span_indices(cls, document, span):
        character_index = 0
        token_start_index = -1
        token_end_index = -1

        for i in range(len(document)):
            if character_index + len(document[i].text) > span[0] and token_start_index == -1:
                token_start_index = i

            if character_index < span[1]:
                token_end_index = i

            character_index += len(document[i].text)

        return cls.reduce_matching_indices_to_avoid_conflicting_spans(document, token_end_index, token_start_index)

    @classmethod
    def reduce_matching_indices_to_avoid_conflicting_spans(cls, document, token_end_index, token_start_index):
        for ent in document.ents:
            if ent.start <= token_start_index <= ent.end:
                token_start_index = ent.end + 1

            if ent.start <= token_end_index <= ent.end:
                token_end_index = ent.start - 1

        return token_end_index, token_start_index
