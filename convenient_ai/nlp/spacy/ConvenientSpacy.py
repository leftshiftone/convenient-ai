from dataclasses import dataclass, field
from typing import List, Union, Pattern

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Token

from convenient_ai.nlp.__common__.io import Json
from convenient_ai.nlp.spacy.types import RulePattern


@dataclass
class ConvenientSpacy:
    """
    Convenient class implementation for the spacy library.
    """
    nlp: Language = field(default_factory=Language)
    pipeline_names: List[str] = field(default_factory=list)

    """
    Returns a ConvenientSpacy instance with a blank model
    """

    @staticmethod
    def from_blank(lang: str) -> 'ConvenientSpacy':
        return ConvenientSpacy(spacy.blank(lang))

    """
    Returns a ConvenientSpacy instance with a predefined model
    """

    @staticmethod
    def from_model(lang: str, **overrides) -> 'ConvenientSpacy':
        return ConvenientSpacy(spacy.load(lang, **overrides))

    """
    Returns a ConvenientSpacy instance with a language instance
    """

    @staticmethod
    def from_language(nlp: Language) -> 'ConvenientSpacy':
        return ConvenientSpacy(nlp)

    """
    Pipes the given text through the spacy pipeline
    """

    def pipe(self, texts: Union[List[str], str]) -> List[Doc]:
        if isinstance(texts, str):
            texts = [texts]

        return self.nlp.pipe(texts)

    """
    Appends an EntityRuler to the spacy pipeline
    """

    def add_ruler(self, patterns: Union[List[RulePattern], RulePattern], before: str = "ner") -> 'ConvenientSpacy':
        if isinstance(patterns, Pattern):
            patterns = [patterns]

        patterns = (List[Pattern])(patterns)

        ruler = EntityRuler(self.nlp)
        [ruler.add_patterns(pattern.asdict) for pattern in patterns]
        self.nlp.add_pipe(ruler, before=before)

        return self

    """
    Appends a custom component to the spacy pipeline
    """

    def add_component(self, component) -> 'ConvenientSpacy':
        Language.factories[component.name] = lambda nlp, **cfg: component
        self.nlp.add_pipe(component, before="ner")

        return self

    """
    Creates the spacy pipeline
    """

    def create_pipeline(self, pipeline_names: List[str]) -> 'ConvenientSpacy':
        [self.pipeline_names.append(pipeline_name) for pipeline_name in pipeline_names]
        [self.nlp.add_pipe(self.nlp.create_pipe(pipeline_name)) for pipeline_name in pipeline_names]

        return self

    """
    Stores the spacy model at the given path
    Creates a config.json file which contains all relevant information to restore the model
    """

    def store(self, path: str) -> 'ConvenientSpacy':
        self.nlp.to_disk(path)
        Json.write(path, "config", {'lang': self.nlp.lang, 'pipeline': self.pipeline_names})

        return self

    """
    Restores the spacy model
    """

    @staticmethod
    def restore(path: str) -> 'ConvenientSpacy':
        config = Json.read(path, "config")
        model = ConvenientSpacy.from_blank(config['lang'])
        model.create_pipeline(config['pipeline'])
        model.nlp.from_disk(path)

        return model

    """
    Fixes the character cases of the first word as well as noun and proper noun words.
    Excludes part of speech words which are given by the 'excludes' attribute.
    At default 'PUNCT' and 'ADV' pos tags will be removed.
    """

    def preprocess_text(self, text: str, pos_excludes: [str] = None, tag_excludes: [str] = None) -> str:
        if pos_excludes is None:
            pos_excludes = ["PUNCT", "ADV", "SPACE"]
        if tag_excludes is None:
            tag_excludes = []

        segments = []

        tokens: [Token] = self.nlp(text.lower())
        for token in tokens:
            if token.is_sent_start:
                segments.append(token.text.capitalize())
            elif token.pos_ == "NOUN":
                segments.append(token.text.capitalize())
            elif token.pos_ == "PROPN":
                segments.append(token.text.capitalize())
            elif token.pos_ not in pos_excludes and token.tag_ not in tag_excludes:
                segments.append(token.text)

        return " ".join(segments)

    def get_pos_count(self, text: str, pos: str) -> int:
        tokens: [Token] = self.nlp(text)

        amount = 0
        for token in tokens:
            if token.pos_ == pos and not token.is_stop:
                amount += 1
        return amount
