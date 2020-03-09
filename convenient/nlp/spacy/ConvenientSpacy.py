from dataclasses import dataclass
from typing import List, Union, Pattern

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc

from convenient.nlp.__common__ import Text
from convenient.nlp.__common__.io import Json
from convenient.nlp.spacy.typing import RulePattern


@dataclass
class ConvenientSpacy:
    """
    Convenient class implementation for the spacy library.
    """
    nlp: Language
    pipeline_names: List[str] = List[str]()

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
    Pipes the given text through the spacy pipeline
    """
    def pipe(self, texts: Union[List[Text], Text]) -> List[Doc]:
        if isinstance(texts, Text):
            texts = [texts]

        texts = (List[Text])(texts)

        return self.nlp.pipe(texts)

    """
    Appends an EntityRuler to the spacy pipeline
    """
    def add_ruler(self, patterns: Union[List[RulePattern], RulePattern], before: str = "ner") -> 'ConvenientSpacy':
        if isinstance(patterns, Pattern):
            patterns = [patterns]

        patterns = (List[RulePattern])(patterns)

        ruler = EntityRuler(self.nlp)
        [ruler.add_patterns(pattern.asdict) for pattern in patterns]
        self.nlp.add_pipe(ruler, before=before)

        return self

    """
    Appends a custom component to the spacy pipeline
    """
    def add_component(self, component) -> 'ConvenientSpacy':
        Language.factories[component.name] = lambda nlp, **cfg: component
        self.nlp.add_pipe(component, first=True)

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
