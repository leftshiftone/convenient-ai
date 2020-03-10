import unittest
from convenient.nlp.spacy import ConvenientSpacy
from convenient.nlp.spacy.custom import RegExMatcher, RegExConfig


class ConvenientSpacyTest(unittest.TestCase):

    def test_basic_model(self):
        model = ConvenientSpacy.from_model("de_core_news_sm")
        docs = list(model.pipe(["AI aus Österreich."]))

        for doc in docs:
            self.assertEqual(len(doc.ents), 1)

        self.assertEqual(model.nlp.meta["lang"], "de")
        self.assertEqual(len(docs), 1)

    def test_custom_component(self):
        docs = ConvenientSpacy.from_model("de_core_news_sm", disable=["tagger", "parser"]) \
            .add_component(RegExMatcher(name="regExMatcher",
                                        config=RegExConfig(
                                            pattern="[0-9]*",
                                            label="NUM",
                                            minimum_length=2,
                                            maximum_length=5))) \
            .pipe("Ich bin 100 Jahre alt.")

        ents = list(docs)[0].ents

        self.assertEqual(len(ents), 1)

        ent = list(ents)[0]
        self.assertEqual(ent.label_, "NUM")
        self.assertEqual(ent.text, "100")