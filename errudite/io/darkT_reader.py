from typing import Dict, Any

import logging
import json

import pandas as pd
from tqdm import tqdm

from overrides import overrides

from .dataset_reader import DatasetReader
from ..utils import ConfigurationError, normalize_file_path, accuracy_score
from ..targets.instance import Instance
from ..targets.relation_extraction import BinaryRelation
from ..targets.label import Label, PredefinedLabel
from ..processor import SpacyAnnotator, spacy_annotator
import spacy
en_nlp = spacy.load('en_core_web_sm')
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("darkT")
class darkTReader(DatasetReader):
    """
    This loads the data from TACRED Corpus:
    https://nlp.stanford.edu/projects/tacred/

    * default evaluation metric: micro-averaged f1
    * target entries in an instance: text, predictions, groundtruth.

    This can be queried via:

    .. code-block:: python

        from errudite.io import DatasetReader
        DatasetReader.by_name("darkT")
    """
    def __init__(self, cache_folder_path: str = None) -> None:
        super().__init__(cache_folder_path)
        Label.set_task_evaluator(accuracy_score, 'accuracy')
        self.spacy_annotator = SpacyAnnotator(pre_tokenized=True,
                                              disable=["tagger", "parser", "ner", "textcat"])

    @overrides
    def _read(self, file_path: str, lazy: bool, sample_size: int):
        instances = []
        texts = []
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(normalize_file_path(file_path), "r") as data_file:
            logger.info("Reading DT instances from text dataset at: %s", file_path)
            data = data_file.readlines()

            for idx, example in enumerate(data, start=1):
                if lazy:
                    texts.append(example[3])
                else:
                    instance = self._text_to_instance(example)
                    if instance is not None:
                        instances.append(instance)
                    if sample_size and idx >= sample_size:
                        break
        if lazy:
            return {"text": texts}

        return instances

    @overrides
    def _text_to_instance(self, example: Dict[str, Any]) -> Instance:  # type: ignore
        doc = en_nlp(example[3])
        tokens = [token.text for token in doc]

        # TACRED entity span indices are inclusive, we need the end index to be exclusive
        id_ = example.get("id")

        # target
        text = BinaryRelation(qid=id_,
                              text=tokens,
                              pos=[token.pos_ for token in doc],
                              dep=[token.dep_ for token in doc],
                              vid=0,
                              annotator=self.spacy_annotator)
        # label
        groundtruth = PredefinedLabel(model="groundtruth",
                                      qid=id_,
                                      text=example[1],
                                      vid=0)

        return self.create_instance(id_,
                                    text=text,
                                    groundtruth=groundtruth)
