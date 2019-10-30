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
from ..processor import SpacyAnnotator


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def normalize_glove(token):
    mapping = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
    }
    return mapping.get(token, token)


@DatasetReader.register("tacred")
class TACREDReader(DatasetReader):
    """
    This loads the data from TACRED Corpus:
    https://nlp.stanford.edu/projects/tacred/

    * default evaluation metric: micro-averaged f1
    * target entries in an instance: text, predictions, groundtruth.

    This can be queried via:
    
    .. code-block:: python

        from errudite.io import DatasetReader
        DatasetReader.by_name("tacred")
    """
    def __init__(self, cache_folder_path: str = None) -> None:
        super().__init__(cache_folder_path)
        Label.set_task_evaluator(accuracy_score, 'accuracy')
        self.spacy_annotator = SpacyAnnotator(pre_tokenized=True)
    
    @overrides
    def _read(self, file_path: str, lazy: bool, sample_size: int):
        instances = []
        texts = []
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(normalize_file_path(file_path), "r") as data_file:
            logger.info("Reading TACRED instances from json dataset at: %s", file_path)
            data = json.load(data_file)
            for idx, example in enumerate(data):
                if lazy:
                    texts.append(" ".join([normalize_glove(token) for token in example["token"]]))
                else:
                    instance = self._text_to_instance(example)
                    if instance is not None:
                        instances.append(instance)
                    if sample_size and idx > sample_size:
                        break
        if lazy:
            return {"text": texts}

        return instances

    @overrides
    def _text_to_instance(self, example: Dict[str, Any]) -> Instance:  # type: ignore
        tokens = [normalize_glove(token) for token in example["token"]]

        # TACRED entity span indices are inclusive, we need the end index to be exclusive
        head = (example["subj_start"], example["subj_end"] + 1)
        tail = (example["obj_start"], example["obj_end"] + 1)
        head_type = example["subj_type"]
        tail_type = example["obj_type"]
        relation = example["relation"]

        id_ = example.get("id")
        ner = example.get("stanford_ner")
        pos = example.get("stanford_pos")
        dep = example.get("stanford_deprel")
        dep_heads = example.get("stanford_head")

        # target
        text = BinaryRelation(qid=id_,
                              text=tokens,
                              head=head,
                              tail=tail,
                              vid=0,
                              annotator=self.spacy_annotator)
        # label
        groundtruth = PredefinedLabel(model="groundtruth",
                                      qid=id_,
                                      text=relation,
                                      vid=0)

        return self.create_instance(id_,
                                    text=text,
                                    groundtruth=groundtruth)
