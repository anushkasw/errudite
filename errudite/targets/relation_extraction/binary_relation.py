from typing import List, Dict, Tuple

import numpy
from spacy.symbols import ENT_TYPE, POS, TAG, DEP, HEAD
from spacy.tokens import Doc
from ..target import Target


class BinaryRelation(Target):
    """initialize a binary relation instance.

        Parameters
        ----------
        qid : str
            The id of the instance.
        text : str
            The raw text will be processed with SpaCy.
        vid : int, optional
            The version, by default 0. When an instance/a target is rewritten, the version 
            will automatically grow.
        annotator : SpacyAnnotator, optional
            The annotator, by default None. If None, use the default annotator.
        metas : Dict[str, any], optional
            Additional metas associated with a target, in the format of {key: value}, by default {}
    """

    def __init__(self,
                 qid: str,
                 text: List[str],
                 head: Tuple[int, int],
                 tail: Tuple[int, int],
                 head_type: str = None,
                 tail_type: str = None,
                 ner: List[str] = None,
                 pos: List[str] = None,
                 dep: List[str] = None,
                 dep_heads: List[str] = None,
                 vid: int = 0,
                 annotator=None,
                 metas: Dict[str, any] = None) -> None:
        Target.__init__(self, qid=qid, text=None, vid=vid, annotator=annotator, metas=metas)

        self.head = head
        self.tail = tail
        self.head_type = head_type
        self.tail_type = tail_type

        vocab = annotator.model.vocab

        words = text
        spaces = [True] * len(words)
        tags = [vocab.strings.add(p) for p in pos]
        # deps = [vocab.strings.add(d) for d in dep]
        # heads = dep_heads
        ent_types = [vocab.strings.add(e) for e in ner]
        # attrs = [ENT_TYPE, POS, TAG, DEP, HEAD]
        attrs = [ENT_TYPE, TAG]
        # array = numpy.array(list(zip(ent_types, pos, tags, deps, heads)), dtype="uint64")
        array = numpy.array(list(zip(ent_types, tags)), dtype="uint64")
        doc = Doc(vocab, words=words, spaces=spaces).from_array(attrs, array)
        if any(pos) and any(tags):
            doc.is_tagged = True
        # if any(deps):
        #     doc.is_parsed = True
        self.doc = doc
