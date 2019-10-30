from typing import Dict, Tuple

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
                 text: str,
                 head: Tuple[int, int],
                 tail: Tuple[int, int],
                 vid: int = 0,
                 annotator=None,
                 metas: Dict[str, any] = None) -> None:
        Target.__init__(self, qid, text, vid, annotator=annotator, metas=metas)

        self.head = head
        self.tail = tail
