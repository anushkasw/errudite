from typing import List, Dict, Tuple, Optional
from ..predictor import Predictor
from ...utils.evaluator import accuracy_score
from ...targets.label import Label, PredefinedLabel, Target


@Predictor.register("re_task_class")
class PredictorRE(Predictor):
    """
    Predictor wrapper for relation extraction tasks.
    perform metrics: ``['accuracy', 'confidence']``

    This can be queried via:

    .. code-block:: python

        from errudite.predictors import Predictor
        Predictor.by_name("relation_extration")
    """
    def __init__(self,
                 name: str,
                 description: str,
                 model: any):
        perform_metrics = ["accuracy", "confidence"]
        Predictor.__init__(self, name, description, model, perform_metrics)
        Label.set_task_evaluator(accuracy_score, task_primary_metric='accuracy')

    def predict(self,
                text: str,
                head: Tuple[int, int],
                tail: Tuple[int, int],
                id_: Optional[str] = None,
                ner: List[str] = None,
                pos: List[str] = None,
                dep: List[str] = None,
                dep_heads: List[int] = None) -> Dict[str, float]:
        """
        run the prediction.

        Raises
        ------
        NotImplementedError
           Should be implemented in subclasses.
        """
        raise NotImplementedError

    @classmethod
    def model_predict(cls,
                      predictor: "PredictorRE",
                      text: Target,
                      groundtruth: Label) -> Label:
        """
        Define a class method that takes Target inputs, run model predictions, 
        and wrap the output prediction into Labels.

        Parameters
        ----------
        predictor : Predictor
            A predictor object, with the predict method implemented.
        binary_rel : Target
            The text target containing an entity pair.
        groundtruth : Label
            A groundtruth, typed Label.

        Returns
        -------
        Label
            The predicted output, with performance saved.
        """
        if not predictor:
            return None

        head = (text.head[0], text.head[1] - 1)
        tail = (text.tail[0], text.tail[1] - 1)

        predicted = predictor.predict(text=text.get_text(),
                                      head=head,
                                      tail=tail,
                                      id_=text.qid,
                                      ner=[])
        if not predicted:
            return None

        relation = PredefinedLabel(model=predictor.name,
                                   qid=text.qid,
                                   text=predicted["text"],
                                   vid=max([text.vid, groundtruth.vid]))
        relation.compute_perform(groundtruths=groundtruth)
        relation.set_perform(confidence=predicted["confidence"])

        return relation
