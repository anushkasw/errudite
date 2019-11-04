from typing import Dict, Tuple, List, Optional

import numpy as np
from .predictor_re import PredictorRE
from ..predictor_allennlp import PredictorAllennlp
from ..predictor import Predictor


@Predictor.register("re")
class PredictorREAllenNLP(PredictorRE, PredictorAllennlp, Predictor):
    """
    This can be queried via:

    .. code-block:: python

        from errudite.predictors import Predictor
        Predictor.by_name("re")
    """
    def __init__(self,
                 name: str,
                 model_path: str = None,
                 description: str = "",
                 model_type: str = None,
                 label_namespace: str = "labels") -> None:
        PredictorAllennlp.__init__(self,
                                   name=name,
                                   model_path=model_path,
                                   description=description,
                                   model_type=model_type)
        PredictorRE.__init__(self, name, description, self.predictor)
        token_to_index = self.predictor._model.vocab.get_token_to_index_vocabulary(label_namespace)
        self.index_to_label = {i: l for l, i in token_to_index.items()}

    def predict(self,
                text: str,
                head: Tuple[int, int],
                tail: Tuple[int, int],
                id_: Optional[str] = None,
                ner: List[str] = None,
                pos: List[str] = None,
                dep: List[str] = None,
                dep_heads: List[int] = None) -> Dict[str, float]:
        predicted = self.predictor.predict_json(dict(text=text,
                                                     head=head,
                                                     tail=tail,
                                                     id_=id_,
                                                     ner=ner,
                                                     pos=pos,
                                                     dep=dep,
                                                     dep_heads=dep_heads))

        label_probs = predicted['class_probabilities']
        return {
            'confidence': max(label_probs),
            'text': self.index_to_label[np.argmax(label_probs)],
        }
