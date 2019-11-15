from typing import Dict, Any

import json
import torch
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor as AllenPredictor
from .predictor import Predictor


class PredictorAllennlp(Predictor):
    def __init__(self,
                 name: str,
                 model_path: str = None,
                 description: str = "",
                 model_type: str = None,
                 overrides: Dict[str, Any] = None) -> None:
        """A class specifically created for wrapping the predictors from 
        Allennlp: https://allenai.github.io/allennlp-docs/api/allennlp.predictors.html

        Parameters
        ----------
        name : str
        The name of the predictor.
        model_path : str, optional
            A local model path if you are using local models, by default None.
            This and ``model_online_path`` cannot both be None.
        model_online_path : str, optional
            An online model path, by default None
        description : str, optional
            A sentence describing the predictor., by default ''
        model_type : str, optional
            The model type as used in Allennlp, by default None

        Returns
        -------
        None
        """
        overrides = overrides or {}
        model = None
        if torch.cuda.is_available():
            archive = load_archive(model_path, cuda_device=0, overrides=json.dumps(overrides))
        else:
            archive = load_archive(model_path, overrides=json.dumps(overrides))
        model = AllenPredictor.from_archive(archive, model_type)
        self.predictor = model
        Predictor.__init__(self, name, description, model, ['accuracy'])

    def _predict_json(self, **inputs) -> Dict[str, float]:
        predicted = self.predictor.predict_json(inputs)
        return predicted
