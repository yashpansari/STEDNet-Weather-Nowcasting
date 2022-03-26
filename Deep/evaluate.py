import numpy as np
import torch
from climatehack import BaseEvaluator
from torchvision.transforms import CenterCrop
from model import ConvLSTMModel

import torch
import torch.nn as nn

class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.
        In this case, it loads the trained model (in evaluation mode)."""

        self.model = ConvLSTMModel(hid_chan=24, in_chan=1, num_layers=10, use_last = False)
        self.model.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        
        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        min1 = min(np.min(data), 0)
        max1 = max(np.max(data), 1023)
        data = (data - min1)/(max1 - min1)
        with torch.no_grad():
            prediction = (
                CenterCrop((64, 64))(self.model.ultimate_pred(torch.from_numpy(data).view(1, 12, 128, 128)))
                .view(24, 64, 64).cpu()
                .detach()
                .numpy()
            )
            prediction = prediction*(max1 - min1) + min1    

            assert prediction.shape == (24, 64, 64)

            return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()
    
if __name__ == "__main__":
    main()