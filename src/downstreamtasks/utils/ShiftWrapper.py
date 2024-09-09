import pytorch_lightning as pl
import torch

class ShiftWrapper(pl.LightningModule):
    """
    Symmetric cropping along position

        crop: number of positions to crop on either side
    """

    def __init__(self, model, n_shift):
        super().__init__()
        self.model = model
        self.n_shift = n_shift -1

    def forward(self, x):
        # print(f"Input dtype: {x.dtype}")
        
        predictions_l = []
        features_biological_l = []
        features_technical_l = []
        for i in range(self.n_shift):
            input_crop = x[:, :, i: -self.n_shift + i]
            # print(f"Shape of input_crop: {input_crop.shape}")
            predictions, features_all, features_biological, features_technical = self.model.forward(input_crop, return_features=True)
            # print(f"Output dtype: {predictions.dtype}")

            predictions_l.append(predictions)
            features_biological_l.append(features_biological)
            features_technical_l.append(features_technical)

        stack_mean = lambda li: torch.mean(torch.stack(li), dim=0)
        output = stack_mean(predictions_l), stack_mean(features_biological_l), stack_mean(features_technical_l)
        return output