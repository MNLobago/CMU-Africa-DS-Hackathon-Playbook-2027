import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


class Predictor:
    TTA_TRANSFORMS = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]

    def __init__(self, model: nn.Module, device: torch.device, model_path: str):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        self.model = model.to(device)
        self.device = device

    def predict(self, loader: DataLoader, tta: bool = False) -> tuple[list, list]:
        if tta:
            return self._predict_tta(loader)
        return self._predict_standard(loader)

    def _predict_standard(self, loader: DataLoader) -> tuple[list, list]:
        filenames, probs = [], []
        with torch.no_grad():
            for imgs, _, fnames in loader:
                imgs = imgs.to(self.device)
                logits = self.model(imgs).squeeze(1)
                p = torch.sigmoid(logits).cpu().tolist()
                probs.extend(p if isinstance(p, list) else [p])
                filenames.extend(list(fnames))  # force list from tuple
        return filenames, probs

    def _predict_tta(self, loader: DataLoader) -> tuple[list, list]:
        """Average predictions across multiple augmented views of each image."""
        from data_loader import MaskDataset
        import numpy as np

        # Rebuild dataset from loader to apply different transforms
        base_dataset = loader.dataset
        all_probs = []

        for transform in self.TTA_TRANSFORMS:
            tta_dataset = MaskDataset(
                df=base_dataset.df,
                images_dir=base_dataset.images_dir,
                transform=transform
            )
            tta_loader = torch.utils.data.DataLoader(
                tta_dataset, batch_size=loader.batch_size,
                shuffle=False, num_workers=2, pin_memory=True
            )
            _, probs = self._predict_standard(tta_loader)
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0).tolist()
        images = [base_dataset.df.iloc[i]["image"] for i in range(len(base_dataset))]
        return images, avg_probs
    
class EnsemblePredictor:
    def __init__(self, predictors: list[Predictor], weights: list[float] = None):
        """
        predictors: list of loaded Predictor instances
        weights: how much to trust each model. None = equal weight.
        """
        self.predictors = predictors
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]  # normalize

    def predict(self, loader: DataLoader, tta: bool = False) -> tuple[list, list]:
        import numpy as np
        all_probs = []
        filenames = None

        for predictor, weight in zip(self.predictors, self.weights):
            fnames, probs = predictor.predict(loader, tta=tta)
            if filenames is None:
                filenames = fnames
            weighted = [p * weight for p in probs]
            all_probs.append(weighted)

        final_probs = np.sum(all_probs, axis=0).tolist()
        return filenames, final_probs