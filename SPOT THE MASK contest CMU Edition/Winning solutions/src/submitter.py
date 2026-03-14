import pandas as pd
import os


class Submitter:
    def __init__(self, output_dir: str = "submissions"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def save(self, filenames: list, probs: list, filename: str) -> str:
        df = pd.DataFrame({"id": filenames, "label": probs})
        df["label"] = df["label"].clip(1e-6, 1 - 1e-6)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        print(f"Submission saved → {path}")
        return path