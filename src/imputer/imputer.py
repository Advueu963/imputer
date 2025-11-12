from abc import ABC, abstractmethod
from typing import Generator, Optional, Union
import re

class Imputer(ABC):
    def __init__(self, model=None, feature_group:  Optional[Union[dict[int, str], str]] = None) -> None:
        self.model = model
        self.feature_group = feature_group

    def expand_coalitions(self, coalitions):
        expanded_coalitions = coalitions.copy()
        if self.feature_group is None:
            return coalitions
        
        if isinstance(self.feature_group, str):
            raise NotImplementedError("String-based feature groups are not implemented yet.")
        else:
            for k, v in self.feature_group.items():
                index = k
                if isinstance(index, int):
                    index = [index]
                coalition_values = [coalitions[idx] for idx in index]
                parts = re.split(r'[,;:]', v)
                parts = [part.strip() for part in parts if part.strip()]
                if parts[-1] != ',':
                    parts.append([","])
                current_index = []
                for i, part in enumerate(parts):
                    if part in [',', ';', ':']:
                        continue
                    parts[i] = int(part)
                for i, part in enumerate(parts):
                    if part != ':':
                        continue
                    prev_part = parts.pop(i - 1) if i > 0 else None
                    next_part = parts.pop(i + 1) if i + 1 < len(parts) else None
                    if prev_part is None or next_part is None or not (isinstance(prev_part, int) and isinstance(next_part, int)):
                        raise ValueError("Invalid feature group format around ':'.")
                    combined = [x for x in range(prev_part, next_part + 1)]
                    parts[i] = combined

                for i, part in enumerate(parts):
                    if part != ';':
                        continue
                    prev_part = parts.pop(i - 1) if i > 0 else None
                    next_part = parts.pop(i + 1) if i + 1 < len(parts) else None
                    if prev_part is None or next_part is None:
                        raise ValueError("Invalid feature group format around ';'.")
                    combined = []
                    for prev in prev_part:
                        if isinstance(prev, int):
                            prev = [prev]
                        for nex in next_part:
                            if isinstance(nex, int):
                                nex = [nex]
                            combined.append(prev.append(nex)) # type: ignore
                    parts[i] = combined
                for i, part in enumerate(parts):
                    if part == ',':
                        continue
                    for idx in part:
                        expanded_coalitions.insert(idx, coalition_values)
                    
        return coalitions

    def __call__(self, data, coalitions):
        coalitions_processed = self.expand_coalitions(coalitions)
        imputation_generator = self.impute(data, coalitions_processed)
        outputs = []
        for i, imputed_data in enumerate(imputation_generator):
            print(f"[Imputer __call__]: Imputed data for coalition {i}: {imputed_data}")
            predictions = self.predict(imputed_data)
            outputs.append(self.postprocess(predictions))
        return outputs

    @abstractmethod
    def impute(self, data, coalitions) -> Generator:
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, imputed_data):
        if self.model:
            outputs_pred = self.model.predict(imputed_data) # TODO: adapt to model interface
            return outputs_pred
        else:
            raise ValueError("Model must be provided for prediction")

    def postprocess(self, predictions):
        return predictions #TODO: placeholder for postprocessing
