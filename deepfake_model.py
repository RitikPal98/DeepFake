import torch
import torch.nn as nn
import timm

class DeepFakeClassifier(nn.Module):
    def __init__(self, encoder='tf_efficientnet_b7_ns'):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.encoder.num_features, 1)

    def forward(self, x):
        x = self.encoder.forward_features(x)
        x = self.avg_pool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class EnsembleModel(nn.Module):
    def __init__(self, model_paths):
        super().__init__()
        self.models = nn.ModuleList([DeepFakeClassifier() for _ in model_paths])
        for model, path in zip(self.models, model_paths):
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Remove 'module.' prefix if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.cat(outputs, dim=1), dim=1, keepdim=True)

def load_ensemble_model(weight_paths):
    model = EnsembleModel(weight_paths)
    model.eval()
    return model