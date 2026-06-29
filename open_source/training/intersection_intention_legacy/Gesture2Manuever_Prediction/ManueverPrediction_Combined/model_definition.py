import torch
import torch.nn as nn


WINDOW_SIZE = 30
CHUNK_SIZE = 512  # Number of windows to process at once (memory-speed tradeoff)


def _extract_windows(skeleton_x, window_size=WINDOW_SIZE):
    """Extract sliding windows from skeleton sequences using unfold.

    Args:
        skeleton_x: [batch, seq_len, features]
    Returns:
        windows: [batch, num_windows, window_size, features]
    """
    # unfold returns [batch, num_windows, features, window_size]
    windows = skeleton_x.unfold(1, window_size, 1)
    # permute to [batch, num_windows, window_size, features]
    return windows.permute(0, 1, 3, 2).contiguous()


def _process_in_chunks(model, flat_windows, return_hidden=False, chunk_size=CHUNK_SIZE):
    """Process flattened windows through a model in memory-managed chunks.

    Args:
        model: gesture classifier model
        flat_windows: [total_windows, window_size, features]
        return_hidden: whether to return hidden states (pre-FC features)
        chunk_size: number of windows per chunk
    Returns:
        outputs: [total_windows, output_dim]
    """
    outputs = []
    for start in range(0, flat_windows.shape[0], chunk_size):
        chunk = flat_windows[start:start + chunk_size]
        if return_hidden:
            out = model(chunk, return_hidden=True)
        else:
            out = model(chunk)
        outputs.append(out)
    return torch.cat(outputs, dim=0)


class CombinedModel_HiddenWeighting(nn.Module):
    """Combined model using hidden state features from both gesture classifiers.

    Passes skeleton windows through frozen gesture models, extracts hidden states
    (pre-FC features), concatenates them, and feeds to LSTM for maneuver prediction.
    """

    def __init__(self, gesture_model_explicit, gesture_model_implicit,
                 label_encoder_explicit, label_encoder_implicit,
                 hidden_size, output_size, num_layers=2, dropout=0.5):
        super(CombinedModel_HiddenWeighting, self).__init__()
        self.gesture_model_explicit = gesture_model_explicit
        self.gesture_model_implicit = gesture_model_implicit
        self.label_encoder_explicit = label_encoder_explicit
        self.label_encoder_implicit = label_encoder_implicit
        self.hidden_size = hidden_size

        feature_dim = gesture_model_explicit.hidden_size * 2 + gesture_model_implicit.hidden_size * 2
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, skeleton_x):
        device = next(self.parameters()).device
        skeleton_x = skeleton_x.to(device)

        # Extract all sliding windows at once using unfold (replaces sequential Python loop)
        windows = _extract_windows(skeleton_x)  # [batch, num_windows, 30, 66]
        batch_size, num_windows = windows.shape[0], windows.shape[1]
        flat_windows = windows.reshape(batch_size * num_windows, WINDOW_SIZE, -1)

        # Process all windows through frozen gesture models in chunks
        with torch.no_grad():
            explicit_features = _process_in_chunks(
                self.gesture_model_explicit, flat_windows, return_hidden=True)
            implicit_features = _process_in_chunks(
                self.gesture_model_implicit, flat_windows, return_hidden=True)

        # Concatenate explicit + implicit hidden features
        combined = torch.cat((explicit_features, implicit_features), dim=1)
        gesture_features = combined.reshape(batch_size, num_windows, -1)  # [batch, num_windows, feature_dim]

        # LSTM forward pass on gesture feature sequence
        lstm_out, _ = self.lstm(gesture_features)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class CombinedModel_HiddenWeighting_Explicit(nn.Module):
    """Combined model using hidden state features from explicit gesture classifier only."""

    def __init__(self, gesture_model_explicit, gesture_model_implicit,
                 label_encoder_explicit, label_encoder_implicit,
                 hidden_size, output_size, num_layers=2, dropout=0.5):
        super(CombinedModel_HiddenWeighting_Explicit, self).__init__()
        self.gesture_model_explicit = gesture_model_explicit
        self.gesture_model_implicit = gesture_model_implicit
        self.label_encoder_explicit = label_encoder_explicit
        self.label_encoder_implicit = label_encoder_implicit
        self.hidden_size = hidden_size

        feature_dim = gesture_model_explicit.hidden_size * 2
        self.lstm = nn.LSTM(input_size=feature_dim,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, skeleton_x):
        device = next(self.parameters()).device
        skeleton_x = skeleton_x.to(device)

        windows = _extract_windows(skeleton_x)
        batch_size, num_windows = windows.shape[0], windows.shape[1]
        flat_windows = windows.reshape(batch_size * num_windows, WINDOW_SIZE, -1)

        with torch.no_grad():
            explicit_features = _process_in_chunks(
                self.gesture_model_explicit, flat_windows, return_hidden=True)

        gesture_features = explicit_features.reshape(batch_size, num_windows, -1)

        lstm_out, _ = self.lstm(gesture_features)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class CombinedModel(nn.Module):
    """Combined model mapping gesture predictions to scalar weights.

    Explicit gestures: Left_Hand=-1, Right_Hand=+1, Neutral=0
    Implicit gestures: Left_Look=-1, Left_Overshoulder=-2, Right_Look=+1, Neutral=0
    Uses precomputed weight lookup tables instead of Python string comparisons.
    """

    def __init__(self, gesture_model_explicit, gesture_model_implicit,
                 label_encoder_explicit, label_encoder_implicit,
                 hidden_size, output_size, num_layers=2, dropout=0.5):
        super(CombinedModel, self).__init__()
        self.gesture_model_explicit = gesture_model_explicit
        self.gesture_model_implicit = gesture_model_implicit
        self.label_encoder_explicit = label_encoder_explicit
        self.label_encoder_implicit = label_encoder_implicit
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=2,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Precompute weight lookup tables (vectorized replacement for Python loops)
        explicit_weights = torch.zeros(len(label_encoder_explicit.categories_[0]))
        for idx, cls in enumerate(label_encoder_explicit.categories_[0]):
            if cls == 'Left_Hand_Explicit':
                explicit_weights[idx] = -1
            elif cls == 'Right_Hand_Explicit':
                explicit_weights[idx] = 1

        implicit_weights = torch.zeros(len(label_encoder_implicit.categories_[0]))
        for idx, cls in enumerate(label_encoder_implicit.categories_[0]):
            if cls == 'Left_Look':
                implicit_weights[idx] = -1
            elif cls == 'Left_Overshoulder':
                implicit_weights[idx] = -2
            elif cls == 'Right_Look':
                implicit_weights[idx] = 1

        self.register_buffer('explicit_weight_map', explicit_weights)
        self.register_buffer('implicit_weight_map', implicit_weights)

    def forward(self, skeleton_x):
        device = next(self.parameters()).device
        skeleton_x = skeleton_x.to(device)

        windows = _extract_windows(skeleton_x)
        batch_size, num_windows = windows.shape[0], windows.shape[1]
        flat_windows = windows.reshape(batch_size * num_windows, WINDOW_SIZE, -1)

        with torch.no_grad():
            explicit_out = _process_in_chunks(self.gesture_model_explicit, flat_windows)
            implicit_out = _process_in_chunks(self.gesture_model_implicit, flat_windows)

            # Vectorized weight mapping via index lookup (replaces Python string comparisons)
            _, explicit_idx = torch.max(explicit_out, 1)
            _, implicit_idx = torch.max(implicit_out, 1)
            explicit_weight = self.explicit_weight_map[explicit_idx]
            implicit_weight = self.implicit_weight_map[implicit_idx]

        gesture_features = torch.stack([explicit_weight, implicit_weight], dim=1)
        gesture_features = gesture_features.reshape(batch_size, num_windows, 2)

        lstm_out, _ = self.lstm(gesture_features)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


class CombinedModel_Explicit(nn.Module):
    """Combined model using only explicit gesture weights (ablation study)."""

    def __init__(self, gesture_model_explicit, gesture_model_implicit,
                 label_encoder_explicit, label_encoder_implicit,
                 hidden_size, output_size, num_layers=2, dropout=0.5):
        super(CombinedModel_Explicit, self).__init__()
        self.gesture_model_explicit = gesture_model_explicit
        self.gesture_model_implicit = gesture_model_implicit
        self.label_encoder_explicit = label_encoder_explicit
        self.label_encoder_implicit = label_encoder_implicit
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Precompute weight lookup table
        explicit_weights = torch.zeros(len(label_encoder_explicit.categories_[0]))
        for idx, cls in enumerate(label_encoder_explicit.categories_[0]):
            if cls == 'Left_Hand_Explicit':
                explicit_weights[idx] = -1
            elif cls == 'Right_Hand_Explicit':
                explicit_weights[idx] = 1

        self.register_buffer('explicit_weight_map', explicit_weights)

    def forward(self, skeleton_x):
        device = next(self.parameters()).device
        skeleton_x = skeleton_x.to(device)

        windows = _extract_windows(skeleton_x)
        batch_size, num_windows = windows.shape[0], windows.shape[1]
        flat_windows = windows.reshape(batch_size * num_windows, WINDOW_SIZE, -1)

        with torch.no_grad():
            explicit_out = _process_in_chunks(self.gesture_model_explicit, flat_windows)

            _, explicit_idx = torch.max(explicit_out, 1)
            explicit_weight = self.explicit_weight_map[explicit_idx]

        gesture_features = explicit_weight.unsqueeze(1).reshape(batch_size, num_windows, 1)

        lstm_out, _ = self.lstm(gesture_features)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out
