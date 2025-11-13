#!/usr/bin/env python
# coding: utf-8

# In[1]:


# OPTIMAL HYPERPARAMETERS - Based on empirical results

PATIENCE = 40
VERBOSE = 10

RNN_TYPE = 'GRU'
BIDIRECTIONAL = True
L1_LAMBDA = 0


# Reverse mapping from integers to pain level names
label_reverse_mapping = {
    0: 'no_pain',
    1: 'low_pain',
    2: 'high_pain'
}

# Create mapping dictionary for pain levels
pain_mapping = {
    'no_pain': 0,
    'low_pain': 1,
    'high_pain': 2
}

labels = ['no_pain', 'low_pain', 'high_pain']

num_classes = len(labels)

# # Define parameters to search
# param_grid = {
#     'window_size': [50, 100, 160],
#     'stride': [25],
#     'n_val_users' : [45],
#     'hidden_size': [64, 128],
#     'hidden_layers': [1, 2],
#     'batch_size': [64, 256],
#     'learning_rate' : [1e-3, 3e-4],
#     'dropout_rate': [0.0, 0.3, 0.5],
#     'l2_lambda': [0, 1e-4, 1e-3],
#     'k' : [5],
#     'epochs': [400]
# }


# # Define parameters to search
# param_grid = {
#     'window_size': [50, 160],
#     'stride': [25],
#     'n_val_users' : [45],
#     'hidden_size': [64, 128],
#     'hidden_layers': [1, 2],
#     'batch_size': [64, 256],
#     'learning_rate' : [1e-3, 3e-4],
#     'dropout_rate': [0.0, 0.3],
#     'l2_lambda': [0, 1e-4],
#     'k' : [5],
#     'epochs': [400]
# }



# param_grid = {

#     'window_size':   [100, 160],      # 2
#     'stride':        [25],
#     'n_val_users':   [45],

#     'hidden_size':   [64, 128],       # 2
#     'hidden_layers': [2],

#     'batch_size':    [128],
#     'learning_rate': [1e-3, 8e-4, 6e-4, 4e-4, 3e-4],  # 5 (log-ish sweep)
#     'dropout_rate':  [0.3],
#     'l2_lambda':     [1e-4],

#     'k':             [5],             # single subject-level split for speed
#     'epochs':        [200]            # rely on early stopping
# }


#test
param_grid = {

    'window_size':   [100],      # 2
    'stride':        [25],
    'n_val_users':   [45],

    'hidden_size':   [64],       # 2
    'hidden_layers': [2],

    'batch_size':    [128],
    'learning_rate': [1e-3],  # 5 (log-ish sweep)
    'dropout_rate':  [0.3],
    'l2_lambda':     [1e-4],

    'k':             [2],             # single subject-level split for speed
    'epochs':        [2]            # rely on early stopping
}


# In[2]:


import os, random, numpy as np



def physical_cores():
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    n = os.cpu_count() or 2
    return max(1, n // 2)  # stima fisici se non disponibile


SEED = 42

# Env PRIMA di import torch
CORES = physical_cores()
OMP = max(1, CORES - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(OMP))
os.environ.setdefault("MKL_NUM_THREADS", str(OMP))
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(OMP))
os.environ.setdefault("TORCH_NUM_INTEROP_THREADS", "1")
os.environ["PYTHONHASHSEED"] = str(SEED)

import torch
from torch import nn

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
try:
    torch.set_num_interop_threads(int(os.environ["TORCH_NUM_INTEROP_THREADS"]))
except RuntimeError as e:
    print("skip set_num_interop_threads:", e)



# Import necessary libraries
import os, subprocess, shlex

# Set environment variables before importing modules
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import logging
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(SEED)
random.seed(SEED)



# Device selection: prefer CUDA when available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True


print("DEBUG TORCH:")
print("  torch.__version__     =", torch.__version__)
print("  torch.version.cuda    =", torch.version.cuda)
print("  torch.cuda.is_available() =", torch.cuda.is_available())
print("  torch.cuda.device_count()  =", torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print("  GPU name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("  get_device_name error:", e)
print("  selected device =", device)


# from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader



print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

# Import other libraries
import copy
import shutil
from itertools import product
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns

# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
# plt.rc('font', size=14)
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


X_train = pd.read_csv('pirate_pain_train.csv')
y_train = pd.read_csv('pirate_pain_train_labels.csv')


# First map the labels in y_train
y_train['label_encoded'] = y_train['label'].map(pain_mapping)

# Then merge with X_train based on sample_index
X_train = X_train.merge(
    y_train[['sample_index', 'label_encoded']],
    on='sample_index',
    how='left'
)

# Rename the column
X_train.rename(columns={'label_encoded': 'label'}, inplace=True)

# Verify the mapping worked correctly
print("\nFirst few rows of X_train with encoded labels:")
print(X_train[['sample_index', 'label']].head(10))

print("\nLabel value counts:")
print(X_train['label'].value_counts())

print("\nCheck for NaN labels:")
print(f"NaN count: {X_train['label'].isna().sum()}")

input_shape = X_train.shape[1:]



# In[4]:


# Drop joint_30 column (contains only NaN values)
print("Dropping joint_30 column (all NaN values)...")
for df in [X_train]:
    if 'joint_30' in df.columns:
        df.drop('joint_30', axis=1, inplace=True)
        print(f"Dropped joint_30 from {df.shape}")

print("\nColumns after dropping joint_30:")
print(f"X_train columns: {X_train.shape[1]}")


# 

# In[5]:


# print("\n--- Data successfully loaded ---")
# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}")

# print("\n--- Initial Feature Count Breakdown ---")
# all_features = X_train.columns.drop(['sample_index', 'time'])
# joint_features = [col for col in all_features if col.startswith('joint_')]
# static_features = [col for col in all_features if col in ['n_legs', 'n_hands', 'n_eyes']]

# print(f"Total Features (excluding IDs): {len(all_features)}")
# print(f"Core Time-Series (Joints): {len(joint_features)} columns")
# print(f"Static Subject Characteristics: {len(static_features)} columns")
# print(f"Pain Survey Features: {len(all_features) - len(joint_features) - len(static_features)} columns")




# In[6]:


# First: Convert categorical variables to binary (two -> 1, others -> 0)
binary_cols = ['n_hands', 'n_eyes', 'n_legs']
for col in binary_cols:
    for df_ in [X_train]:
        df_[col] = df_[col].map(lambda x: 1 if str(x).lower().strip() == 'two' else 0)


# In[7]:


print("\n--- Data structure ---")
print("\nX_train Info:")
X_train.info(verbose=True)
print(f"\nMissing values in X_train: {X_train.isnull().sum().sum()}")
print("\ny_train Info:")
y_train.info(verbose=True)
print(f"\nMissing values in y_train: {y_train.isnull().sum().sum()}")


# In[8]:


X_train.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


def build_sequences(df, window=200, stride=50):
    """
    Build sequences from time-series data

    Args:
        df: DataFrame with the data
        window: Window size for sequences
        stride: Stride for overlapping windows

    Returns:
        dataset: numpy array of sequences
        labels: numpy array of labels
    """
    # Initialise lists to store sequences and their corresponding labels
    dataset = []
    labels = []

    # Iterate over unique IDs in the DataFrame
    for sample_id in df['sample_index'].unique():
        # Extract sensor data for the current ID
        drop_cols = [c for c in ['sample_index', 'time', 'label', 'labels'] if c in df.columns]
        temp = df[df['sample_index'] == sample_id].drop(columns=drop_cols).values.astype('float32')

        # Retrieve the activity label for the current ID
        label_series = df[df['sample_index'] == sample_id]['label']

        # Check if label column exists and has values
        if label_series.empty:
            print(f"Warning: No label found for sample_id {sample_id}")
            continue

        label_value = label_series.values[0]

        # Skip samples with NaN labels
        if pd.isna(label_value):
            print(f"Warning: NaN label for sample_id {sample_id}, skipping...")
            continue

        # Convert to int and validate
        try:
            label = int(label_value)
            if label < 0 or label > 2:  # Assuming 3 classes: 0, 1, 2
                print(f"Warning: Invalid label {label} for sample_id {sample_id}, skipping...")
                continue
        except (ValueError, TypeError) as e:
            print(f"Warning: Cannot convert label {label_value} to int for sample_id {sample_id}: {e}")
            continue

        # Calculate padding length to ensure full windows
        padding_len = window - len(temp) % window if len(temp) % window != 0 else 0

        # Create zero padding with correct number of features
        if padding_len > 0:
            padding = np.zeros((padding_len, temp.shape[1]), dtype='float32')
            temp = np.concatenate((temp, padding))

        # Build feature windows and associate them with labels
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            labels.append(label)
            idx += stride

    # Convert lists to numpy arrays for further processing
    dataset = np.array(dataset, dtype='float32')
    labels = np.array(labels, dtype='int64')

    print(f"Built {len(dataset)} sequences with {len(labels)} labels")

    return dataset, labels


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


BATCH_SIZE = 512


# In[11]:


def make_loader(ds, batch_size, shuffle, drop_last, num_workers=None):
    """
    Robust DataLoader for macOS/CPU and CUDA.
    - Defaults to num_workers=0 on LOCAL/CPU to avoid hangs.
    - Enables pin_memory/prefetch only on CUDA.
    """
    import os
    import torch
    from torch.utils.data import DataLoader

    if num_workers is None:
        if 'LOCAL' in globals() and globals()['LOCAL']:
            num_workers = 0
        else:
            cpu_cores = os.cpu_count() or 2
            num_workers = min(4, max(0, cpu_cores - 1))

    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
        kwargs['pin_memory_device'] = 'cuda'
        if num_workers > 0:
            kwargs['prefetch_factor'] = 4

    return DataLoader(**kwargs)

# In[ ]:





# In[ ]:





# In[12]:


def recurrent_summary(model, input_size):
    """
    Custom summary function that emulates torchinfo's output while correctly
    counting parameters for RNN/GRU/LSTM layers.

    This function is designed for models whose direct children are
    nn.Linear, nn.RNN, nn.GRU, or nn.LSTM layers.

    Args:
        model (nn.Module): The model to analyze.
        input_size (tuple): Shape of the input tensor (e.g., (seq_len, features)).
    """

    # Dictionary to store output shapes captured by forward hooks
    output_shapes = {}
    # List to track hook handles for later removal
    hooks = []

    def get_hook(name):
        """Factory function to create a forward hook for a specific module."""
        def hook(module, input, output):
            # Handle RNN layer outputs (returns a tuple)
            if isinstance(output, tuple):
                # output[0]: all hidden states with shape (batch, seq_len, hidden*directions)
                shape1 = list(output[0].shape)
                shape1[0] = -1  # Replace batch dimension with -1

                # output[1]: final hidden state h_n (or tuple (h_n, c_n) for LSTM)
                if isinstance(output[1], tuple):  # LSTM case: (h_n, c_n)
                    shape2 = list(output[1][0].shape)  # Extract h_n only
                else:  # RNN/GRU case: h_n only
                    shape2 = list(output[1].shape)

                # Replace batch dimension (middle position) with -1
                shape2[1] = -1

                output_shapes[name] = f"[{shape1}, {shape2}]"

            # Handle standard layer outputs (e.g., Linear)
            else:
                shape = list(output.shape)
                shape[0] = -1  # Replace batch dimension with -1
                output_shapes[name] = f"{shape}"
        return hook

    # 1. Determine the device where model parameters reside
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # Fallback for models without parameters

    # 2. Create a dummy input tensor with batch_size=1
    dummy_input = torch.randn(1, *input_size).to(device)

    # 3. Register forward hooks on target layers
    # Iterate through direct children of the model (e.g., self.rnn, self.classifier)
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.RNN, nn.GRU, nn.LSTM)):
            # Register the hook and store its handle for cleanup
            hook_handle = module.register_forward_hook(get_hook(name))
            hooks.append(hook_handle)

    # 4. Execute a dummy forward pass in evaluation mode
    model.eval()
    with torch.no_grad():
        try:
            model(dummy_input)
        except Exception as e:
            print(f"Error during dummy forward pass: {e}")
            # Clean up hooks even if an error occurs
            for h in hooks:
                h.remove()
            return

    # 5. Remove all registered hooks
    for h in hooks:
        h.remove()

    # --- 6. Print the summary table ---

    print("-" * 79)
    # Column headers
    print(f"{'Layer (type)':<25} {'Output Shape':<28} {'Param #':<18}")
    print("=" * 79)

    total_params = 0
    total_trainable_params = 0

    # Iterate through modules again to collect and display parameter information
    for name, module in model.named_children():
        if name in output_shapes:
            # Count total and trainable parameters for this module
            module_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

            total_params += module_params
            total_trainable_params += trainable_params

            # Format strings for display
            layer_name = f"{name} ({type(module).__name__})"
            output_shape_str = str(output_shapes[name])
            params_str = f"{trainable_params:,}"

            print(f"{layer_name:<25} {output_shape_str:<28} {params_str:<15}")

    print("=" * 79)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_trainable_params:,}")
    print(f"Non-trainable params: {total_params - total_trainable_params:,}")
    print("-" * 79)


# In[13]:


class RecurrentClassifier(nn.Module):
    """
    Generic RNN classifier (RNN, LSTM, GRU).
    Uses the last hidden state for classification.
    """
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            rnn_type='GRU',        # 'RNN', 'LSTM', or 'GRU'
            bidirectional=False,
            dropout_rate=0.2
            ):
        super().__init__()

        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        # Map string name to PyTorch RNN class
        rnn_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'GRU': nn.GRU
        }

        if rnn_type not in rnn_map:
            raise ValueError("rnn_type must be 'RNN', 'LSTM', or 'GRU'")

        rnn_module = rnn_map[rnn_type]

        # Dropout is only applied between layers (if num_layers > 1)
        dropout_val = dropout_rate if num_layers > 1 else 0

        # Create the recurrent layer
        self.rnn = rnn_module(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # Input shape: (batch, seq_len, features)
            bidirectional=bidirectional,
            dropout=dropout_val
        )

        # Calculate input size for the final classifier
        if self.bidirectional:
            classifier_input_size = hidden_size * 2 # Concat fwd + bwd
        else:
            classifier_input_size = hidden_size

        # Final classification layer
        self.classifier = nn.Linear(classifier_input_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """

        # rnn_out shape: (batch_size, seq_len, hidden_size * num_directions)
        rnn_out, hidden = self.rnn(x)

        # LSTM returns (h_n, c_n), we only need h_n
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]

        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)

        if self.bidirectional:
            # Reshape to (num_layers, 2, batch_size, hidden_size)
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)

            # Concat last fwd (hidden[-1, 0, ...]) and bwd (hidden[-1, 1, ...])
            # Final shape: (batch_size, hidden_size * 2)
            hidden_to_classify = torch.cat([hidden[-1, 0, :, :], hidden[-1, 1, :, :]], dim=1)
        else:
            # Take the last layer's hidden state
            # Final shape: (batch_size, hidden_size)
            hidden_to_classify = hidden[-1]

        # Get logits
        logits = self.classifier(hidden_to_classify)
        return logits


# Create model and display architecture with parameter count
rnn_model = RecurrentClassifier(
    input_size=input_shape[-1], # Pass the number of features
    hidden_size=128,
    num_layers=2,
    num_classes=num_classes,
    dropout_rate=0.,
    rnn_type='RNN'
    ).to(device)
recurrent_summary(rnn_model, input_size=input_shape)


# In[ ]:





# In[ ]:





# In[14]:


# Initialize best model tracking variables
best_model = None
best_performance = float('-inf')


# In[15]:


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, l1_lambda=0, l2_lambda=0):
    """
    Perform one complete training epoch through the entire training dataset.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
        scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
        l1_lambda (float): Lambda for L1 regularization
        l2_lambda (float): Lambda for L2 regularization

    Returns:
        tuple: (average_loss, f1 score) - Training loss and f1 score for this epoch
    """
    model.train()  # Set model to training mode

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    # Iterate through training batches
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move data to device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)

        # Clear gradients from previous step
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with mixed precision (if CUDA available)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Add L1 and L2 regularization
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            l2_norm = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm


        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    # Calculate epoch metrics
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_f1 = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_f1


# In[16]:


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Perform one complete validation epoch through the entire validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate (must be in eval mode)
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        criterion (nn.Module): Loss function used to calculate validation loss
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)

    Returns:
        tuple: (average_loss, accuracy) - Validation loss and accuracy for this epoch

    Note:
        This function automatically sets the model to evaluation mode and disables
        gradient computation for efficiency during validation.
    """
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_predictions = []
    all_targets = []

    # Disable gradient computation for validation
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision (if CUDA available)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(inputs)
                loss = criterion(logits, targets)

            # Accumulate metrics
            running_loss += loss.item() * inputs.size(0)
            predictions = logits.argmax(dim=1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate epoch metrics
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = f1_score(
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        average='weighted'
    )

    return epoch_loss, epoch_accuracy


# In[17]:


def log_metrics_to_tensorboard(writer, epoch, train_loss, train_f1, val_loss, val_f1, model):
    """
    Log training metrics and model parameters to TensorBoard for visualization.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter object for logging
        epoch (int): Current epoch number (used as x-axis in TensorBoard plots)
        train_loss (float): Training loss for this epoch
        train_f1 (float): Training f1 score for this epoch
        val_loss (float): Validation loss for this epoch
        val_f1 (float): Validation f1 score for this epoch
        model (nn.Module): The neural network model (for logging weights/gradients)

    Note:
        This function logs scalar metrics (loss/f1 score) and histograms of model
        parameters and gradients, which helps monitor training progress and detect
        issues like vanishing/exploding gradients.
    """
    # Log scalar metrics
    writer.add_scalar('Loss/Training', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('F1/Training', train_f1, epoch)
    writer.add_scalar('F1/Validation', val_f1, epoch)

    # Log model parameters and gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if the tensor is not empty before adding a histogram
            if param.numel() > 0:
                writer.add_histogram(f'{name}/weights', param.data, epoch)
            if param.grad is not None:
                # Check if the gradient tensor is not empty before adding a histogram
                if param.grad.numel() > 0:
                    if param.grad is not None and torch.isfinite(param.grad).all():
                        writer.add_histogram(f'{name}/gradients', param.grad.data, epoch)


# In[18]:


def fit(model, train_loader, val_loader, epochs, criterion, optimizer, scaler, device,
        l1_lambda=0, l2_lambda=0, patience=0, evaluation_metric="val_f1", mode='max',
        restore_best_weights=True, writer=None, verbose=10, experiment_name=""):
    """
    Train the neural network model on the training data and validate on the validation data.

    Args:
        model (nn.Module): The neural network model to train
        train_loader (DataLoader): PyTorch DataLoader containing training data batches
        val_loader (DataLoader): PyTorch DataLoader containing validation data batches
        epochs (int): Number of training epochs
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss, MSELoss)
        optimizer (torch.optim): Optimization algorithm (e.g., Adam, SGD)
        scaler (GradScaler): PyTorch's gradient scaler for mixed precision training
        device (torch.device): Computing device ('cuda' for GPU, 'cpu' for CPU)
        l1_lambda (float): L1 regularization coefficient (default: 0)
        l2_lambda (float): L2 regularization coefficient (default: 0)
        patience (int): Number of epochs to wait for improvement before early stopping (default: 0)
        evaluation_metric (str): Metric to monitor for early stopping (default: "val_f1")
        mode (str): 'max' for maximizing the metric, 'min' for minimizing (default: 'max')
        restore_best_weights (bool): Whether to restore model weights from best epoch (default: True)
        writer (SummaryWriter, optional): TensorBoard SummaryWriter object for logging (default: None)
        verbose (int, optional): Frequency of printing training progress (default: 10)
        experiment_name (str, optional): Experiment name for saving models (default: "")

    Returns:
        tuple: (model, training_history) - Trained model and metrics history
    """

    # Initialize metrics tracking
    training_history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': []
    }

    # Configure early stopping if patience is set
    if patience > 0:
        patience_counter = 0
        best_metric = float('-inf') if mode == 'max' else float('inf')
        best_epoch = 0

    print(f"Training {epochs} epochs...")

    # Main training loop: iterate through epochs
    for epoch in range(1, epochs + 1):

        # Forward pass through training data, compute gradients, update weights
        train_loss, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, l1_lambda, l2_lambda
        )

        # Evaluate model on validation data without updating weights
        val_loss, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Store metrics for plotting and analysis
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_f1'].append(train_f1)
        training_history['val_f1'].append(val_f1)

        # Write metrics to TensorBoard for visualization
        if writer is not None:
            log_metrics_to_tensorboard(
                writer, epoch, train_loss, train_f1, val_loss, val_f1, model
            )

        # Print progress every N epochs or on first epoch
        if verbose > 0:
            if epoch % verbose == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{epochs} | "
                    f"Train: Loss={train_loss:.4f}, F1 Score={train_f1:.4f} | "
                    f"Val: Loss={val_loss:.4f}, F1 Score={val_f1:.4f}")

        # Early stopping logic: monitor metric and save best model
        if patience > 0:
            current_metric = training_history[evaluation_metric][-1]
            is_improvement = (current_metric > best_metric) if mode == 'max' else (current_metric < best_metric)

            if is_improvement:
                best_metric = current_metric
                best_epoch = epoch
                torch.save(model.state_dict(), "models/"+experiment_name+'_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    # Restore best model weights if early stopping was used
    if restore_best_weights and patience > 0:
        model.load_state_dict(torch.load("models/"+experiment_name+'_model.pt'))
        print(f"Best model restored from epoch {best_epoch} with {evaluation_metric} {best_metric:.4f}")

    # Save final model if no early stopping
    if patience == 0:
        torch.save(model.state_dict(), "models/"+experiment_name+'_model.pt')

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    return model, training_history


# # KFOLD

# In[19]:


def k_shuffle_split_cross_validation_round_rnn(df, epochs, criterion, device,
                            k, n_val_users, batch_size, hidden_layers, hidden_size, learning_rate, dropout_rate,
                            window_size, stride, rnn_type, bidirectional,
                            l1_lambda=0, l2_lambda=0, patience=0, evaluation_metric="val_f1", mode='max',
                            restore_best_weights=True, writer=None, verbose=10, seed=42, experiment_name=""):
    """
    Perform K-fold shuffle split cross-validation with sample-based splitting for Pirate Pain time series data.

    Args:
        df: DataFrame with columns ['sample_index', 'time', 'label', 'pain_survey_*', 'joint_*', 'n_legs', 'n_hands', 'n_eyes']
        epochs: Number of training epochs
        criterion: Loss function
        device: torch.device for computation
        k: Number of cross-validation splits
        n_val_users: Number of samples for validation set
        n_test_users: Number of samples for test set
        batch_size: Batch size for training
        hidden_layers: Number of recurrent layers
        hidden_size: Hidden state dimensionality
        learning_rate: Learning rate for optimizer
        dropout_rate: Dropout rate
        window_size: Length of sliding windows
        stride: Step size for sliding windows
        rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
        bidirectional: Whether to use bidirectional RNN
        l1_lambda: L1 regularization coefficient (if used)
        l2_lambda: L2 regularization coefficient (weight_decay)
        patience: Early stopping patience
        evaluation_metric: Metric to monitor for early stopping
        mode: 'max' or 'min' for evaluation metric
        restore_best_weights: Whether to restore best weights after training
        writer: TensorBoard writer
        verbose: Verbosity level
        seed: Random seed
        experiment_name: Name for experiment logging

    Returns:
        fold_losses: Dict with validation losses for each split
        fold_metrics: Dict with validation F1 scores for each split
        best_scores: Dict with best F1 score for each split plus mean and std
    """

    # Initialise containers for results across all splits
    fold_losses = {}
    fold_metrics = {}
    best_scores = {}

    # Define pain level mapping
    pain_mapping = {
        'no_pain': 0,
        'low_pain': 1,
        'high_pain': 2
    }

    # Define columns to normalize
    pain_survey_columns = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4', 'n_legs', 'n_hands', 'n_eyes']
    joint_columns = [f'joint_{i:02d}' for i in range(30)]  # joint_00 through joint_29
    scale_columns = pain_survey_columns + joint_columns

    # Get model architecture parameters
    # Count features (excluding sample_index, time, label)
    feature_cols = scale_columns  # All features that will be used
    in_features = len(feature_cols)
    num_classes = 3  # no_pain, low_pain, high_pain

    # Initialise model architecture
    model = RecurrentClassifier(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=hidden_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        rnn_type=rnn_type
    ).to(device)

    # Store initial weights to reset model for each split
    initial_state = copy.deepcopy(model.state_dict())

    # Iterate through K random splits
    for split_idx in range(k):

        if verbose > 0:
            print(f"Split {split_idx+1}/{k}")

        # Get unique sample IDs and shuffle them with split-specific seed
        unique_samples = df['sample_index'].unique()
        random.seed(seed + split_idx)
        random.shuffle(unique_samples)

        # Calculate the number of samples for the training set
        n_train_samples = len(unique_samples) - n_val_users

        # Split the shuffled sample IDs into training, validation, and test sets
        train_samples = unique_samples[:n_train_samples]
        val_samples = unique_samples[n_train_samples:n_train_samples + n_val_users]

        # Split the dataset into training, validation, and test sets based on sample IDs
        df_train = df[df['sample_index'].isin(train_samples)].copy()
        df_val = df[df['sample_index'].isin(val_samples)].copy()

        if verbose > 0:
            print(f"  Training set shape: {df_train.shape}")
            print(f"  Validation set shape: {df_val.shape}")

        # Map pain labels to integers (if not already mapped)
        if df_train['label'].dtype == 'object':
            df_train['label'] = df_train['label'].map(pain_mapping)
            df_val['label'] = df_val['label'].map(pain_mapping)

        # Normalise features using training set statistics
        train_max = df_train[scale_columns].max()
        train_min = df_train[scale_columns].min()

        for column in scale_columns:
            df_train[column] = (df_train[column] - train_min[column]) / (train_max[column] - train_min[column] + 1e-8)
            df_val[column] = (df_val[column] - train_min[column]) / (train_max[column] - train_min[column] + 1e-8)

        # Build sequences using the existing build_sequences function
        X_train, y_train = build_sequences(df_train, window=window_size, stride=stride)
        X_val, y_val = build_sequences(df_val, window=window_size, stride=stride)

        if verbose > 0:
            print(f"  Training sequences shape: {X_train.shape}")
            print(f"  Validation sequences shape: {X_val.shape}")
            print(f"  Test sequences shape: {X_test.shape}")

        # Create PyTorch datasets
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds   = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        # Create data loaders
        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = make_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # Reset model to initial weights for fair comparison across splits
        model.load_state_dict(initial_state)

        # Define optimizer with L2 regularization
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

        # Enable mixed precision training for GPU acceleration
        split_scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))

        # Create directory for model checkpoints
        os.makedirs(f"models/{experiment_name}", exist_ok=True)

        # Train model on current split
        model, training_history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            criterion=criterion,
            optimizer=optimizer,
            scaler=split_scaler,
            device=device,
            writer=writer,
            patience=patience,
            verbose=verbose,
            l1_lambda=l1_lambda,
            evaluation_metric=evaluation_metric,
            mode=mode,
            restore_best_weights=restore_best_weights,
            experiment_name=experiment_name+"/split_"+str(split_idx)
        )

        # Store results for this split
        fold_losses[f"split_{split_idx}"] = training_history['val_loss']
        fold_metrics[f"split_{split_idx}"] = training_history['val_f1']
        best_scores[f"split_{split_idx}"] = max(training_history['val_f1'])

    # Compute mean and standard deviation of best scores across splits
    best_scores["mean"] = np.mean([best_scores[k] for k in best_scores.keys() if k.startswith("split_")])
    best_scores["std"] = np.std([best_scores[k] for k in best_scores.keys() if k.startswith("split_")])

    if verbose > 0:
        print(f"Best score: {best_scores['mean']:.4f}±{best_scores['std']:.4f}")

    return fold_losses, fold_metrics, best_scores


# In[20]:


def grid_search_cv_rnn(df, param_grid, fixed_params, cv_params, verbose=True):
    """
    Execute grid search with K-shuffle-split cross-validation for RNN models on time series data.

    Args:
        df: DataFrame with columns ['user_id', 'activity', 'x_axis', 'y_axis', 'z_axis', 'id']
        param_grid: Dict of parameters to test, e.g. {'batch_size': [16, 32], 'rnn_type': ['LSTM', 'GRU']}
        fixed_params: Dict of fixed hyperparameters (hidden_size, learning_rate, window_size, stride, etc.)
        cv_params: Dict of CV settings (epochs, k, patience, criterion, scaler, device, etc.)
        verbose: Print progress for each configuration

    Returns:
        results: Dict with scores for each configuration
        best_config: Dict with best hyperparameter combination
        best_score: Best mean F1 score achieved
    """
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = {}
    best_score = -np.inf
    best_config = None

    total = len(combinations)

    for idx, combo in enumerate(combinations, 1):
        # Create current configuration dict
        current_config = dict(zip(param_names, combo))
        config_str = "_".join([f"{k}_{v}" for k, v in current_config.items()])

        if verbose:
            print(f"\nConfiguration {idx}/{total}:")
            for param, value in current_config.items():
                print(f"  {param}: {value}")

        # Merge current config with fixed parameters
        run_params = {**fixed_params, **current_config}

        # Execute cross-validation
        _, _, fold_scores = k_shuffle_split_cross_validation_round_rnn(
            df=df,
            experiment_name=config_str,
            **run_params,
            **cv_params
        )

        # Store results
        results[config_str] = fold_scores

        # Track best configuration
        if fold_scores["mean"] > best_score:
            best_score = fold_scores["mean"]
            best_config = current_config.copy()
            if verbose:
                print("  NEW BEST SCORE!")

        if verbose:
            print(f"  F1 Score: {fold_scores['mean']:.4f}±{fold_scores['std']:.4f}")

    return results, best_config, best_score


def plot_top_configurations_rnn(results, k_splits, top_n=5, figsize=(14, 7)):
    """
    Visualise top N RNN configurations with boxplots of F1 scores across CV splits.

    Args:
        results: Dict of results from grid_search_cv_rnn
        k_splits: Number of CV splits used
        top_n: Number of top configurations to display
        figsize: Figure size tuple
    """
    # Sort by mean score
    config_scores = {name: data['mean'] for name, data in results.items()}
    sorted_configs = sorted(config_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top N
    top_configs = sorted_configs[:min(top_n, len(sorted_configs))]

    # Prepare boxplot data
    boxplot_data = []
    labels = []

    # Define a dictionary for replacements, ordered to handle prefixes correctly
    replacements = {
        'batch_size_': 'BS=',
        'learning_rate_': '\nLR=',
        'hidden_layers_': '\nHL=',
        'hidden_size_': '\nHS=',
        'dropout_rate_': '\nDR=',
        'window_size_': '\nWS=',
        'stride_': '\nSTR=',
        'rnn_type_': '\nRNN=',
        'bidirectional_': '\nBIDIR=',
        'l1_lambda_': '\nL1=',
        'l2_lambda_': '\nL2='
    }

    # Replacements for separators
    separator_replacements = {
        '_learning_rate_': '\nLR=',
        '_hidden_layers_': '\nHL=',
        '_hidden_size_': '\nHS=',
        '_dropout_rate_': '\nDR=',
        '_window_size_': '\nWS=',
        '_stride_': '\nSTR=',
        '_rnn_type_': '\nRNN=',
        '_bidirectional_': '\nBIDIR=',
        '_l1_lambda_': '\nL1=',
        '_l2_lambda_': '\nL2=',
        '_': ''
    }

    for config_name, mean_score in top_configs:
        # Extract best score from each split (auto-detect number of splits)
        split_scores = []
        for i in range(k_splits):
            if f'split_{i}' in results[config_name]:
                split_scores.append(results[config_name][f'split_{i}'])
        boxplot_data.append(split_scores)

        # Verify we have the expected number of splits
        if len(split_scores) != k_splits:
            print(f"Warning: Config {config_name} has {len(split_scores)} splits, expected {k_splits}")

        # Create readable label using the replacements dictionary
        readable_label = config_name
        for old, new in replacements.items():
            readable_label = readable_label.replace(old, new)

        # Apply separator replacements
        for old, new in separator_replacements.items():
             readable_label = readable_label.replace(old, new)

        labels.append(f"{readable_label}\n(μ={mean_score:.3f})")

    # Create plot
#     fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Styling
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Highlight best configuration
    ax.get_xticklabels()[0].set_fontweight('bold')

    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Configuration')
    ax.set_title(f'Top {len(top_configs)} RNN Configurations - F1 Score Distribution Across {k_splits} Splits')
    ax.grid(alpha=0.3, axis='y')

#     plt.xticks(rotation=0, ha='center')
#     plt.tight_layout()
#     plt.show()


# In[21]:


# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()


# In[22]:


# === Replaced: proper training call under main-guard ===
if __name__ == "__main__":
    # Train model and track training history
    rnn_model, training_history = fit(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        writer=writer,
        verbose=1,
        experiment_name="rnn",
        patience=PATIENCE,
    )

    # Update best model if current performance is superior
    if training_history and isinstance(training_history, dict) and 'val_f1' in training_history and training_history['val_f1']:
        if training_history['val_f1'][-1] > best_performance:
            best_model = rnn_model
            best_performance = training_history['val_f1'][-1]
# === End replacement ===


# In[23]:


plot_top_configurations_rnn(results, k_splits=param_grid['k'][0], top_n=5)


# # Prediction

# In[24]:


# === Replaced: proper training call under main-guard ===
if __name__ == "__main__":
    # Train model and track training history
    rnn_model, training_history = fit(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        writer=writer,
        verbose=1,
        experiment_name="rnn",
        patience=PATIENCE,
    )

    # Update best model if current performance is superior
    if training_history and isinstance(training_history, dict) and 'val_f1' in training_history and training_history['val_f1']:
        if training_history['val_f1'][-1] > best_performance:
            best_model = rnn_model
            best_performance = training_history['val_f1'][-1]
# === End replacement ===


# In[ ]:


import shutil
from google.colab import files

zip_path = shutil.make_archive('/content/models', 'zip', '/content', 'models')
files.download(zip_path)


# In[ ]:




