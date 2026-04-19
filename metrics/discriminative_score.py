"""
Discriminative Score for Time Series Generation Evaluation

Based on the implementation from alexouadi/SBTS:
https://github.com/alexouadi/SBTS

The discriminative score measures how well a post-hoc classifier can
distinguish between real and generated time series.

A score close to 0 indicates the generated data is indistinguishable
from real data (ideal). A score close to 0.5 indicates perfect
discrimination (poor generation quality).

Reference:
- Yoon et al., "Time-series Generative Adversarial Networks", NeurIPS 2019
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score


def train_test_divide(data_x, data_x_hat, train_rate=0.8):
    """
    Divide train and test data for both original and synthetic data.
    
    Args:
        data_x: Original data tensor
        data_x_hat: Generated data tensor
        train_rate: Ratio of training data
        
    Returns:
        train_x, train_x_hat, test_x, test_x_hat
    """
    # Original data split
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]
    
    train_x = data_x[train_idx]
    test_x = data_x[test_idx]
    
    # Synthetic data split
    no_hat = len(data_x_hat)
    idx_hat = np.random.permutation(no_hat)
    train_idx_hat = idx_hat[:int(no_hat * train_rate)]
    test_idx_hat = idx_hat[int(no_hat * train_rate):]
    
    train_x_hat = data_x_hat[train_idx_hat]
    test_x_hat = data_x_hat[test_idx_hat]
    
    return train_x, train_x_hat, test_x, test_x_hat


def batch_generator(data, batch_size):
    """
    Mini-batch generator.
    
    Args:
        data: Data tensor
        batch_size: Batch size
        
    Returns:
        Mini-batch of data
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    return data[train_idx]


class Discriminator(nn.Module):
    """
    GRU-based discriminator for time series classification.
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


class CNNDiscriminator(nn.Module):
    """
    1D-CNN based discriminator for time series classification.
    Focuses on local temporal patterns and shapes.
    """
    def __init__(self, input_size, seq_len):
        super().__init__()
        # Use 3 layers of 1D convolutions
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, seq_len, dim) -> needs (batch, dim, seq_len)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        y_hat_logit = self.fc(x)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat_logit, y_hat


def discriminative_score_cnn_metrics(ori_data, generated_data, iterations=2000, 
                                     device=torch.device('cpu'), device_ids=None):
    """
    Compute the discriminative score using a 1D-CNN discriminator.
    
    This metric complements the GRU-based score by looking for 
    local temporal artifacts rather than sequential dependencies.
    """
    # Convert to tensors if needed
    if isinstance(ori_data, np.ndarray):
        ori_data = torch.tensor(ori_data, dtype=torch.float32)
    if isinstance(generated_data, np.ndarray):
        generated_data = torch.tensor(generated_data, dtype=torch.float32)
    
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)
    
    # Ensure 3D shape
    if ori_data.dim() == 2:
        ori_data = ori_data.unsqueeze(-1)
    if generated_data.dim() == 2:
        generated_data = generated_data.unsqueeze(-1)
    
    # Basic parameters
    no, seq_len, dim = ori_data.shape
    no_hat = generated_data.shape[0]
    
    # Build CNN discriminator
    batch_size = max(1, min(128, no // 2))
    
    discriminator = CNNDiscriminator(dim, seq_len)
    if device_ids is not None and len(device_ids) > 1:
        discriminator = nn.DataParallel(discriminator, device_ids=device_ids)
    discriminator = discriminator.to(device)
    
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Train/test split
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)
    
    # Training
    discriminator.train()
    for itt in range(iterations):
        X_mb = batch_generator(train_x, batch_size)
        X_hat_mb = batch_generator(train_x_hat, batch_size)
        
        d_optimizer.zero_grad()
        
        y_logit_real, _ = discriminator(X_mb)
        y_logit_fake, _ = discriminator(X_hat_mb)
        
        d_loss_real = nn.BCEWithLogitsLoss()(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = nn.BCEWithLogitsLoss()(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake
        
        d_loss.backward()
        d_optimizer.step()
    
    # Evaluation
    discriminator.eval()
    with torch.no_grad():
        _, y_pred_real = discriminator(test_x)
        _, y_pred_fake = discriminator(test_x_hat)
    
    y_pred_final = np.squeeze(
        np.concatenate((y_pred_real.cpu().numpy(), y_pred_fake.cpu().numpy()), axis=0)
    )
    y_label_final = np.concatenate(
        (np.ones([len(y_pred_real)]), np.zeros([len(y_pred_fake)])), axis=0
    )
    
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score


def discriminative_score_metrics(ori_data, generated_data, iterations=2000, 
                                  device=torch.device('cpu'), device_ids=None):
    """
    Compute the discriminative score.
    
    The discriminative score is the absolute difference between 0.5 and
    the accuracy of a classifier trained to distinguish real from generated data.
    
    Score interpretation:
    - 0.0: Perfect generation (classifier cannot distinguish)
    - 0.5: Poor generation (classifier achieves 100% or 0% accuracy)
    
    Args:
        ori_data: Original data (tensor or numpy array)
        generated_data: Generated data (tensor or numpy array)
        iterations: Number of training iterations
        device: Torch device
        device_ids: GPU device IDs for DataParallel
        
    Returns:
        discriminative_score: Float in [0, 0.5]
    """
    # Convert to tensors if needed
    if isinstance(ori_data, np.ndarray):
        ori_data = torch.tensor(ori_data, dtype=torch.float32)
    if isinstance(generated_data, np.ndarray):
        generated_data = torch.tensor(generated_data, dtype=torch.float32)
    
    ori_data = ori_data.to(device)
    generated_data = generated_data.to(device)
    
    # Ensure 3D shape
    if ori_data.dim() == 2:
        ori_data = ori_data.unsqueeze(-1)
    if generated_data.dim() == 2:
        generated_data = generated_data.unsqueeze(-1)
    
    # Basic parameters
    no, seq_len, dim = ori_data.shape
    no_hat = generated_data.shape[0]
    if no < 2 or no_hat < 2 or seq_len < 2:
        raise ValueError("discriminative_score_metrics requires at least 2 samples and seq_len >= 2")
    
    # Build discriminator
    hidden_dim = max(int(dim / 2), 4)
    batch_size = max(1, min(128, no // 2))
    num_layers = 2
    
    discriminator = Discriminator(dim, hidden_dim, num_layers)
    if device_ids is not None and len(device_ids) > 1:
        discriminator = nn.DataParallel(discriminator, device_ids=device_ids)
    discriminator = discriminator.to(device)
    
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    
    # Train/test split
    train_x, train_x_hat, test_x, test_x_hat = train_test_divide(ori_data, generated_data)
    
    # Training
    discriminator.train()
    for itt in range(iterations):
        X_mb = batch_generator(train_x, batch_size)
        X_hat_mb = batch_generator(train_x_hat, batch_size)
        
        d_optimizer.zero_grad()
        
        y_logit_real, _ = discriminator(X_mb)
        y_logit_fake, _ = discriminator(X_hat_mb)
        
        d_loss_real = nn.BCEWithLogitsLoss()(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = nn.BCEWithLogitsLoss()(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake
        
        d_loss.backward()
        d_optimizer.step()
    
    # Evaluation
    discriminator.eval()
    with torch.no_grad():
        _, y_pred_real = discriminator(test_x)
        _, y_pred_fake = discriminator(test_x_hat)
    
    y_pred_final = np.squeeze(
        np.concatenate((y_pred_real.cpu().numpy(), y_pred_fake.cpu().numpy()), axis=0)
    )
    y_label_final = np.concatenate(
        (np.ones([len(y_pred_real)]), np.zeros([len(y_pred_fake)])), axis=0
    )
    
    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    discriminative_score = np.abs(0.5 - acc)
    
    return discriminative_score
