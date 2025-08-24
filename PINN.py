import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader
from dataset import StreamDataset
from flow_detection import SVCalculator, ClassicalFlow, ClassicalFlowToSV
from filtering import KalmanFilter
import matplotlib.pyplot as plt

# Define PINN model
class PINNOpticalFlow(nn.Module):
    def __init__(self, intrinsic_matrix, frame_rate):
        super(PINNOpticalFlow, self).__init__()
        
        # Sub-module for flow calculation
        flow_calculator = ClassicalFlow()
        
        # Neural network for predicting flow features
        self.hidden_layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # Outputting (u, v) velocity
        )
        
    def forward(self, x):
        return self.hidden_layers(x)
    
    # Physics-based loss
    def physics_loss(self, predictions, depth):
        depth.requires_grad = True
        u, v = predictions[..., 0], predictions[..., 1]
        du_dt = grad(u.sum(), depth, create_graph=True, allow_unused=True)[0]
        dv_dt = grad(v.sum(), depth, create_graph=True, allow_unused=True)[0]

        if du_dt is None or dv_dt is None:
            physics_residual = torch.tensor(0.0, requires_grad=True, device=predictions.device)
        else:
            physics_residual = du_dt + dv_dt

        # Add the physical constraint: h = x + mp / xp
        x = depth  # Example input
        mp = torch.tensor(1.0, device=predictions.device)  # Example constant
        xp = torch.clamp(x, min=1e-6)  # Avoid division by zero
        h = x + mp / xp

        physical_constraint_residual = (predictions[..., 1] - h).pow(2).mean()


        return physics_residual.pow(2).mean() + physical_constraint_residual

    def combined_loss(self, data_loss, predictions, depth, physics_loss_weight=1.0):
        physics_loss = self.physics_loss(predictions, depth)
        return data_loss + physics_loss_weight * physics_loss

# Data Preprocessing
def preprocess_frames(frames):
    batch_size, num_frames, channels, height, width = frames.shape
    x_coords = torch.arange(0, width).repeat(height, 1).view(-1, 1).repeat(batch_size * num_frames, 1)
    y_coords = torch.arange(0, height).repeat(width, 1).t().contiguous().view(-1, 1).repeat(batch_size * num_frames, 1)
    t = torch.linspace(0, 1, steps=num_frames).view(-1, 1).repeat(batch_size, height * width).view(-1, 1)
    inputs = torch.cat([x_coords, y_coords, t], dim=1).float()
    return inputs

# Training Function
def train_pinn(model, train_dataset, epochs=100, batch_size=8, learning_rate=1e-4, physics_weight=1.0):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_data_loss, total_physics_loss = 0.0, 0.0
        for batch in train_loader:
            frames, labels = batch
            frames, labels = frames.to(model.hidden_layers[0].weight.device), labels.to(model.hidden_layers[0].weight.device)
            
            # Preprocess frames for model input
            inputs = preprocess_frames(frames)
            
            # Forward pass
            predictions = model(inputs)
            
            # Reshape labels to match predictions
            repeated_labels = labels[:, :, 0:2].view(-1, 2).repeat(inputs.shape[0] // labels.shape[0], 1)
            data_loss = criterion(predictions, repeated_labels)
            
            # Combined loss
            total_loss = model.combined_loss(data_loss, predictions, labels[:, :, 1], physics_loss_weight=physics_weight)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_data_loss += data_loss.item()
            total_physics_loss += model.physics_loss(predictions, labels[:, :, 1]).item()
        
        avg_data_loss = total_data_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Data Loss: {avg_data_loss:.4f}, Physics Loss: {avg_physics_loss:.4f}")

# Main
if __name__ == "__main__":
    intrinsic_matrix = torch.Tensor([[1912.5, 0, 640], [0, 1912.5, 360], [0, 0, 1]])
    frame_rate = 30
    model = PINNOpticalFlow(intrinsic_matrix, frame_rate)
    train_dataset = StreamDataset('FlowDataSet/train')
    train_pinn(model, train_dataset, epochs=100, batch_size=8, learning_rate=1e-4, physics_weight=1.0)