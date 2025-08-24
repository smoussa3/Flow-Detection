import torch
from dataset import StreamDataset
from PINN import PINNOpticalFlow
import matplotlib.pyplot as plt

def preprocess_frames(frames):
    # Get dimensions
    batch_size, num_frames, channels, height, width = frames.shape
    
    # Generate (x, y) coordinates for each pixel in the frame
    x_coords = torch.arange(0, width).repeat(height, 1).view(-1, 1).repeat(batch_size * num_frames, 1)
    y_coords = torch.arange(0, height).repeat(width, 1).t().contiguous().view(-1, 1).repeat(batch_size * num_frames, 1)
    
    # Generate time steps for each frame and repeat for each pixel
    t = torch.linspace(0, 1, steps=num_frames).view(-1, 1).repeat(batch_size, height * width).view(-1, 1)
    
    # Concatenate (x, y, t) to create input of shape [batch_size * num_frames * height * width, 3]
    inputs = torch.cat([x_coords, y_coords, t], dim=1).float()
    return inputs.to(frames.device)

def test_pinn(model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    true_velocities = []
    predicted_velocities = []
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            frames, labels = batch
            frames, labels = frames.to(model.hidden_layers[0].weight.device), labels.to(model.hidden_layers[0].weight.device)
            
            # Preprocess frames to create model-compatible inputs
            inputs = preprocess_frames(frames)
            
            # Forward pass
            predictions = model(inputs)
            true_velocity = labels[:, :, 0]  # True velocity from labels
            
            # Store results for comparison
            true_velocities.append(true_velocity.item())
            predicted_velocities.append(predictions[0][0].item())
            print(f"True Velocity: {true_velocity.item()}, Predicted Velocity: {predictions[0][0].item()}")
    
    # Plot true vs. predicted velocities
    plt.plot(true_velocities, label="True Velocity")
    plt.plot(predicted_velocities, label="Predicted Velocity")
    plt.xlabel("Frame")
    plt.ylabel("Velocity")
    plt.title("True vs Predicted Velocity per Frame")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    intrinsic_matrix = torch.Tensor([[1912.5, 0, 640], [0, 1912.5, 360], [0, 0, 1]])
    frame_rate = 30
    model = PINNOpticalFlow(intrinsic_matrix, frame_rate)
    
    # Load model weights if available
    # model.load_state_dict(torch.load("pinn_model.pth"))
    
    # Load test dataset
    test_dataset = StreamDataset('FlowDataSet/test')
    
    # Run testing
    test_pinn(model, test_dataset)