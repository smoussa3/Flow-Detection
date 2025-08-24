import torch
import torch.nn as nn
from PINN import PINNOpticalFlow

def main():
    # Training settings
    path_to_train = "FlowDataset/train"  # Path to training data
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 10
    model_save_path = "pinn_model.pth"

    # Initialize model, optimizer, and loss function
    model = PINNOpticalFlow()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loss_fn = nn.MSELoss()

    # Load training data
    train_loader = PINNOpticalFlow.get_data_loader(path_to_train, batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0.0
        data_loss_total = 0.0
        physics_loss_total = 0.0

        for frame_pairs, labels in train_loader:
            # Extract velocity, depth, and metadata
            velocity = labels[:, :, 0]  # Ground truth velocity
            depth = labels[:, :, 1]    # Depth values
            metadata = labels[:, :, 2:]  # Metadata (mp and xp)

            # Forward pass
            predictions = model((frame_pairs, depth))

            # Compute data loss
            data_loss = data_loss_fn(predictions, velocity)

            # Compute physics-informed loss
            physics_loss = model.physics_loss(predictions, velocity, metadata)

            # Total loss: data + physics (weighted)
            total_loss_combined = data_loss + 0.1 * physics_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss_combined.backward()
            optimizer.step()

            # Track losses
            total_loss += total_loss_combined.item()
            data_loss_total += data_loss.item()
            physics_loss_total += physics_loss.item()

        # Log progress for the epoch
        avg_total_loss = total_loss / len(train_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        avg_physics_loss = physics_loss_total / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        print(f"  Data Loss: {avg_data_loss:.4f}")
        print(f"  Physics Loss: {avg_physics_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    main()