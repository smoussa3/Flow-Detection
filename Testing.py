from dataset import StreamDataset
import dataset
from flow_detection import SVCalculator, ClassicalFlow, ClassicalFlowToSV
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = StreamDataset('FlowDataSet/train')
    video_index = 0

    flow_calc = ClassicalFlow()
    flow_to_sv = ClassicalFlowToSV(axis = 0)

    img_size = (720, 1080)

    crop_ratio = 0.0
    target_h = int(((img_size[0] * (1 - crop_ratio)) // 8) * 8)
    target_w = int(((img_size[1] * (1 - crop_ratio)) // 8) * 8)
    target_size = (target_h, target_w) # we center crop to make the image fit RAFT and to avoid major distortion effects

    # calculated from camera calibration
    intrinsic_matrix = torch.Tensor(
            [
            [1912.5, 0, 640],
            [0, 1912.5, 360],
            [0, 0, 1]
        ]
    )
    frame_rate = 30

    model = SVCalculator(flow_calc, flow_to_sv, intrinsic_matrix, frame_rate)

    # Predict Velocity
    true_velocities = []
    predicted_velocities = []

    video_index = 0 # Assume working with first video
    velocity, depth = dataset.get_video_metadata(video_index)
    depth = torch.Tensor([[depth]])

    for i in range(dataset.get_video_length(video_index) - 1):
        frame_one, frame_two = dataset.read_frame(video_index, i), dataset.read_frame(video_index, i + 1)
        x = torch.stack((frame_one, frame_two), dim = 0)
        x = torch.unsqueeze(x, dim = 0)

        predicted_velocity = model(x, depth = depth)
        print(f'True Velocity: {velocity}, Predicted: {predicted_velocity[0][0].item()}')

        true_velocities.append(velocity)
        predicted_velocities.append(predicted_velocity[0][0].item())

    #plotting velocities 
    plt.plot(true_velocities, label='True Velocity')
    plt.plot(predicted_velocities, label='Predicted Velocity')
    plt.xlabel('Frame')
    plt.ylabel('Velocity')
    plt.title('True vs Predicted Velocity per Frame')
    plt.legend()
    plt.show()