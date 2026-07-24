import os
import torch
import matplotlib.pyplot as plt

# Import your custom modules from your scripts folder
from DataLoading import CityScapesDataset, CityscapesDataLoader,train_dataloader,val_dataloader , train_dataset , val_dataset
from UNetModel import U_Net

# 1. Setup paths and parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_VAL_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\val\img"
LABEL_VAL_DIR=r"E:\DeepLearningToGo\CityScapes\data\cityscapes_dataset\val\label"
CHECKPOINT_PATH = r"utils\model.pth"

def main():
    # 2. Load Validation Data
    val_dataset = CityScapesDataset(IMAGE_VAL_DIR, LABEL_VAL_DIR)
    val_loader = CityscapesDataLoader(val_dataset, batch_size=1, shuffle=True).get_loader()

    # 3. Instantiate Model & Load Saved Weights
    model = U_Net(in_channels=3, num_classes=19).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Successfully loaded weights from {CHECKPOINT_PATH}")
    else:
        print("Checkpoint not found! Running with untrained weights.")

    model.eval()

    # 4. Grab ONE image batch
    images, masks = next(iter(val_loader))
    images, masks = images.to(DEVICE), masks.to(DEVICE)

    # 5. Predict
    with torch.no_grad():
        logits = model(images)  # Shape: [1, 19, H, W]
        preds = torch.argmax(logits, dim=1)  # Shape: [1, H, W]

    # 6. Prepare Tensors for Matplotlib Plotting
    img_plot = images[0].cpu().permute(1, 2, 0).numpy()
    gt_plot = masks[0].cpu().numpy()
    pred_plot = preds[0].cpu().numpy()

    # 7. Plot Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_plot)
    axes[0].set_title("Input Street Image")
    axes[0].axis("off")

    axes[1].imshow(gt_plot, cmap="jet")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_plot, cmap="jet")
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("prediction_result.png")
    print("Saved plot to 'prediction_result.png'")
    plt.show()

if __name__ == "__main__":
    main()