import matplotlib.pyplot as plt
import common.settings as settings
import common.data_preparation as data_prep

from IMPROVED_UNETPP.model import ImprovedUNetPlusPlus, val_extra_transforms
ModelClass = ImprovedUNetPlusPlus

NUM_SAMPLES = 5


def main():
    test_loader = data_prep.get_test_loader(
        extra_transforms=val_extra_transforms,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS
    )

    model = ModelClass()
    model.load(ModelClass.get_model_save_path())
    print(f"Модель загружена из {ModelClass.get_model_save_path()}")

    vis_images = []
    vis_masks = []
    for imgs, masks in test_loader:
        for i in range(len(imgs)):
            if len(vis_images) >= NUM_SAMPLES:
                break
            vis_images.append(imgs[i])
            vis_masks.append(masks[i])
        if len(vis_images) >= NUM_SAMPLES:
            break

    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    if NUM_SAMPLES == 1:
        axes = [axes]

    for i in range(NUM_SAMPLES):
        ax_img = axes[i][0]
        ax_truth = axes[i][1]
        ax_pred = axes[i][2]
        model.visualize_sample(vis_images[i], vis_masks[i], ax_img, ax_truth, ax_pred)
        ax_img.set_title("Image")
        ax_truth.set_title("Ground truth")
        ax_pred.set_title("Prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()