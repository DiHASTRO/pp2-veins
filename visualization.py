import matplotlib.pyplot as plt

import common.settings as settings
import common.data_preparation as data_prep

from CONSISTENT_TVERSKY_V1.model import (
    ConsistentTverskyDeepLabV3Plus,
    train_extra_transforms,
    val_extra_transforms,
)

ModelClass = ConsistentTverskyDeepLabV3Plus

FOLD_NUM = 1
NUM_SAMPLES = 5


def main():
    loaders = data_prep.create_cross_val_loaders(
        train_extra_transforms=train_extra_transforms,
        val_extra_transforms=val_extra_transforms,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS,
    )

    _, test_loader = loaders[FOLD_NUM - 1]

    model = ModelClass()
    model_path = ModelClass.get_model_save_path(FOLD_NUM)
    model.load(model_path)

    print(f"Модель загружена из {model_path}")

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

        model.visualize_sample(
            vis_images[i],
            vis_masks[i],
            ax_img,
            ax_truth,
            ax_pred,
        )

    plt.tight_layout()
    plt.savefig(f"visualization_fold_{FOLD_NUM}.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()