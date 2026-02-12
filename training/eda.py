import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import random

DATASET_PATH = "dataset/"

def plot_image_counts():
    categories = os.listdir(DATASET_PATH)

    image_counts = {}
    for category in categories:
        path = os.path.join(DATASET_PATH, category)
        if os.path.isdir(path):
            image_counts[category] = len(os.listdir(path))

    df_counts = pd.DataFrame(list(image_counts.items()), columns=["Category", "Count"])
    print(df_counts)

    plt.figure(figsize=(10, 6))
    plt.bar(df_counts["Category"], df_counts["Count"])
    plt.xticks(rotation=45)
    plt.title("Number of Images per Category")
    plt.show()


def show_sample_images():
    categories = os.listdir(DATASET_PATH)
    plt.figure(figsize=(15, 10))

    for i, category in enumerate(categories[:6]):
        path = os.path.join(DATASET_PATH, category)
        if os.path.isdir(path):
            random_image_name = random.choice(os.listdir(path))
            img_path = os.path.join(path, random_image_name)

            plt.subplot(2, 3, i + 1)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(category)
            plt.axis("off")

    plt.show()


if __name__ == "__main__":
    plot_image_counts()
    show_sample_images()
