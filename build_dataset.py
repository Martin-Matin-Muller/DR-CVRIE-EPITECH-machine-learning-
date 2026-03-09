import os
import numpy as np

ROOT = "archive/chest_xray"   # change if needed


def load_pgm(path):
    with open(path, "rb") as f:
        header = f.readline().strip()
        if header != b"P5":
            raise ValueError("Not PGM")

        # skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        width, height = map(int, line.split())
        maxval = int(f.readline())

        data = f.read()

    img = np.frombuffer(data, dtype=np.uint8)
    return img.reshape((height, width))


X = []
y = []

for label in os.listdir(ROOT):
    label_dir = os.path.join(ROOT, label)
    if not os.path.isdir(label_dir):
        continue

    print("Loading class:", label)

    for file in os.listdir(label_dir):
        if file.endswith(".pgm"):
            path = os.path.join(label_dir, file)
            img = load_pgm(path)

            X.append(img.reshape(-1))   # flatten
            y.append(label)

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Saved dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)