import nibabel as nib
import matplotlib.pyplot as plt
import sys

img = nib.load(sys.argv[1])
data = img.get_fdata()

z = data.shape[2] // 2
plt.imshow(data[:, :, z].T, cmap="gray", origin="lower")
plt.title(f"Slice {z}, shape={data.shape}")
plt.axis("off")
plt.show()