# SEC-DIP--Coin-Detection-using-OpenCV-in-Python
NAME : R VENKATRAMANI
REG NO: 212225240182

#PROGRAM
```

import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "coin.jpg"

# Load image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

coin_count = 0

for contour in contours:

    area = cv2.contourArea(contour)

    # Remove small noise
    if area < 100:
        continue

    # Find enclosing circle
    ((x, y), radius) = cv2.minEnclosingCircle(contour)

    # Filter small objects
    if radius > 15:
        coin_count += 1

        # Draw circle
        cv2.circle(
            output,
            (int(x), int(y)),
            int(radius),
            (0, 255, 0),
            2
        )

        # Number the coin
        cv2.putText(
            output,
            str(coin_count),
            (int(x - 10), int(y + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

print("Total coins detected:", coin_count)

# Display all stages
plt.figure(figsize=(20, 5))

plt.subplot(1, 5, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(edges, cmap="gray")
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Coins: {coin_count}")
plt.axis("off")

plt.tight_layout()
plt.show()


```
#IMAGE

<img width="1298" height="308" alt="Screenshot 2026-09-16 132417" src="https://github.com/user-attachments/assets/44ac651f-d749-41d5-a241-99e2041d7367" />




