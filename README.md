#  Lane Detection

##  Aim

To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

---

## Learning Objective

* Understand each stage of image processing
* Learn how to build a complete computer vision pipeline
* Practice writing code in guided sections

**Important Instruction:**
 Write code **ONLY in places marked as `# Your Code Here`**
 Do NOT modify any other part of the code

---

##  Software Used

* Anaconda – Python 3.7
* Jupyter Notebook / VS Code
* OpenCV (cv2)
* NumPy
* Matplotlib

---

##  Algorithm & Explanation

---
##  Developed By

* **Name:** FARZANA M
* **Register No:** 212225040087

###  Step 1: Import Libraries

```
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

###  Step 2: Read the Image

```import cv2

image = cv2.imread("building.jpg")

if image is None:
    raise FileNotFoundError("Could not load 'building.jpg'. Check the file path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("Image loaded successfully!")
```

---

###  Step 3: Convert to Grayscale

```
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


```

---

###  Step 4: Display Images

```
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()

```
<img width="1136" height="693" alt="Screenshot 2026-11-24 150910" src="https://github.com/user-attachments/assets/5ada5842-2719-48d0-a4b6-a0d46b2bdeb9" />

---

###  Step 5: Thresholding

```
threshold = 150

ret, thresh = cv2.threshold(
    gray, threshold, 255, cv2.THRESH_BINARY
)

plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()

```
<img width="572" height="710" alt="Screenshot 2026-11-24 150922" src="https://github.com/user-attachments/assets/50a3ff38-f9f0-4f3d-9308-e231f12641fb" />



---

###  Step 6: Region of Interest (ROI)

```import cv2
import numpy
import matplotlib.pyplot as plt

height, width = thresh.shape

roi_vertices = numpy.array([[
    (int(0.1 * width), height),
    (int(0.45 * width), int(0.6 * height)),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.9 * width), height)
]], dtype=numpy.int32)

mask = numpy.zeros_like(thresh)

cv2.fillPoly(mask, roi_vertices, 255)

roi_masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 6))
plt.imshow(roi_masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()
```
<img width="532" height="711" alt="Screenshot 2026-11-24 150929" src="https://github.com/user-attachments/assets/05c67899-7b58-475b-95fe-80dda1f27921" />



### Step 7: Edge Detection (Canny)

```
edges = cv2.Canny(roi_masked, 50, 150)
plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()


```

<img width="542" height="747" alt="Screenshot 2026-11-24 150935" src="https://github.com/user-attachments/assets/02fa9066-fd30-4550-b533-9fea098444e0" />

---

###  Step 8: Gaussian Blur

```
smoothed = cv2.GaussianBlur(edges, (5, 5), 0)
plt.figure(figsize=(6, 6))
plt.imshow(smoothed, cmap="gray")
plt.title("Smoothed (Blurred) Edge Image")
plt.axis("off")
plt.show()

```
<img width="472" height="682" alt="Screenshot 2026-11-24 150942" src="https://github.com/user-attachments/assets/964e9caf-e88d-4cb1-ad6f-2e0d0173215d" />


---

###  Step 9: Hough Transform

```import cv2
import numpy as np
import matplotlib.pyplot as plt

lines = cv2.HoughLinesP(
    smoothed,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100
)

line_image = np.zeros_like(image)

if lines is not None:
    lines = np.array(lines).reshape(-1, 4)

    for x1, y1, x2, y2 in lines:
        cv2.line(
            line_image,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (255, 0, 0),
            5
        )

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(line_image_rgb)
plt.title("Detected Lines")
plt.axis("off")
plt.show()

```
<img width="526" height="710" alt="Screenshot 2026-11-24 150948" src="https://github.com/user-attachments/assets/69c26876-e71d-4b58-9653-0f87df48a2cd" />


---

### Step 10: Lane Detection Logic

```
final_output = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(6, 6))
plt.imshow(final_output_rgb)
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
```
<img width="485" height="702" alt="Screenshot 2026-11-24 150956" src="https://github.com/user-attachments/assets/890934ae-a3ae-429d-b31e-4482e477390d" />


---


## Result

Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.

---
