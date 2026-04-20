import cv2
import numpy as np
import math

def process_tunnel_mission():
    image_path = r"C:\Users\Lenovo\OneDrive\Desktop\tunnel3.jpeg"
    img = cv2.imread(image_path)
    
    if img is None:
        print("Image not found.")
        return
    
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. THE NOISE KILLER: Median Blur
    # A large median blur (15) effectively 'paints over' bricks and small noise
    # while keeping the large tunnel shapes intact.
    blurred = cv2.medianBlur(gray, 15)

    # 2. Thresholding
    # Since the tunnels are the only thing 'standing out', we use Otsu 
    # to find the two main 'levels' of the image (Wall vs. Tunnel).
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Morphological Cleanup
    # One last 'Opening' to delete any leftover tiny speckles of noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 4. Find the Tunnels
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tunnels = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Only accept large shapes (tunnels)
        if area > 1000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                tunnels.append({'center': (cX, cY), 'box': (x, y, w, h)})

    # Debug: This should look like two very clean white blobs on a black background
    cv2.imshow("Cleaned Computer View", thresh)

    if len(tunnels) < 2:
        print(f"Still seeing {len(tunnels)} tunnels. The noise might be too thick.")
        cv2.waitKey(0)
        return

    # 5. Logic: Left to Right
    tunnels.sort(key=lambda t: t['center'][0])
    
    # Red Box for Entrance (Left)
    e = tunnels[0]['box']
    cv2.rectangle(output, (e[0], e[1]), (e[0]+e[2], e[1]+e[3]), (0, 0, 255), 5)
    
    # Green Box for Exit (Right)
    x = tunnels[1]['box']
    cv2.rectangle(output, (x[0], x[1]), (x[0]+x[2], x[1]+x[3]), (0, 255, 0), 5)

    cv2.imshow("Final Result (No Noise)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_tunnel_mission()
