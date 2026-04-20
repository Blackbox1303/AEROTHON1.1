import cv2
import numpy as np

def get_tunnel_coordinates(image_path):
    # 1. Load image and convert to grayscale
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Thresholding: Make the dark tunnels white and the light page black
    # Using GaussianBlur to reduce noise from 'other things' on the page
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Find the shapes (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tunnel_coords = []

    for cnt in contours:
        # Filter by area to make sure we aren't picking up tiny dots or dust
        # Increase '500' if it picks up too much noise, decrease if it misses tunnels
        if cv2.contourArea(cnt) > 500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # Calculate the center point (X, Y) of the tunnel
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                tunnel_coords.append((cX, cY))

                # Optional: Draw for visualization
                cv2.circle(img, (cX, cY), 10, (0, 255, 0), -1)
                cv2.putText(img, f"({cX},{cY})", (cX + 15, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. Display and Return
    print(f"Found {len(tunnel_coords)} tunnel(s): {tunnel_coords}")
    cv2.imshow("Detected Tunnel Coordinates", img)
    cv2.waitKey(0)
    
    return tunnel_coords

# Usage:
# coordinates = get_tunnel_coordinates("your_image_here.jpg")
