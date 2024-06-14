import cv2
import os
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to select a video file using a file dialog
def select_video_file():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
    return file_path

# Select a video file
video_file = select_video_file()

if not video_file:
    print("No video file selected. Exiting.")
    exit()

# Open the selected video file
cap = cv2.VideoCapture(video_file)

# Create a folder named 'captured_frames' 
output_folder = 'captured_frames'
os.makedirs(output_folder, exist_ok=True)

# Set the desired frame rate (frames per second)
frame_rate = 30  # Adjust this according to your video's frame rate

# Calculate the frame interval to achieve a 2-second gap
frame_interval = int(frame_rate * 2)

# Capture frames with a 2-second gap
frame_count = 0
while True:
    # Skip frames to achieve the desired gap
    for _ in range(frame_interval):
        ret, _ = cap.read()  # Read and discard frames
    
    # Capture a frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret or frame is None or frame.size == 0:
        print("End of video or error in capturing frames.")
        break

    frame_count += 1

    # Display the frame using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'Video Frame - {frame_count}')
    plt.show()

    # Save the frame as an image in the 'captured_frames' folder
    frame_filename = os.path.join(output_folder, f'frame_{frame_count}.png')
    cv2.imwrite(frame_filename, frame)
    print(f"Frame {frame_count} saved as {frame_filename}")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

print(f"Total frames captured: {frame_count}")
