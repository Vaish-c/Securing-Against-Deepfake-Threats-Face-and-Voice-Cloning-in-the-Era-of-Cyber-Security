import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess

class DeepFakeDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Deep Fake Detection")
        
        self.description_box = tk.Text(self.master, height=10, width=50)
        self.description_box.grid(row=0, column=0, columnspan=3, padx=10, pady=10)
        
        self.create_button("AI Voice Detection", self.run_voice_detection, row=1, column=0)
        self.create_button("Fake Face Detection", self.run_fake_face_detection, row=1, column=1)
        self.create_button("Frame Capturing", self.run_frame_capturing, row=1, column=2)
    
    def create_button(self, text, command, row, column):
        button = tk.Button(self.master, text=text, command=command)
        button.grid(row=row, column=column, padx=5, pady=5)
    
    def run_voice_detection(self):
        
            result = subprocess.run(
                ["python",r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\AI FAKE VOICE DETECTION\soundclassinput.py"], 
                capture_output=True, 
                text=True
            )
            # Insert the AI voice detection results into the description box
            self.description_box.insert(tk.END, "AI Voice Detection Results:\n")
            self.description_box.insert(tk.END, result.stdout + "\n")
      
    
    def run_fake_face_detection(self):
        result = subprocess.run(
            ["python",r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\given\predict2.py"], 
            capture_output=True, 
            text=True
        )
        self.description_box.insert(tk.END, "Fake Face Detection Results:\n")
        self.description_box.insert(tk.END, result.stdout + "\n")
    
    def run_frame_capturing(self):
        result = subprocess.run(
            ["python", r"C:\Users\vaishnavi\onedrive_backup\Desktop\BE\projectFinal\deepfakemodules\captured_frames\framecapturing\frame3.py"], 
            capture_output=True, 
            text=True
        )
        self.description_box.insert(tk.END, "Frame Capturing Results:\n")
        self.description_box.insert(tk.END, result.stdout + "\n")
         
def main():
    root = tk.Tk()
    app = DeepFakeDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
