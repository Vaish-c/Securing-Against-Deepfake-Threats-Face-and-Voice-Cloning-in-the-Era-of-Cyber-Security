import tkinter as tk
from tkinter import messagebox

def login():
    username = username_entry.get()
    password = password_entry.get()

    if username == "Admin" and password == "1234":
        messagebox.showinfo("Login Successful", "Welcome, Admin!")
        # Execute the specified code
        import subprocess
        subprocess.Popen(["python", "C:/Users/vaishnavi/onedrive_backup/Desktop/BE/projectFinal/deepfakemodules/GUI.py"])
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Create main window
root = tk.Tk()
root.title("Login")
root.geometry("300x150")

# Username label and entry
username_label = tk.Label(root, text="Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

# Password label and entry
password_label = tk.Label(root, text="Password:")
password_label.pack()
password_entry = tk.Entry(root, show="*")
password_entry.pack()

# Login button
login_button = tk.Button(root, text="Login", command=login)
login_button.pack()

root.mainloop()
