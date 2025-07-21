import os
import tkinter as tk
from tkinter import ttk

import src.texture_loader as texture_loader

def main():
    root = tk.Tk()
    root.title("MineTexture")
    root.geometry("400x200")
    root.resizable(False, False)

    paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    loader = texture_loader.VanillaTexture()
    try:
        versions = loader.get_versions()
    except Exception:
        versions = []

    frame_left = ttk.Frame(paned)
    paned.add(frame_left, weight=1)

    label = ttk.Label(frame_left, text="Vanilla texture:")
    label.pack(fill=tk.X, padx=10, pady=(10, 0))

    selected_version = tk.StringVar()
    combobox = ttk.Combobox(frame_left, textvariable=selected_version, values=versions, state='readonly')
    combobox.pack(fill=tk.X, padx=10, pady=4)

    frame_right = ttk.Frame(paned)
    paned.add(frame_right, weight=1)

    label = ttk.Label(frame_right, text="Styled texture:")
    label.pack(fill=tk.X, padx=10, pady=(10, 0))

    styled_texture_path = tk.StringVar()
    styled_path_textbox = ttk.Entry(frame_right, textvariable=styled_texture_path, state='readonly')
    styled_path_textbox.pack(fill=tk.X, padx=10, pady=4)

    styled_path_button = ttk.Button(frame_right, text="Set path to texturepack")
    styled_path_button.pack(fill=tk.X, padx=10, pady=4)

    sep = ttk.Separator(root, orient=tk.HORIZONTAL)
    sep.pack(fill=tk.X, padx=5, pady=10)

    button_frame = ttk.Frame(root)
    button_frame.pack(pady=(0,10))

    train_button = ttk.Button(button_frame, text="Train Model")
    load_button = ttk.Button(button_frame, text="Load Model")
    apply_button = ttk.Button(button_frame, text="Apply on texturepack")

    train_button.pack(side=tk.LEFT, padx=5)
    load_button.pack(side=tk.LEFT, padx=5)
    apply_button.pack(side=tk.LEFT, padx=5)

    status_label = ttk.Label(root, text="Ready.")
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    root.mainloop()


    #texture = texture_loader.VanillaTexture()
    #versions = texture.get_versions()

    #texture.download('1.21.8')
    pass

if __name__ == "__main__":
    main()
