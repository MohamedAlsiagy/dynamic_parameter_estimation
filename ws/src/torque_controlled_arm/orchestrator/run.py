from orchestrator.gui import DatasetTrackerGUI
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetTrackerGUI(root)
    root.mainloop()