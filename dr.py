import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import joblib

def run() -> None:
    root = tk.Tk()
    root.title("Selection d'image")
    root.geometry("640x280")
    root.resizable(False, False)

    selected_image = tk.StringVar(value="Aucune image selectionnee")
    result_text = tk.StringVar(value="Résultat: --")
    
    try:
        model = joblib.load("CVRIE_PNEUNOMINA_RECOGNITION.pkl")
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible de charger le modèle: {e}")
        model = None

    def choose_image() -> None:
        image_path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if image_path:
            selected_image.set(image_path)
            result_text.set("Résultat: --")
    
    def predict_image() -> None:
        if not model:
            messagebox.showerror("Erreur", "Le modèle n'est pas chargé")
            return

        image_path = selected_image.get()
        if image_path == "Aucune image selectionnee":
            messagebox.showwarning("Erreur", "Choisis une image d'abord")
            return
        
        try:
            img = imread(image_path)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = resize(img, (64, 64))
            img_flat = img.reshape(1, -1).astype(np.float32)

            prediction = model.predict(img_flat)[0]
            proba = model.predict_proba(img_flat)[0]
            categories = ["NORMAL", "PNEUMONIA"]
            predicted_class = categories[int(prediction)]
            confidence = proba[int(prediction)] * 100
            result_text.set(f"Résultat: {predicted_class} ({confidence:.1f}%)")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {e}")

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Button(frame, text="Choisir une image", command=choose_image, width=28).pack(
        anchor="w", pady=(0, 8)
    )
    tk.Label(frame, textvariable=selected_image, wraplength=600, justify="left", fg="blue").pack(
        anchor="w", pady=(0, 12)
    )
    
    tk.Button(frame, text="Lancer le modèle", command=predict_image, width=28, bg="#4CAF50", fg="white").pack(
        anchor="w", pady=(0, 8)
    )
    tk.Label(frame, textvariable=result_text, wraplength=600, justify="left", font=("Arial", 12, "bold"), fg="darkgreen").pack(
        anchor="w"
    )

    root.mainloop()

if __name__ == "__main__":
    run()
