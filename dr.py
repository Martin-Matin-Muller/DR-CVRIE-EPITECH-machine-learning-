import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
import joblib

PREVIEW_MAX_SIDE = 360

def _confidence_from_model(model, img_flat: np.ndarray, prediction: int) -> tuple[float | None, bool]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(img_flat)[0]
        return float(proba[int(prediction)] * 100), False

    if hasattr(model, "decision_function"):
        scores = np.atleast_1d(model.decision_function(img_flat))
        if scores.ndim == 1:
            p = 1.0 / (1.0 + np.exp(-scores[0]))
            return float((p if int(prediction) == 1 else (1.0 - p)) * 100), True
        shifted = scores[0] - np.max(scores[0])
        exp_scores = np.exp(shifted)
        softmax = exp_scores / np.sum(exp_scores)
        return float(softmax[int(prediction)] * 100), True

    return None, True


def _build_preview_image(image_path: str, max_side: int = PREVIEW_MAX_SIDE) -> tk.PhotoImage:
    img = imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    h, w = img.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    preview = resize(
        img,
        (new_h, new_w),
        anti_aliasing=True,
        preserve_range=True,
    )
    preview_u8 = np.clip(preview * 255.0, 0, 255).astype(np.uint8)

    header = f"P6 {new_w} {new_h} 255\n".encode("ascii")
    return tk.PhotoImage(data=header + preview_u8.tobytes(), format="PPM")

def run() -> None:
    root = tk.Tk()
    root.title("Selection d'image")
    root.geometry("760x760")
    root.resizable(False, False)

    selected_image = tk.StringVar(value="Aucune image selectionnee")
    result_text = tk.StringVar(value="Résultat: --")
    
    model_path = Path(__file__).with_name("CVRIE_PNEUMONIA_RECOGNITION.pkl")

    try:
        model = joblib.load(model_path)
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
            try:
                preview = _build_preview_image(image_path)
                image_panel.configure(image=preview, text="")
                image_panel.image = preview
            except Exception as e:
                image_panel.configure(image="", text="Aperçu indisponible")
                image_panel.image = None
                messagebox.showwarning("Avertissement", f"Impossible d'afficher l'image: {e}")
    
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
            img = resize(img, (64, 64), anti_aliasing=True, preserve_range=True)
            img_flat = img.reshape(1, -1).astype(np.float32)

            prediction = model.predict(img_flat)[0]
            categories = ["NORMAL", "PNEUMONIA"]
            predicted_class = categories[int(prediction)]
            confidence, estimated = _confidence_from_model(model, img_flat, int(prediction))
            if confidence is None:
                result_text.set(f"Résultat: {predicted_class}")
            elif estimated:
                result_text.set(f"Résultat: {predicted_class} (score estimé {confidence:.1f}%)")
            else:
                result_text.set(f"Résultat: {predicted_class} ({confidence:.1f}%)")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {e}")

    # GUI
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

    image_panel = tk.Label(
        frame,
        text="Aucune image sélectionnée",
        width=PREVIEW_MAX_SIDE,
        height=PREVIEW_MAX_SIDE,
        relief="solid",
        bd=1,
        bg="#f7f7f7",
        anchor="center",
    )
    image_panel.pack(anchor="w", pady=(14, 0))

    root.mainloop()

if __name__ == "__main__":
    run()
