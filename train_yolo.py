"""
Avfallspolisen – Träningsskript
Tränar en YOLOv8-modell på det egna datasetet för avfallssortering.

Användning:
    python train_yolo.py

Kräver att dataset.yaml och datamappen (dataset/) är korrekt uppsatta.
"""

from ultralytics import YOLO

# Ladda en förtränad YOLOv8-modell som startpunkt
modell = YOLO("yolov8n.pt")

# Träna modellen på vårt avfallssorteringsdataset
resultat = modell.train(
    data="dataset.yaml",   # Sökväg till dataset-konfigurationen
    epochs=100,            # Antal träningsepoker
    imgsz=640,             # Bildstorlekt (pixlar)
    batch=16,              # Antal bilder per batch
    name="waste_sorting_model",  # Namn på träningskörningen (sparas i runs/)
    patience=20,           # Tidigt stopp om ingen förbättring efter 20 epoker
    save=True,             # Spara bästa och sista vikter
    plots=True,            # Generera träningsdiagram
)

print("Träning klar!")
print(f"Bästa modellvikter sparade i: runs/detect/waste_sorting_model/weights/best.pt")
