"""
Avfallspolisen – Utvärderingsskript
Utvärderar den tränade YOLO-modellen kvantitativt och kvalitativt.

Användning:
    python evaluate_model.py
"""

import os

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Ladda modellen
# ---------------------------------------------------------------------------

print("Laddar modell...")
try:
    modell = YOLO("best.pt")
    print("Modell laddad: best.pt")
except Exception:
    print("Varning: 'best.pt' hittades inte. Använder 'yolov8n.pt' som reserv.")
    modell = YOLO("yolov8n.pt")

# ---------------------------------------------------------------------------
# 1. Kvantitativ utvärdering – mAP-värden mot test-datasetet
# ---------------------------------------------------------------------------

print("\n--- Kvantitativ utvärdering (mAP) ---")
val_resultat = modell.val(
    data="dataset.yaml",   # Konfiguration med test-split
    split="test",          # Kör mot test-datan
    plots=True,            # Spara diagram och Confusion Matrix
)

print(f"\nmAP50:    {val_resultat.box.map50:.4f}")
print(f"mAP50-95: {val_resultat.box.map:.4f}")

# Per-klass-resultat
print("\nPer-klass mAP50:")
for i, klass_namn in enumerate(["Dryckeskartong", "Konservburk", "Pantburk"]):
    if i < len(val_resultat.box.ap50):
        print(f"  {klass_namn}: {val_resultat.box.ap50[i]:.4f}")

# ---------------------------------------------------------------------------
# 2. Kvalitativ utvärdering – kör modellen på "tortyr-bilder"
# ---------------------------------------------------------------------------

TORTURE_MAPP = "torture_test"
UTDATA_MAPP  = "torture_results"

if os.path.isdir(TORTURE_MAPP):
    os.makedirs(UTDATA_MAPP, exist_ok=True)
    print(f"\n--- Kvalitativ utvärdering (tortyr-bilder från '{TORTURE_MAPP}/') ---")

    bildfilter = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    bildlista = [
        f for f in os.listdir(TORTURE_MAPP)
        if f.lower().endswith(bildfilter)
    ]

    if not bildlista:
        print(f"Inga bilder hittades i '{TORTURE_MAPP}/'.")
    else:
        for bildnamn in bildlista:
            bildsokvag = os.path.join(TORTURE_MAPP, bildnamn)
            bild = cv2.imread(bildsokvag)
            if bild is None:
                print(f"  Kunde inte läsa: {bildnamn}")
                continue

            # Kör inferens
            inferens = modell(bild, verbose=False)

            # Spara bild med annotationer
            annoterad = inferens[0].plot()
            utsokvag = os.path.join(UTDATA_MAPP, bildnamn)
            cv2.imwrite(utsokvag, annoterad)
            print(f"  Resultat sparat: {utsokvag}")

        print(f"\nAlla tortyr-resultat sparade i '{UTDATA_MAPP}/'.")
else:
    print(f"\nObs: Mappen '{TORTURE_MAPP}/' finns inte. "
          "Skapa den och lägg in testbilder för kvalitativ utvärdering.")

# ---------------------------------------------------------------------------
# 3. Confusion Matrix visas automatiskt av model.val() när plots=True
# ---------------------------------------------------------------------------

print("\nUtvärdering klar! Confusion Matrix och diagram sparades i runs/detect/val/.")
