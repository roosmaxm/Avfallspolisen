# Avfallspolisen 🚔♻️

En Python-applikation som analyserar en videoström i realtid för att kontrollera om avfall är korrekt sorterat. Applikationen använder en tränad YOLOv8-modell tillsammans med OpenCV.

---

## Klassdefinitioner

| Klass | ID | Singular        | Plural          |
|-------|----|-----------------|-----------------|
| Dryckeskartong (mjölk, yoghurt, etc.) | 0 | Dryckeskartong | KARTONGER |
| Konservburk | 1 | Konservburk | KONSERVBURKAR |
| Pantburk | 2 | Pantburk | PANTBURKAR |

---

## Filstruktur

```
Avfallspolisen/
├── waste_sorting_app.py          # Huvudapplikationen
├── waste_sorting_notebook.ipynb  # Jupyter Notebook-version
├── train_yolo.py                 # Skript för att träna YOLO-modellen
├── evaluate_model.py             # Skript för kvantitativ & kvalitativ utvärdering
├── dataset.yaml                  # YOLO dataset-konfiguration
├── requirements.txt              # Python-beroenden
└── README.md                     # Projektdokumentation
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Användning

### Webbkamera (standard)

```bash
python waste_sorting_app.py
```

### Videofil

```bash
python waste_sorting_app.py --source video.mp4
```

Tryck `q` i OpenCV-fönstret för att avsluta applikationen.

---

## Sorteringslogik

Applikationen klassificerar varje detekterat objekt baserat på dess avstånd till andra objekt (tröskel: 150 pixlar):

| Status | Villkor | Färg | Visualisering |
|--------|---------|------|---------------|
| **SORTERAT** | Två eller fler objekt av **samma klass** ligger nära varandra | 🟢 Grön | Gruppnamn i plural visas EN gång ovanför gruppen (t.ex. `PANTBURKAR`) |
| **OSORTERAT** | Objekt av **olika klasser** ligger nära varandra | 🔴 Röd | Klassnamnet i singular vid varje objekt |
| **ENSAMT** | Inga andra objekt i närheten | 🟡 Gul | Klassnamnet i singular vid objektet |

Grupperingen använder **Union-Find (Disjoint Set Union)** för att korrekt hantera kedjor av nära objekt.

---

## Datasetstruktur

Skapa följande mappstruktur för träning och validering:

```
dataset/
├── images/
│   ├── train/     # Träningsbilder (.jpg, .png)
│   ├── val/       # Valideringsbilder
│   └── test/      # Testbilder
└── labels/
    ├── train/     # YOLO-annotationsfiler (.txt)
    ├── val/
    └── test/
```

Varje `.txt`-fil följer YOLO-format:
```
<class_id> <x_center> <y_center> <width> <height>
```
Alla koordinater är normaliserade (0–1) relativt bildens storlek.

---

## Träning

Träna modellen på ditt eget dataset:

```bash
python train_yolo.py
```

Den tränade modellen sparas under `runs/detect/waste_sorting_model/weights/best.pt`.
Kopiera `best.pt` till projektets rotkatalog för att använda den i applikationen.

---

## Utvärdering

Utvärdera modellens prestanda mot test-datasetet:

```bash
python evaluate_model.py
```

Skriptet:
- Beräknar **mAP50** och **mAP50-95** per klass
- Kör modellen på bilder i mappen `torture_test/` (om den finns) och sparar resultaten i `torture_results/`
- Genererar en **Confusion Matrix** (sparas i `runs/detect/val/`)

---

## Jupyter Notebook

Öppna notebooken för en interaktiv, steg-för-steg-genomgång:

```bash
jupyter notebook waste_sorting_notebook.ipynb
```

---

## Teknisk stack

| Komponent | Teknologi |
|-----------|-----------|
| Objektdetektering | [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) |
| Videohantering & visualisering | [OpenCV](https://opencv.org/) |
| Numeriska beräkningar | [NumPy](https://numpy.org/) |
| Deep learning-ramverk | [PyTorch](https://pytorch.org/) |
| Notebook-miljö | [Jupyter](https://jupyter.org/) |
| Språk | Python 3.8+ |
