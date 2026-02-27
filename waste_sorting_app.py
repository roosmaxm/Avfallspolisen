"""
Avfallspolisen – Huvudapplikation
Analyserar en videoström och kontrollerar om avfall är korrekt sorterat
med hjälp av en tränad YOLO-modell och OpenCV.

Användning:
    python waste_sorting_app.py                      # Webbkamera (index 0)
    python waste_sorting_app.py --source video.mp4   # Videofil
"""

import argparse
import sys
import warnings

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Avståndströskel i pixlar – objekt inom detta avstånd anses vara "nära"
PROXIMITY_THRESHOLD = 150

# Klassnamn: singular används vid enskilda/osorterade objekt,
# plural används som grupptext för sorterade objekt
CLASS_NAMES = {
    0: {"singular": "Dryckeskartong", "plural": "KARTONGER"},
    1: {"singular": "Konservburk",    "plural": "KONSERVBURKAR"},
    2: {"singular": "Pantburk",       "plural": "PANTBURKAR"},
}

# Färger i BGR-format (OpenCV)
COLOR_SORTED   = (0, 255, 0)    # Grön  – korrekt sorterat
COLOR_UNSORTED = (0, 0, 255)    # Röd   – felosorterat
COLOR_ALONE    = (0, 255, 255)  # Gul   – ensamt objekt

# ---------------------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------------------


def berakna_centroid(box):
    """Beräknar centroiden (mittpunkten) för en bounding box.

    Args:
        box: Sekvens med fyra värden [x1, y1, x2, y2].

    Returns:
        Tupel (cx, cy) med centroidens koordinater som heltal.
    """
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def berakna_avstand(punkt1, punkt2):
    """Beräknar det euklidiska avståndet mellan två punkter.

    Args:
        punkt1: Tupel (x, y).
        punkt2: Tupel (x, y).

    Returns:
        Flyttalsavståndet mellan punkterna.
    """
    return float(np.sqrt((punkt1[0] - punkt2[0]) ** 2 + (punkt1[1] - punkt2[1]) ** 2))


# ---------------------------------------------------------------------------
# Union-Find (Disjoint Set Union) för gruppering av nära objekt
# ---------------------------------------------------------------------------


class UnionFind:
    """Enkel Union-Find-struktur för att gruppera objekt."""

    def __init__(self, n):
        self.parent = list(range(n))

    def hitta(self, x):
        """Hitta roten för element x med sökvägskomprimering."""
        if self.parent[x] != x:
            self.parent[x] = self.hitta(self.parent[x])
        return self.parent[x]

    def forena(self, x, y):
        """Förena de grupper som x respektive y tillhör."""
        rx, ry = self.hitta(x), self.hitta(y)
        if rx != ry:
            self.parent[rx] = ry


def bestam_status(detektioner):
    """Bestämmer sorteringsstatus för alla detekterade objekt.

    Algoritmen:
    1. Beräkna centroider för alla objekt.
    2. Jämför avstånd mellan alla par.
    3. Använd Union-Find för att bygga grupper av nära objekt.
    4. Klassificera varje grupp:
       - Ensam (ett objekt) → ALONE
       - Alla samma klass   → SORTED
       - Blandade klasser   → UNSORTED

    Args:
        detektioner: Lista med dicts {box, class_id, conf}.

    Returns:
        Lista med dicts {box, class_id, conf, status, grupp_id}.
        Möjliga statusvärden: "sorted", "unsorted", "alone".
    """
    n = len(detektioner)
    if n == 0:
        return []

    # Beräkna centroider
    centroider = [berakna_centroid(d["box"]) for d in detektioner]

    # Union-Find för att gruppera objekt som är nära varandra
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            avstand = berakna_avstand(centroider[i], centroider[j])
            if avstand <= PROXIMITY_THRESHOLD:
                uf.forena(i, j)

    # Bygg grupp-mappning: rot → lista av index
    grupper = {}
    for i in range(n):
        rot = uf.hitta(i)
        grupper.setdefault(rot, []).append(i)

    # Klassificera varje grupp
    resultat = []
    for i, det in enumerate(detektioner):
        rot = uf.hitta(i)
        grupp_index = grupper[rot]
        klasser_i_grupp = {detektioner[j]["class_id"] for j in grupp_index}

        if len(grupp_index) == 1:
            status = "alone"
        elif len(klasser_i_grupp) == 1:
            status = "sorted"
        else:
            status = "unsorted"

        resultat.append({
            **det,
            "status": status,
            "grupp_id": rot,
            "centroid": centroider[i],
        })

    return resultat


# ---------------------------------------------------------------------------
# Visualisering
# ---------------------------------------------------------------------------


def rita_pa_frame(frame, resultat):
    """Ritar bounding boxes och text på en frame baserat på sorteringsstatus.

    Regler:
    - SORTERAT   → grön box, gruppens pluralnamn ritas EN gång ovanför gruppen
    - OSORTERAT  → röd box, singularnamn vid varje objekt
    - ENSAMT     → gul box, singularnamn vid objektet

    Args:
        frame:    NumPy-array (BGR) att rita på.
        resultat: Lista från `bestam_status`.

    Returns:
        Frame med ritade annotationer.
    """
    # Samla sorterade grupper för att rita grupptext en gång
    sorterade_grupper = {}  # grupp_id → lista av dets

    for det in resultat:
        status = det["status"]
        box = det["box"]
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        class_id = det["class_id"]
        klassinfo = CLASS_NAMES.get(class_id, {"singular": str(class_id), "plural": str(class_id)})

        if status == "sorted":
            farg = COLOR_SORTED
            cv2.rectangle(frame, (x1, y1), (x2, y2), farg, 2)
            grupp_id = det["grupp_id"]
            sorterade_grupper.setdefault(grupp_id, []).append(det)

        elif status == "unsorted":
            farg = COLOR_UNSORTED
            cv2.rectangle(frame, (x1, y1), (x2, y2), farg, 2)
            text = klassinfo["singular"]
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, farg, 2)

        else:  # alone
            farg = COLOR_ALONE
            cv2.rectangle(frame, (x1, y1), (x2, y2), farg, 2)
            text = klassinfo["singular"]
            cv2.putText(frame, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, farg, 2)

    # Rita grupptext EN gång per sorterad grupp, ovanför högsta boxen i gruppen
    for grupp_id, dets in sorterade_grupper.items():
        # Alla objekt i gruppen har samma klass (annars vore de "unsorted")
        class_id = dets[0]["class_id"]
        klassinfo = CLASS_NAMES.get(class_id, {"singular": str(class_id), "plural": str(class_id)})
        plural_text = klassinfo["plural"]

        # Hitta bounding-box för hela gruppen
        alla_x1 = [int(d["box"][0]) for d in dets]
        alla_y1 = [int(d["box"][1]) for d in dets]
        grupp_x1 = min(alla_x1)
        grupp_y1 = min(alla_y1)

        cv2.putText(frame, plural_text, (grupp_x1, grupp_y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_SORTED, 2)

    return frame


# ---------------------------------------------------------------------------
# Huvudloop
# ---------------------------------------------------------------------------


def ladda_modell():
    """Laddar YOLO-modellen. Försöker först best.pt, sedan yolov8n.pt."""
    try:
        modell = YOLO("best.pt")
        print("Modell laddad: best.pt")
    except Exception:
        warnings.warn(
            "Kunde inte ladda 'best.pt'. Använder 'yolov8n.pt' som reserv. "
            "Träna en modell med train_yolo.py för bästa resultat.",
            UserWarning,
            stacklevel=2,
        )
        modell = YOLO("yolov8n.pt")
        print("Modell laddad: yolov8n.pt (reserv)")
    return modell


def kör_applikation(kalla):
    """Startar videoanalysen och visar resultatet i ett OpenCV-fönster.

    Args:
        kalla: Videokälla – 0 för webbkamera eller sökväg till videofil.
    """
    modell = ladda_modell()

    cap = cv2.VideoCapture(kalla)
    if not cap.isOpened():
        print(f"Fel: Kunde inte öppna videokällan '{kalla}'.", file=sys.stderr)
        sys.exit(1)

    print("Tryck 'q' för att avsluta.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Videoströmmen är slut eller kunde inte läsas.")
            break

        # Kör YOLO-inferens
        inference_resultat = modell(frame, verbose=False)

        # Extrahera detektioner
        detektioner = []
        for r in inference_resultat:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                # Visa bara klasser som vi känner till
                if class_id in CLASS_NAMES:
                    detektioner.append({
                        "box": xyxy,
                        "class_id": class_id,
                        "conf": conf,
                    })

        # Bestäm sorteringsstatus
        resultat = bestam_status(detektioner)

        # Rita annotationer på framen
        annoterad_frame = rita_pa_frame(frame, resultat)

        cv2.imshow("Avfallspolisen", annoterad_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Kommandoradshantering
# ---------------------------------------------------------------------------


def main():
    """Parsar kommandoradsargument och startar applikationen."""
    parser = argparse.ArgumentParser(
        description="Avfallspolisen – analyserar videoström för avfallssortering"
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Videokälla: 0 för webbkamera (standard) eller sökväg till videofil",
    )
    args = parser.parse_args()

    # Konvertera till heltal om det är ett kameranummer
    kalla = args.source
    try:
        kalla = int(kalla)
    except (ValueError, TypeError):
        pass  # Behåll som sträng (filsökväg)

    kör_applikation(kalla)


if __name__ == "__main__":
    main()
