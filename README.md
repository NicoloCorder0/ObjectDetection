# Object Detection con OpenCV e YOLOv3

Questo progetto implementa un sistema di riconoscimento oggetti utilizzando Python, OpenCV e YOLOv3. Il sistema permette agli utenti di caricare un'immagine tramite una interfaccia grafica e visualizza i risultati dell'analisi degli oggetti riconosciuti nell'immagine.

## Prerequisiti

Prima di eseguire il codice, assicurati di avere installato Python e le seguenti librerie:

- OpenCV
- Numpy
- Tkinter (generalmente incluso con Python)

## Installazione

Per installare le dipendenze necessarie, esegui il seguente comando:

```bash
pip install numpy opencv-python
pip install numpy
```
Se utilizzi Anaconda:
```bash
conda install -c conda-forge opencv
conda install numpy
```

## Struttura dei File

Assicurati di avere i seguenti file nella tua directory di progetto:

- "yolov3.weights" : File dei pesi del modello YOLOv3.
- "yolov3.cfg" : File di configurazione per YOLOv3.
-  "coco.names" : File contenente i nomi delle classi riconosciute da YOLOv3.

Questi file possono essere scaricati da fonti ufficiali o da repository GitHub che offrono modelli pre-addestrati YOLOv3 :

https://pjreddie.com/media/files/yolov3.weights

https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.cfg

https://github.com/pjreddie/darknet/blob/master/data/coco.names
## Esecuzione

Per avviare l'applicazione, esegui il file script da terminale o command prompt:

```bash
python Object.py
```
Una volta avviato lo script, ti verrà chiesto di selezionare un'immagine da analizzare attraverso una finestra di dialogo.

## Descrizione del Codice

**Import delle Librerie:**

```bash
import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk
```
**Disabilitazione della Finestra Root di Tkinter:**

```bash
root = Tk()
root.withdraw()
```
**Selezione dell'immagine e caricamento:**

```bash
file_path = filedialog.askopenfilename()
img = cv2.imread(file_path)
height, width, channels = img.shape
```
Apre una finestra che permette all'utente di selezionare un file dal proprio dispositivo, successivamente carica l'immagine usando OpenCV per ottenere le sue dimensioni e i canali

**Inizializzazione della rete neurale:**

```bash
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
```
Carica il modello YOLO, la configurazione del modello e i nomi delle classi di oggetti riconoscibili dai file precedentemente inseriti nella directory del progetto

```bash
layers = net.getLayerNames()
outLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(outLayers)
```
Determina gli strati di output della rete e prepare l'immagine creando un blob ("contenitore" per i dati dell'immagine) che poi passa attraverso la rete

**Rilevamento oggetti:**

```bash
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
```
Questo blocco di codice scorre ogni rilevamento fatto dalla rete, verifica la confidenza dell'oggetto rilevato e, se è maggiore del 50% (in questo caso), calcola la posizione del riquadro dell'oggetto

**Disegno dei riquadri e delle etichette sull'immagine:**

```bash
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[i], 2)
        cv2.putText(img, label, (x, y), cv2.FONT_ITALIC, 0.7, colors[i], 2)
```
Disegna i riquadri e le etichette (nomi degli oggetti) sull'immagine per ogni oggetto rilevato

**Mostra l'immagine:**

```bash
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
