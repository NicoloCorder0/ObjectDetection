import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# Disabilita la finestra root di Tkinter
root = Tk()
root.withdraw()

# Chiedi all'utente di selezionare un file immagine
file_path = filedialog.askopenfilename()

# Carica l'immagine selezionata
img = cv2.imread(file_path)
height, width, channels = img.shape

# Inizializza la rete
net = cv2.dnn.readNet(r"yolov3.weights", r"yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers = net.getLayerNames()
outLayers = [layers[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Crea il blob dall'immagine
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(outLayers)

# Variabili per il riconoscimento
boxes = []
confidences = []
class_ids = []

# Rilevamento oggetti
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.7:
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

# Disegna i riquadri e le etichette
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[i], 2)
        cv2.putText(img, label, (x, y), cv2.FONT_ITALIC, 0.7, colors[i], 2)

# Mostra l'immagine
cv2.imshow("Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()