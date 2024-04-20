"""
Aufgabe:
Hintergrund: Die Datentransformation ist ein wichtiger Schritt bei der Vorbereitung
von Daten für das Training von Deep-Learning-Modellen. Sie umfasst häufig die
Normalisierung, Erweiterung und Konvertierung von Daten in ein Format, das für
die Aufnahme in das Modell geeignet ist.

Beschreibung:
Verwenden Sie den MNIST-Datensatz, der aus Graustufenbildern von handgeschriebenen
Ziffern besteht.
Deine Aufgabe ist es, ein PyTorch-Skript zu schreiben, das den MNIST-Datensatz
lädt und mehrere Transformationen auf ihn anwendet.

Erforderliche Transformationen:
- Ändern Sie die Größe der Bilder von 28x28 auf 32x32 Pixel.
- Normalisieren Sie die Bilder so, dass ihre Pixelwerte einen Mittelwert von 0,5
und eine Standardabweichung von 0,5 haben.
- Horizontales Spiegeln der Bilder nach dem Zufallsprinzip mit einer Wahrscheinlichkeit von 0,5.
- Konvertieren Sie die Bilder in das Tensorformat.

Schritte:
1. Importieren Sie die notwendigen Bibliotheken von torch und torchvision.
2. Definieren Sie die Transformationen mit torchvision.transforms.
3. Laden Sie den MNIST-Datensatz aus torchvision.datasets und wenden Sie die
definierten Transformationen an.
4. Iterieren Sie über den Datensatz und zeigen Sie die ersten vier Bilder und
ihre Beschriftungen an, um zu überprüfen, ob die Transformationen korrekt angewendet wurden.

Hinweise:
Verwenden Sie transforms.Compose, um mehrere Transformationen miteinander zu verknüpfen.
Verwenden Sie transforms.ToTensor(), transforms.Normalize(), transforms.Resize(), und transforms.RandomHorizontalFlip() für die Transformationen.
Verwenden Sie matplotlib.pyplot, um die transformierten Bilder zu visualisieren.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# Transformationen
transformations = transforms.Compose([
    transforms.Resize(32),  # 32x32 pixels
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisierung
])

# Datensatz laden
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

# Hilfsfunktion, um später die Bilder anzuzeigen
def imshow(img):
    img = img / 2 + 0.5  # zurück normalisieren
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# einfach zufällige Bilder nehmen
dataiter = iter(train_loader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images[:4]))
print('Labels:', labels[:4])
