"""
Hintergrund: CIFAR-10 ist ein Satz von 60.000 32x32-Farbbildern in 10 Klassen,
mit 6.000 Bildern pro Klasse. Er wird häufig zum Trainieren von Algorithmen für maschinelles Lernen und Computer Vision verwendet.

Beschreibung:
Konstruieren Sie ein CNN und trainieren Sie es, um Bilder aus dem CIFAR-10-Datensatz
zu klassifizieren.
Evaluieren Sie die Leistung des Modells auf dem Testsatz.

Modell-Anforderungen:
- Das CNN sollte mindestens zwei Faltungsschichten haben, gefolgt von Max-Pooling-Schichten.
- Verwenden Sie ReLU-Aktivierungsfunktionen und eine abschließende vollständig verbundene Schicht.
- Wenden Sie eine geeignete Verlustfunktion und einen Optimierer Ihrer Wahl an.

Schritte:
1. Importieren Sie die erforderlichen Module(torch, torch.nn, torch.optim,
torchvision.datasets, torchvision.transforms).
2. Definieren Sie die CNN-Architektur in einer Klasse, die von torch.nn.Module erbt.
3. Laden Sie CIFAR-10 mithilfe von torchvision.datasets.CIFAR10 mit geeigneten
4. Transformationen (Normalisierung auf den Bereich [0,1]).
5. Teilen Sie den Datensatz mit torch.utils.data.DataLoader in Trainings- und Teststapel auf.
6. Trainieren Sie das Modell für mehrere Epochen und verfolgen Sie den Verlust und die Genauigkeit.
7. Beurteilen Sie das Modell anhand der Testdaten und geben Sie die Genauigkeit aus.

Hinweise:
Verwenden Sie einen Lernraten-Scheduler, wenn das Modell mit der Konvergenz kämpft.
Erwägen Sie die Verwendung von Dropout oder Batch-Normalisierung, um die Modellgeneralisierung zu verbessern.
Experimentieren Sie mit verschiedenen Architekturen und Hyperparametern.
"""