import torch
import torchvision
import torchvision.transforms as transforms

# Transformationskette erstellen: ToTensor -> Bilder zu Tensor,
# Normalize -> von [0, 255] RGB zu [0, 1] RGB
# Warum?
# - Konvergenzgeschwindigkeit von CNNs erhöht
# - verschiedene Skalierungen der Bilder beseitigen
# - numerische Stabilität erhöhen (Overflows, Underflow)
# - speziell für Bilder: verschiedene Lichtbedingungen ausgleichen
# - Modell ist weniger sensitiv für die Skalierung der Bilder und generalisiert besser
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Trainingsdaten laden
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# Testdaten laden
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

# sehr einfaches neuronales Netzwerk
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Kostenfunktion und Optimierungsalgorithmus wählen
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training
for epoch in range(2): # Epochen = Trainingszyklen
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # damit wir nicht die Gradienten der verschiedenen Batches mischen
        optimizer.zero_grad()
        # Vorwärtsrichtung (Forward Pass) berechnen = Vorhersagen
        outputs = model(inputs)
        # Kosten berechnen
        loss = criterion(outputs, labels)
        # Rückwärtsrichtung berechnen (Backpropagation)
        loss.backward()
        # Gewichte updaten
        optimizer.step()
    print('Finished Training')

# Testing
model.eval()
correct = 0
total = 0
with torch.no_grad(): # keine Gradienten berechnen
    for images, labels in testloader:
        outputs = model(images)
        # Vorhersage = der "wahrscheinlichste Wert" am Ausgang
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy of our model: {accuracy}")
