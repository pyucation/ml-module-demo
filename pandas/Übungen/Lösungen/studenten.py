"""
Aufgaben:
    1. Daten Laden und Überprüfen:
        Lade den Datensatz in einen DataFrame.
        Überprüfe die ersten zehn Einträge im DataFrame und gib die Namen der Spalten aus.

    2. Datenfilterung:
        Filtere die Daten, um nur die Einträge von Studenten im Bachelorstudium anzuzeigen.
        Filtere die Daten nach Studenten, deren Abschlussnote besser als 2.0 ist.

    3. Daten Sortieren:
        Sortiere die Daten nach der Abschlussnote in aufsteigender Reihenfolge.
        Sortiere die Daten nach dem Alter in absteigender Reihenfolge.

    4. Gruppierung und Aggregation:
        Berechne den Durchschnitt der Abschlussnoten für jedes Studienfach.
        Finde das am häufigsten genannte Lieblingsfarbe unter allen Studenten.

    Zusatz: Datenvisualisierung:
        Verwende Matplotlib oder Seaborn, um die Verteilung der Abschlussnoten in einem Histogramm darzustellen.
"""
import pandas as pd


file_path = 'pandas/Übungen/studenten_datensatz.csv'
data = pd.read_csv(file_path, delimiter=';')

# 1.
# überprüfe die ersten zehn Einträge
print(data.head(10))

# gib die Namen der Spalten aus
print(data.columns)

# 2.
# filtere nur die Bachelorstudenten
bachelor_students = data[data['Abschluss'] == 'Bachelor']
print(bachelor_students)

# filtere Studenten mit einer Abschlussnote besser als 2.0
top_students = data[data['Abschlussnote'] < 2.0]
print(top_students)

# 3.
# sortiere die Daten nach der Abschlussnote in aufsteigender Reihenfolge
sorted_by_grade = data.sort_values(by='Abschlussnote')
print(sorted_by_grade)

# sortiere die Daten nach dem Alter in absteigender Reihenfolge
sorted_by_age = data.sort_values(by='Alter', ascending=False)
print(sorted_by_age)

# 4.
# berechne den Durchschnitt der Abschlussnoten für jedes Studienfach
average_grades_by_subject = data.groupby('Studienfach')['Abschlussnote'].mean()
print(average_grades_by_subject)

# finde das am häufigsten genannte Lieblingsfarbe
most_frequent_color = data['Lieblingsfarbe'].mode()[0]
print(f"Die am häufigsten genannte Lieblingsfarbe ist: {most_frequent_color}")


# ----- Zusatz -----
import matplotlib.pyplot as plt

data['Abschlussnote'].hist(bins=10)
plt.title('Verteilung der Abschlussnoten')
plt.xlabel('Abschlussnote')
plt.ylabel('Anzahl der Studenten')
plt.show()
