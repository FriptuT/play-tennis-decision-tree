from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



# Setul de date pentru arborele decizional
set_de_date = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

# Etichetele pentru caracteristicile setului de date
caracteristici = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis']

# Extragem caracteristicile și etichetele din setul de date
X = [row[:-1] for row in set_de_date]  # Caracteristicile
y = [row[-1] for row in set_de_date]  # Etichetele



# Transformăm caracteristicile în valori numerice pentru a putea fi procesate de algoritmul de decizie
le = LabelEncoder()
X_encoded = [le.fit_transform(xi) for xi in zip(*X)] #zip este util pt a itera prin col din X
y_encoded = le.fit_transform(y)  #avand doar 1 coloana per row nu mai e necesara functia zip


# Creăm un clasificator de arbore decizional și îl antrenăm pe setul nostru de date
clf = DecisionTreeClassifier()
clf = clf.fit(list(zip(*X_encoded)), y_encoded)



# Afisăm arborele decizional generat

plt.figure(figsize=(10,6))
tree.plot_tree(clf, feature_names=caracteristici[:-1], class_names=le.classes_, filled=True)
plt.show()
