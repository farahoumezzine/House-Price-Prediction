import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
data = pd.read_csv('Housing_Price_Data.csv')

# Aperçu des données
print ("data head")
print(data.head())
print("data info")
print(data.info())
print("data describe")
print(data.describe())

# Visualisation des distributions
plt.figure(figsize=(15,10))
for i, col in enumerate(data.select_dtypes(include=['int64', 'float64']).columns):
    plt.subplot(3, 3, i+1)
    sns.histplot(data[col], kde=True)
plt.tight_layout()
plt.show()

# Visualisation des corrélations
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.show()

