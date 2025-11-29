#!/usr/bin/env python
# coding: utf-8

# # **Section 0 : Importer les bibliotheques, charger et inspecter les donnees**

# In[61]:


# On importe les bibliothèques nécessaire
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                           average_precision_score, accuracy_score, precision_score,
                           recall_score, f1_score, confusion_matrix, classification_report,
                           ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

print("Nos Bibliothèques de modélisation avancée importées avec succès !")


# ### **Configuration de l'affichage**

# In[62]:


pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


# In[63]:


# Random state pour la reproductibilité
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ### **Chargement des données**

# In[ ]:


from google.colab import files
uploaded = files.upload()


# In[64]:


# Lecture des données
data = pd.read_csv("creditcard.csv", on_bad_lines='skip', engine='python')
print(f"Shape du dataset: {data.shape}")

# Affichage des premières lignes
print("\nAperçu des données:")
display(data.head())


# In[ ]:





# - V1–V28 : Variables issues d’une transformation PCA permettant d’anonymiser les données tout en conservant la structure utile pour la détection de fraudes.
# - Amount : montant de la transaction
# - Time : temps écoulé en secondes entre chaque transaction et la première transaction du dataset.
# - Class : indique si la transaction est frauduleuse ou non

# ### **Traduire les noms des variables en français**

# In[65]:


# Les colonnes de V1 à V28 représentent des dimensions mathématiques permettant de capturer les comportements transactionnels typiques et atypiques.
# Nous allons juste traduire les colonne Time, Amount et Class en français
column_translations = {
    'Time': 'Temps_ecoule_sec',
    'Amount': 'Montant_Transaction',
    'Class': 'Classe'
}
data.rename(columns=column_translations, inplace=True)

print("Noms des colonnes après traduction:")
print(data.columns.tolist())


# ### **Inspection basique**

# In[66]:


print("INSPECTION GÉNÉRALE DU DATASET")
print("-"*30)

print(f"Dimensions: {data.shape[0]} lignes × {data.shape[1]} colonnes")
print(f"Types de données:")
print(data.dtypes.value_counts())


# In[67]:


print(f"Informations générales:")
data.info()


# In[ ]:





# ### **Vérification des valeurs manquantes**

# In[68]:


missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100

missing_info = pd.DataFrame({
    'Valeurs_Manquantes': missing_values,
    'Pourcentage': missing_percentage
}).sort_values('Pourcentage', ascending=False)

print(f"\nValeurs manquantes:")
print(missing_info[missing_info['Valeurs_Manquantes'] > 0])


# Il n'y a pas de valeurs manquantes..........

# ### **Vérification des doublons**

# In[69]:


duplicates = data.duplicated().sum()
print(f"\nDoublons détectés: {duplicates} ({duplicates/len(data)*100:.4f}%)")

if duplicates > 0:
    data = data.drop_duplicates()
    print(f"Doublons supprimés. Nouvelle shape: {data.shape}")


# 
# 
# ---
# 
# 

# # **Section 1 : Analyse exploratoire (EDA)**

# 
# 
# ---
# 
# 

# ### **--- Inspection de la variable Cible**

# In[70]:


# Distribution des classes
class_distribution = data['Classe'].value_counts()
class_percentage = data['Classe'].value_counts(normalize=True) * 100

print(f"\nDistribution des classes:")
for classe, count in class_distribution.items():
    print(f"  Classe {classe}: {count:6d} échantillons ({class_percentage[classe]:.4f}%)")


# In[71]:


# Visualisation de la distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
bars = plt.bar(class_distribution.index.astype(str), class_distribution.values,
               color=['skyblue', 'salmon'], alpha=0.8)
plt.title('Distribution des Classes')
plt.xlabel('Classe (0=Normale, 1=Fraude)')
plt.ylabel('Nombre d\'échantillons')

# Ajoutons des pourcentages sur les barres
for bar, percentage in zip(bars, class_percentage):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 100,
             f'{percentage:.4f}%', ha='center', va='bottom', fontweight='bold')

plt.subplot(1, 2, 2)
plt.pie(class_distribution.values, labels=class_distribution.index,
        autopct='%1.4f%%', colors=['lightblue', 'lightcoral'], startangle=90)
plt.title('Répartition des Classes (%)')

plt.tight_layout()
plt.show()


# L’analyse notre variable cible "Classe" montre une répartition relativement déséquilibrée entre les deux modalités : 283 253 transaction normales et 473 transactions fraudulauses. Cet écart est forte et traduit un déséquilibre significatif des classes. Donc, il est nécessaire d’ajuster les données avant la modélisation, car les deux catégories ne sont pas bien représentées pour garantir une analyse fiable et non biaisée.

# ### **Statistiques descriptives**

# In[72]:


# Statistiques générales
print("Statistiques générales:")
display(data.describe().T)


# ### **Montant des transactions**

# In[73]:


# visualisons la moyenne , la médiane le max et l'ecartype du montant de transactions
print("\nANALYSE Du MONTANT DES TRANSACTIONS:")
print(f"Montant moyen: {data['Montant_Transaction'].mean():.2f} €")
print(f"Montant médian: {data['Montant_Transaction'].median():.2f} €")
print(f"Montant max: {data['Montant_Transaction'].max():.2f} €")
print(f"Écart-type: {data['Montant_Transaction'].std():.2f} €")


# ### **Distribution des montants par classe**

# In[74]:


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(data['Montant_Transaction'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution du Montant des Transactions')
plt.xlabel('Montant en (€)')
plt.ylabel('Fréquence')
plt.yscale('log')  # Échelle log pour mieux visualiser

plt.subplot(1, 3, 2)
montant_normal = data[data['Classe'] == 0]['Montant_Transaction']
montant_fraude = data[data['Classe'] == 1]['Montant_Transaction']

plt.hist(montant_normal, bins=50, alpha=0.7, label='Normal', color='green', edgecolor='black')
plt.title('Montants - Transactions Normales')
plt.xlabel('Montant en (€)')
plt.ylabel('Fréquence')
plt.yscale('log')

plt.subplot(1, 3, 3)
plt.hist(montant_fraude, bins=50, alpha=0.7, label='Fraude', color='red', edgecolor='black')
plt.title('Montants des Transactions Frauduleuses')
plt.xlabel('Montant en euro')
plt.ylabel('Fréquence')
plt.yscale('log')

plt.tight_layout()
plt.show()


# ### **Analyse du temps**

# In[75]:


print("\nANALYSE DU TEMPS ÉCOULÉ")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(data['Temps_ecoule_sec'], bins=50, alpha=0.7, color='purple', edgecolor='black')
plt.title('Distribution du Temps Écoulé')
plt.xlabel('Secondes depuis première transaction')
plt.ylabel('Fréquence')

plt.subplot(1, 2, 2)
# Conversion en heures pour une meilleure interprétation
temps_heures = data['Temps_ecoule_sec'] / 3600
plt.hist(temps_heures, bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribution du Temps (en heures)')
plt.xlabel('Heures depuis première transaction')
plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()


# ### **Distribution des features V1-V28**

# In[76]:


# Sélection de quelques features pour la visualisation
features_to_plot = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']

plt.figure(figsize=(16, 12))
for i, feature in enumerate(features_to_plot, 1):
    plt.subplot(4, 2, i)
    plt.hist(data[feature], bins=50, alpha=0.7, color='teal', edgecolor='black')
    plt.title(f'Distribution de {feature}')
    plt.xlabel(feature)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()



# In[77]:


# Boxplots pour détecter les outliers
print("\nBOXPLOTS POUR DÉTECTION DES OUTLIERS")

plt.figure(figsize=(16, 8))
features_boxplot = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

for i, feature in enumerate(features_boxplot, 1):
    plt.subplot(2, 3, i)
    data.boxplot(column=feature)
    plt.title(f'Boxplot de {feature}')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ### **--- Faisons le Test de Shapiro-Wilk pour vérifier la normalité**

# In[ ]:





# # **Section 2 : Pré-traitement des données**

# ### **Vérification finale de la qualité de nos données**

# In[78]:


print(f"Shape finale du dataset aprè supp. des doublons: {data.shape}")
print(f"Valeurs manquantes totales: {data.isnull().sum().sum()}")
print(f"Doublons restants: {data.duplicated().sum()}")

# Vérification des constantes
constant_columns = [col for col in data.columns if data[col].nunique() == 1]
print(f"Colonnes constantes: {constant_columns}")


# ### **Séparation des features et de la target**

# In[79]:


print("\nSÉPARATION FEATURES/TARGET")

X = data.drop('Classe', axis=1)
y = data['Classe']

print(f"Features (X): {X.shape}")
print(f"Target (y): {y.shape}")
print(f"Proportion de fraudes: {y.mean()*100:.4f}%")


# ### **Techniques de ré-échantillonnage**

# ### **Split train/test avec stratification**

# In[80]:


print("\nSPLIT TRAIN/TEST AVEC STRATIFICATION")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"X_train: {X_train.shape} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"X_test: {X_test.shape} ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")

print(f"\nDistribution dans y_train:")
print(f"  Normal: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.4f}%)")
print(f"  Fraude: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.4f}%)")

print(f"\nDistribution dans y_test:")
print(f"  Normal: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.4f}%)")
print(f"  Fraude: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.4f}%)")


# ### **Scaling des features**

# In[81]:


print("\nSCALING DES FEATURES")

# Utilisation de RobustScaler
scaler = RobustScaler()

# Application du scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrame pour meilleure lisibilité
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("Notre Scaling terminé avec RobustScaler")

# Vérification des statistiques après scaling
print(f"\nStatistiques après scaling (X_train):")
print(f"Moyennes: {X_train_scaled.mean().mean():.6f}")
print(f"Écarts-types: {X_train_scaled.std().mean():.6f}")
print(f" Min: {X_train_scaled.min().min():.6f}")
print(f" Max: {X_train_scaled.max().max():.6f}")


# In[ ]:





# ### **Vérification de la conservation des distributions**

# In[82]:


print("\nVÉRIFICATION DE LA CONSERVATION DES DISTRIBUTIONS")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Avant scaling
axes[0, 0].hist(X_train['V1'].values, bins=50, alpha=0.7, color='blue', label='Avant scaling')
axes[0, 0].set_title('Distribution V1 - Avant Scaling')
axes[0, 0].set_xlabel('V1')
axes[0, 0].set_ylabel('Fréquence')

axes[0, 1].hist(X_train['V2'].values, bins=50, alpha=0.7, color='blue', label='Avant scaling')
axes[0, 1].set_title('Distribution V2 - Avant Scaling')
axes[0, 1].set_xlabel('V2')
axes[0, 1].set_ylabel('Fréquence')

# Après scaling
axes[1, 0].hist(X_train_scaled['V1'].values, bins=50, alpha=0.7, color='red', label='Après scaling')
axes[1, 0].set_title('Distribution V1 - Après Scaling')
axes[1, 0].set_xlabel('V1 (scaled)')
axes[1, 0].set_ylabel('Fréquence')

axes[1, 1].hist(X_train_scaled['V2'].values, bins=50, alpha=0.7, color='red', label='Après scaling')
axes[1, 1].set_title('Distribution V2 - Après Scaling')
axes[1, 1].set_xlabel('V2 (scaled)')
axes[1, 1].set_ylabel('Fréquence')

plt.tight_layout()
plt.show()


# ### **Méthode d'Oversampling avec SMOTE**

# In[ ]:





# # **Section 3 : Analyse bivariee**

# ### **Fonction utilitaire pour l'analyse bivariée**

# In[83]:


def plot_bivariate_analysis(feature, data, target_col='Classe'):
    """
    Analyse bivariée d'une feature avec la target classs
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Distribution par classe
    data_normal = data[data[target_col] == 0][feature]
    data_fraud = data[data[target_col] == 1][feature]

    # Histogramme comparatif
    axes[0, 0].hist(data_normal, bins=50, alpha=0.7, label='Normal', color='green')
    axes[0, 0].hist(data_fraud, bins=50, alpha=0.7, label='Fraude', color='red')
    axes[0, 0].set_title(f'Distribution de {feature} par Classe')
    axes[0, 0].set_xlabel(feature)
    axes[0, 0].set_ylabel('Fréquence')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # Boxplot par classe
    data_to_plot = [data_normal, data_fraud]
    axes[0, 1].boxplot(data_to_plot, labels=['Normal', 'Fraude'])
    axes[0, 1].set_title(f'Boxplot de {feature} par Classe')
    axes[0, 1].set_ylabel(feature)

    # Densité par classe
    data_normal.plot(kind='density', ax=axes[1, 0], label='Normal', color='green')
    data_fraud.plot(kind='density', ax=axes[1, 0], label='Fraude', color='red')
    axes[1, 0].set_title(f'Densité de {feature} par Classe')
    axes[1, 0].set_xlabel(feature)
    axes[1, 0].legend()

    # Violin plot
    violin_data = [data_normal, data_fraud]
    axes[1, 1].violinplot(violin_data, showmeans=True)
    axes[1, 1].set_xticks([1, 2])
    axes[1, 1].set_xticklabels(['Normal', 'Fraude'])
    axes[1, 1].set_title(f'Violin Plot de {feature} par Classe')
    axes[1, 1].set_ylabel(feature)

    plt.tight_layout()
    plt.show()

    # Statistiques descriptives par classe
    print(f"\nSTATISTIQUES DE {feature} PAR CLASSE:")
    stats_df = data.groupby(target_col)[feature].describe()
    display(stats_df)


# In[84]:


# Sélection des features les plus discriminantes (basé sur la variance)
features_analysis = ['V1', 'V2', 'V3', 'V4', 'V7', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']

for feature in features_analysis[:4]:  # on limite à 4 pour la demo
    print(f"\n{'='*30}")
    print(f"ANALYSE DE LA FEATURE: {feature}")
    print(f"{'='*30}")
    plot_bivariate_analysis(feature, data)


# ### **Analyse du montant par classe**

# In[85]:


print("\nANALYSE BIVARIÉE DU MONTANT PAR CLASSE")

plt.figure(figsize=(15, 10))

# Distribution du montant
plt.subplot(2, 2, 1)
montant_normal = data[data['Classe'] == 0]['Montant_Transaction']
montant_fraude = data[data['Classe'] == 1]['Montant_Transaction']

plt.hist(montant_normal, bins=50, alpha=0.7, label='Normal', color='green', density=True)
plt.hist(montant_fraude, bins=50, alpha=0.7, label='Fraude', color='red', density=True)
plt.title('Distribution du Montant par Classe (Densité)')
plt.xlabel('Montant en (€)')
plt.ylabel('Densité')
plt.legend()
plt.yscale('log')

# Boxplot du montant
plt.subplot(2, 2, 2)
data_to_plot = [montant_normal, montant_fraude]
plt.boxplot(data_to_plot, labels=['Normal', 'Fraude'])
plt.title('Boxplot du Montant par Classe')
plt.ylabel('Montant (€)')

# Statistiques cumulatives
plt.subplot(2, 2, 3)
plt.hist(montant_normal, bins=50, alpha=0.7, label='Normal', color='green',
         cumulative=True, density=True, histtype='step')
plt.hist(montant_fraude, bins=50, alpha=0.7, label='Fraude', color='red',
         cumulative=True, density=True, histtype='step')
plt.title('Fonction de Répartition du Montant')
plt.xlabel('Montant (€)')
plt.ylabel('Probabilité Cumulée')
plt.legend()

# Focus sur les petites transactions
plt.subplot(2, 2, 4)
montant_normal_small = montant_normal[montant_normal <= 1000]
montant_fraude_small = montant_fraude[montant_fraude <= 1000]

plt.hist(montant_normal_small, bins=50, alpha=0.7, label='Normal', color='green', density=True)
plt.hist(montant_fraude_small, bins=50, alpha=0.7, label='Fraude', color='red', density=True)
plt.title('Distribution du Montant (<1000€)')
plt.xlabel('Montant (€)')
plt.ylabel('Densité')
plt.legend()

plt.tight_layout()
plt.show()

# Statistiques détaillées du montant
print("\nSTATISTIQUES DÉTAILLÉES DU MONTANT PAR CLASSE:")
montant_stats = data.groupby('Classe')['Montant_Transaction'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max'
]).round(2)
display(montant_stats)


# ### **Matrice de corrélation**

# In[86]:


print("\nMATRICE DE CORRÉLATION")

# Calcul de la matrice de corrélation
correlation_matrix = data.corr()

# Focus sur les corrélations avec la target
target_correlations = correlation_matrix['Classe'].sort_values(ascending=False)

print("CORRÉLATIONS AVEC LA TARGET (Classe):")
print(target_correlations)

# Visualisation des corrélations les plus fortes
plt.figure(figsize=(12, 8))
top_correlations = target_correlations[1:11]  # Exclure la corrélation avec elle-même
colors = ['red' if x > 0 else 'blue' for x in top_correlations.values]

plt.barh(range(len(top_correlations)), top_correlations.values, color=colors, alpha=0.7)
plt.yticks(range(len(top_correlations)), top_correlations.index)
plt.title('Top 10 des Corrélations avec la Classe')
plt.xlabel('Coefficient de Corrélation')
plt.grid(axis='x', alpha=0.3)

# Ajout des valeurs sur les barres
for i, v in enumerate(top_correlations.values):
    plt.text(v, i, f' {v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.show()


# In[ ]:





# ### **Heatmap de corrélation des features les plus corrélées**

# In[87]:


print("\nHEATMAP DES CORRÉLATIONS")

# Sélection des features les plus corrélées avec la target
top_features = target_correlations.abs().sort_values(ascending=False).head(15).index
correlation_top = data[top_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_top, dtype=bool))  # Masque pour le triangle supérieur

sns.heatmap(correlation_top, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': .8})
plt.title('Heatmap des Corrélations - Top 15 Features')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# # **Section 4 : Gestion des désiquilibres**

# In[88]:


# Appliquer SMOTE uniquement sur le training set scaled
smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("SMOTE appliqué avec succès")

# Affichage des distributions avant/après
print(f"\n--- DISTRIBUTIONS AVANT/APRÈS SMOTE ---")
print(f"AVANT SMOTE (Training Set):")
train_before_0 = (y_train == 0).sum()
train_before_1 = (y_train == 1).sum()
print(f"  Classe 0 (Normale): {train_before_0:6d} échantillons ({(y_train == 0).mean()*100:.4f}%)")
print(f"  Classe 1 (Fraude):  {train_before_1:6d} échantillons ({(y_train == 1).mean()*100:.4f}%)")
print(f"  Ratio: {train_before_0/train_before_1:.1f}:1")

print(f"\nAPRÈS SMOTE (Training Set):")
train_after_0 = (y_train_balanced == 0).sum()
train_after_1 = (y_train_balanced == 1).sum()
print(f"  Classe 0 (Normale): {train_after_0:6d} échantillons ({(y_train_balanced == 0).mean()*100:.4f}%)")
print(f"  Classe 1 (Fraude):  {train_after_1:6d} échantillons ({(y_train_balanced == 1).mean()*100:.4f}%)")
print(f"  Ratio: {train_after_0/train_after_1:.2f}:1")

print(f"\nTEST SET (inchangé):")
test_0 = (y_test == 0).sum()
test_1 = (y_test == 1).sum()
print(f"  Classe 0 (Normale): {test_0:6d} échantillons ({(y_test == 0).mean()*100:.4f}%)")
print(f"  Classe 1 (Fraude):  {test_1:6d} échantillons ({(y_test == 1).mean()*100:.4f}%)")
print(f"  Ratio: {test_0/test_1:.1f}:1")

# Visualisation comparative
print(f"\n--- VISUALISATION DES DISTRIBUTIONS ---")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution originale complète
original_counts = pd.Series(y).value_counts()
axes[0, 0].bar(['Normale', 'Fraude'], original_counts.values,
                color=['lightblue', 'salmon'], alpha=0.8)
axes[0, 0].set_title('DISTRIBUTION ORIGINALE\n(Complet Dataset)', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Nombre déchantillons', fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(axes[0, 0].containers[0]):
    axes[0, 0].text(i, v.get_height() + 1000, f'{v.get_height():,}',
                    ha='center', fontweight='bold', fontsize=10)

# Training set avant SMOTE
train_before_counts = pd.Series(y_train).value_counts()
axes[0, 1].bar(['Normale', 'Fraude'], train_before_counts.values,
               color=['lightblue', 'salmon'], alpha=0.8)
axes[0, 1].set_title('TRAINING SET - AVANT SMOTE\n(Même distribution que la réalité)',
                     fontweight='bold', fontsize=12)
axes[0, 1].set_ylabel('Nombre déchantillons', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(axes[0, 1].containers[0]):
    axes[0, 1].text(i, v.get_height() + 500, f'{v.get_height():,}',
                    ha='center', fontweight='bold', fontsize=10)

# Training set après SMOTE
train_after_counts = pd.Series(y_train_balanced).value_counts()
axes[1, 0].bar(['Normale', 'Fraude'], train_after_counts.values,
               color=['lightgreen', 'lightcoral'], alpha=0.8)
axes[1, 0].set_title('TRAINING SET - APRÈS SMOTE\n(Distribution équilibrée pour lentraînement)',
                     fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('Nombre déchantillons', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(axes[1, 0].containers[0]):
    axes[1, 0].text(i, v.get_height() + 500, f'{v.get_height():,}',
                    ha='center', fontweight='bold', fontsize=10)

# Test set (inchangé)
test_counts = pd.Series(y_test).value_counts()
axes[1, 1].bar(['Normale', 'Fraude'], test_counts.values,
               color=['lightblue', 'salmon'], alpha=0.8)
axes[1, 1].set_title('TEST SET - DISTRIBUTION RÉELLE\n(Pour évaluation réaliste)',
                     fontweight='bold', fontsize=12)
axes[1, 1].set_ylabel('Nombre déchantillons', fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(axes[1, 1].containers[0]):
    axes[1, 1].text(i, v.get_height() + 200, f'{v.get_height():,}',
                    ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()


# In[ ]:





# In[89]:


# Résumé final
print("1. SPLIT TRAIN/TEST avec stratification")
print("2. SCALING avec RobustScaler (train fit_transform, test transform)")
print("3. RÉÉQUILIBRAGE avec SMOTE (uniquement sur training set)")
print(f"4. Données finales prêtes pour la modélisation:")
print(f" - X_train_balanced: {X_train_balanced.shape}")
print(f" - y_train_balanced: {y_train_balanced.shape}")
print(f" - X_test_scaled:    {X_test_scaled.shape}")
print(f" -y_test:           {y_test.shape}")

# Vérification de la qualité des données après SMOTE
print(f"\n--- QUALITÉ DES DONNÉES APRÈS SMOTE --")
print(f"Vérification des valeurs manquantes: {np.isnan(X_train_balanced).sum().sum()}")
print(f"Vérification des infinis: {np.isinf(X_train_balanced).sum().sum()}")
print(f"Types de données: {X_train_balanced.dtypes.unique()}")


# # **Section 4 : Modèles & Évaluation**

# ### **Définir les fonctions d'évaluation**

# In[90]:


print("FONCTIONS D'ÉVALUATION")


def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Évalue un modèle et retourne les métriques principales
    """
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)

    # Rapport de classification
    class_report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraude'])

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_confusion_matrix(cm, model_name=""):
    """Plot une matrice de confusion stylisée"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraude'],
                yticklabels=['Normal', 'Fraude'])
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Étiquettes prédites')
    plt.show()

def plot_roc_curves(results_dict):
    """Plot les courbes ROC pour tous les modèles"""
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        y_test = results.get('y_test', y_test)  # Fallback to global y_test
        y_pred_proba = results['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = results['roc_auc']

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall_curves(results_dict):
    """Plot les courbes Precision-Recall pour tous les modèles"""
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        y_test = results.get('y_test', y_test)  # Fallback to global y_test
        y_pred_proba = results['y_pred_proba']

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = results['avg_precision']

        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbes Precision-Recall - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.show()

def display_metrics_comparison(results_dict):
    """Affiche un tableau comparatif des métriques"""
    metrics_df = pd.DataFrame({
        model_name: {
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score'],
            'ROC AUC': results['roc_auc'],
            'Avg Precision': results['avg_precision']
        }
        for model_name, results in results_dict.items()
    }).T

    # Style le DataFrame
    styled_df = metrics_df.style\
        .background_gradient(cmap='Blues', axis=0)\
        .format('{:.4f}')\
        .set_caption('COMPARAISON DES MÉTRIQUES PAR MODÈLE')

    return styled_df


# In[ ]:





# In[ ]:





# ### **Initialisation des modèles**

# In[91]:


import time

# Dictionnaire pour stocker les résultats
results = {}

# MODÈLES OPTIMISÉS pour vitesse
fast_models = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=500,
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1]),
        n_estimators=50,  # Réduit
        max_depth=6,      # Limité
        n_jobs=-1
    ),
    'LightGBM': LGBMClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_estimators=50,  # Réduit
        n_jobs=-1,
        verbose=-1
    )
}

print("Modèles définis :")
for model_name, model in fast_models.items():
    print(f"- {model_name}")



# ### **Entraînement et évaluation des modèles**

# In[92]:


print("\nDébut de l'entraînement des modèles optimisés...")
print("-" * 60)

for model_name, model in fast_models.items():
    print(f"\nEntraînement: {model_name}")
    print("." * 40)

    # Entraînement avec timer
    start_time = time.time()
    model.fit(X_train_balanced, y_train_balanced)
    training_time = time.time() - start_time

    # Évaluation
    model_results = evaluate_model(model, X_test_scaled, y_test, model_name)
    model_results['training_time'] = training_time

    # Stockage des résultats
    results[model_name] = model_results

    # Affichage compact des résultats
    print(f"Terminé en {training_time:.2f}s")
    print(f"Accuracy: {model_results['accuracy']:.4f}")
    print(f"Precision: {model_results['precision']:.4f}")
    print(f"Recall: {model_results['recall']:.4f}")
    print(f"F1-Score: {model_results['f1_score']:.4f}")
    print(f"ROC AUC: {model_results['roc_auc']:.4f}")

    # Matrice de confusion (optionnel - commenter si trop long)
    plot_confusion_matrix(model_results['confusion_matrix'], model_name)


# In[93]:


print(f"\nRapport de Classification - {model_name}:")
print(model_results['classification_report'])


# In[ ]:





# In[ ]:





# In[ ]:





# ### **Comparaison des modèles**

# In[102]:


# Tableau comparatif
print("TABLEAU COMPARATIF DES MÉTRIQUES:")
comparison_df = display_metrics_comparison(results)
display(comparison_df)

#  Redéfinir les fonctions avec le bon scope
def plot_roc_curves(results_dict, y_test_global):
    """Plot les courbes ROC pour tous les modèles"""
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        # Utiliser y_test_global passé en paramètre
        y_pred_proba = results['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_test_global, y_pred_proba)
        roc_auc = results['roc_auc']

        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs')
    plt.ylabel('Taux de Vrais Positifs')
    plt.title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

def plot_precision_recall_curves(results_dict, y_test_global):
    """Plot les courbes Precision-Recall pour tous les modèles"""
    plt.figure(figsize=(10, 8))

    for model_name, results in results_dict.items():
        # Utiliser y_test_global passé en paramètre
        y_pred_proba = results['y_pred_proba']

        precision, recall, _ = precision_recall_curve(y_test_global, y_pred_proba)
        avg_precision = results['avg_precision']

        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})', linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbes Precision-Recall - Comparaison des Modèles', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.show()

# Visualisation des métriques
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'avg_precision']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Avg Precision']

for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
    ax = axes[idx//3, idx%3]
    metric_values = [results[model][metric] for model in results.keys()]

    bars = ax.bar(results.keys(), metric_values, color=plt.cm.Set3(np.arange(len(results))))
    ax.set_title(f'{name} par Modèle', fontweight='bold')
    ax.set_ylabel(name)
    ax.tick_params(axis='x', rotation=45)

    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# Courbes ROC
print("\nCOURBES ROC - COMPARAISON:")
plot_roc_curves(results, y_test)

# Courbes Precision-Recall -
print("\nCOURBES PRECISION-RECALL - COMPARAISON:")
plot_precision_recall_curves(results, y_test)


# In[ ]:





# In[ ]:


# Création du DataFrame de comparaison
comparison_data = []
for model_name, metrics in results.items():
    comparison_data.append({
        'Modèle': model_name,
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1'],
        'ROC-AUC': metrics['roc_auc'],
        'Avg_Precision': metrics['avg_precision']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Avg_Precision', ascending=False)

print("CLASSEMENT DES MODÈLES (par Average Precision):")
display(comparison_df.round(4))


# ### **Analyse détaillée du meilleur modèle**

# In[99]:


# Identification du meilleur modèle basé sur le F1-Score
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
best_model_results = results[best_model_name]

print(f"MEILLEUR MODÈLE IDENTIFIÉ: {best_model_name}")
print(f"F1-Score: {best_model_results['f1_score']:.4f}")
print(f"ROC AUC: {best_model_results['roc_auc']:.4f}")
print(f"Temps d'entraînement: {best_model_results['training_time']:.2f} secondes")

# Récupérer le modèle entraîné
best_model = None
for model_name, model in models.items():
    if model_name == best_model_name:
        best_model = model
        break

# Analyse détaillée
print(f"\nANALYSE DÉTAILLÉE - {best_model_name}")
print(f"{'-'*60}")

#  Matrice de confusion détaillée
print("MATRICE DE CONFUSION DÉTAILLÉE:")
plot_confusion_matrix(best_model_results['confusion_matrix'], f"MEILLEUR: {best_model_name}")

#  Métriques détaillées
print("MÉTRIQUES DÉTAILLÉES:")
detailed_metrics = {
    'Accuracy': best_model_results['accuracy'],
    'Precision': best_model_results['precision'],
    'Recall': best_model_results['recall'],
    'F1-Score': best_model_results['f1_score'],
    'ROC AUC': best_model_results['roc_auc'],
    'Average Precision': best_model_results['avg_precision']
}

for metric, value in detailed_metrics.items():
    print(f"   {metric}: {value:.4f}")

# Feature Importance (si disponible)
if hasattr(best_model, 'feature_importances_'):
    print("\nL'IMPORTANCE DES FEATURES:")
    feature_importance = pd.DataFrame({
        'feature': X_train_balanced.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Top 15 features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
    plt.title(f'Top 15 Features les Plus Importantes - {best_model_name}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

    print("Top 10 features les plus importantes:")
    display(feature_importance.head(10))


# In[ ]:





# In[100]:


#  Courbes détaillées
print("\nCOURBES DE PERFORMANCE DÉTAILLÉES:")

# Courbe ROC détaillée
fpr, tpr, _ = roc_curve(y_test, best_model_results['y_pred_proba'])
roc_auc = best_model_results['roc_auc']

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs')
plt.ylabel('Taux de Vrais Positifs')
plt.title(f'Courbe ROC Détaillée - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

# Courbe Precision-Recall détaillée
precision, recall, _ = precision_recall_curve(y_test, best_model_results['y_pred_proba'])
avg_precision = best_model_results['avg_precision']

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Courbe Precision-Recall Détaillée - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc='upper right')
plt.grid(alpha=0.3)
plt.show()


# In[ ]:





# In[101]:


# Analyse des prédictions
print("\nANALYSE DES PRÉDICTIONS:")
y_pred = best_model_results['y_pred']
y_pred_proba = best_model_results['y_pred_proba']

# Distribution des probabilités de prédiction
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Normal', color='green')
plt.title('Distribution des Probabilités - Transactions Normales')
plt.xlabel('Probabilité de Fraude')
plt.ylabel('Fréquence')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Fraude', color='red')
plt.title('Distribution des Probabilités - Fraudes')
plt.xlabel('Probabilité de Fraude')
plt.ylabel('Fréquence')
plt.legend()

plt.tight_layout()
plt.show()

