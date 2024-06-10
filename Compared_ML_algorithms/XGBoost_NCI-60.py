
#Importing libraries & Setting random seed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from warnings import filterwarnings
import tensorflow as tf
import numpy as np
import pandas as pd
import random as rn
from pathlib import Path
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, matthews_corrcoef
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from plot_metric.functions import BinaryClassification
sd = 123 
np.random.seed(sd)
rn.seed(sd)
os.environ['PYTHONHASHSEED']=str(sd)
from keras import backend as K
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
tf.random.set_seed(sd)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
K.set_session(sess)
# Silence some expected warnings
filterwarnings("ignore")


#To reading NCI-60 dataset into a pandas dataframe

df = pd.read_csv("/home/dell11/all_bioactivity/all_activities/Anticancer/NCI_60_activity.csv", sep=',')

# To print the information about NCI-60 dataframe
df.info()

#Visualizing the NCI-60 data stored in dataframe
df.head()

# Conversion of double characters to single character 
def compute_double_characters(df):
    distinct_characters = set(df.smiles.apply(list).sum())
    
    upper_case_letters = []
    lower_case_letters = []
    for character in distinct_characters:
        if character.isalpha():
            if character.isupper():
                upper_case_letters.append(character)
            elif character.islower():
                 lower_case_letters.append(character)
    print(f"Upper letter characters {sorted(upper_case_letters)}")
    print(f"Lower letter characters {sorted( lower_case_letters)}")

    elements = ["Ac", "Al", "Am", "Sb", "Ar", "As", "At", "Ba", "Bk", "Be", "Bi", "Bh", "B", "Br", "Cd", "Ca",
        "Cf", "C", "Ce", "Cs", "Cl", "Cr", "Co", "Cn", "Cu", "Cm", "Ds", "Db", "Dy", "Es", "Er", "Eu","Fm", 
        "Fl", "F", "Fr", "Gd", "Ga", "Ge", "Au", "Hf", "Hs", "He", "Ho", "H", "In", "I", "Ir", "Fe",
        "Kr", "La", "Lr", "Pb", "Li", "Lv", "Lu", "Mg", "Mn", "Mt", "Md", "Hg", "Mo", "Mc", "Nd", "Ne",
        "Np", "Ni", "Nh", "Nb", "N", "No", "Og", "Os", "O", "Pd", "P", "Pt", "Pu", "Po", "K", "Pr", "Pm",
        "Pa","Ra", "Rn", "Re", "Rh", "Rg", "Rb", "Ru", "Rf", "Sm", "Sc", "Sg", "Se", "Si", "Ag", "Na",
        "Sr", "S", "Ta", "Tc", "Te", "Ts", "Tb", "Tl", "Th", "Tm", "Sn", "Ti", "W", "U", "V", "Xe", "Yb",
        "Y", "Zn", "Zr"]


    double_characters = []
    for uc in upper_case_letters:
        for lc in  lower_case_letters:
            two_char = uc + lc
            if two_char in elements:
                double_characters.append(two_char)

    double_characters_NCI = set()
    for dc in double_characters:
        if df.smiles.str.contains(dc).any():
            double_characters_NCI.add(dc)

    return double_characters_NCI

NCI_characters = compute_double_characters(df)
print(f"\n Double Characters in the data set: {sorted(NCI_characters)}")



substitute = {"Ac" : "J", "Ag" : "Q", "As" : "X", "Au" : "j", "Ba" : "k", "Bi" : "p", "Br" : "q", "Ca" : "v", "Cd" : "w", "Ce" : "x", "Cl" : "z", "Co" : "α", "Cr" : "β", "Cu" : "γ", "Dy" : "δ", "Er" : "ε", "Eu" : "ζ", "Fe" : "η", "Ga" : "θ","Gd" : "ι", "Ge" :"κ", "Hf" : "λ", "Hg" : "μ", "In" :"ν", "Ir" : "ξ", "La" : "é", "Li" : "π", "Mg" : "ρ", "Mn" : "σ", "Mo" : "τ", "Na" : "υ", "Nb" : "φ", "Nd" : "χ", "Ni" : "ψ", "Os" : "ω", "Pb" : "ς", "Pd" :	"Γ", "Pt" : "Θ", "Re" :"Ξ","Rh" : "Π", "Ru" : "Σ", "Sb" : "Φ", "Se" : "Ψ", "Si" : "Ω", "Sm" : "Á", "Sn" : "É", "Ta" : "Í", "Te" : "Ó", "Th" : "Ö", "Ti" : "Ő", "Tl" : "Ú", "Zn" : "Ü", "Zr" : "Ű"}

#Processing of SMILES

def smiles_processing(df, convert):
    # Create a new column having processed canonical SMILES
    df["smiles_modified"] = df["smiles"].copy()

    # Replace the two letter elements found with one character
    for record, conv in convert.items():
        df["smiles_modified"] = df["smiles_modified"].str.replace(
            record, conv
        )

    dist_char = set(df.smiles_modified.apply(list).sum())
    return df, dist_char

df, dist_char = smiles_processing(df, substitute)
df.head(3)

print(f"Distinct characters present in processed SMILES:\n{sorted(dist_char)}")


# Identifyig longest SMILES
smiles_largest = max(df["smiles"], key=len)
index_smiles_largest = df.smiles[df.smiles == smiles_largest].index.tolist()
print(f"Longest SMILES - {smiles_largest}")
print(f"of length {len(smiles_largest)} characters, index: {index_smiles_largest[0]}.")
largest_smiles_len = len(smiles_largest)

# One-hot encoding
def smiles_to_onehot(smiles, character_maximum, dist_char):
    smiles_index = {ch: index for index, ch in enumerate(dist_char)}
    smiles_onehot = np.zeros((len(dist_char), character_maximum))
    for index, ch in enumerate(smiles):
        smiles_onehot[smiles_index[ch], index] = 1
    return smiles_onehot

df["dist_char_ohe"] = df["smiles_modified"].apply(
    smiles_to_onehot, character_maximum=largest_smiles_len, dist_char=dist_char
)

#SMOTE oversampling, train-test split 
X = df[['dist_char_ohe','Activity']]
X.columns = ['feature_N' + str(i + 1) for i in range(X.shape[1])]
x = X['feature_N1'].explode().to_frame()
x['observation_id'] = x.groupby(level=0).cumcount()
x = x.pivot(columns='observation_id', values='feature_N1').fillna(0)
x = x.add_prefix('list_elementN_')
X.drop(columns='feature_N1', axis=1, inplace=True)
X = pd.concat([X, x], axis=1)
X = X[['list_elementN_0','feature_N2']]
X = pd.concat([X.pop('list_elementN_0').apply(pd.Series), X['feature_N2']], axis=1)
feature = X.drop('feature_N2', axis=1)
target = X['feature_N2']
X1= feature
y1=target
smote = SMOTE(random_state=0)
X_over, y_over = smote.fit_resample(X1, y1)
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X_over, y_over, random_state=42)

#Implementing XGBoost algorithm

xgb = XGBClassifier(subsample=1.0, min_child_weight=5, max_depth=3, gamma=5, colsample_bytree=1.0)
xgb.fit(X_train2, Y_train2)

#Prediction of test set & Performance evaluation

y_pred1 = xgb.predict(X_test2)
y_pred1 = y_pred1.flatten()
print(y_pred1.round(2))
y_pred1 = np.where(y_pred1 > 0.5, 1, 0)
print(y_pred1)

print(classification_report(Y_test2, y_pred1))
print(matthews_corrcoef(Y_test2, y_pred1))

print(classification_report(Y_test2, y_pred1))
print(matthews_corrcoef(Y_test2, y_pred1))

print("overall_score")

print("precision_score")
print(precision_score(Y_test2,y_pred1))

print("recall_score")
print(recall_score(Y_test2,y_pred1))

print("f1_score")
print(f1_score(Y_test2,y_pred1))

print("roc_auc_score")
print(roc_auc_score(Y_test2,y_pred1))



