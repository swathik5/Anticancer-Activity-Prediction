#Importing libraries & Setting random seed
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import warnings
warnings.filterwarnings('ignore')
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
from tensorflow.keras import layers
from keras.models import Model
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from scikeras.wrappers import KerasClassifier

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
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, X_train2, Y_train2, _cv=5):

      _scoring = ['accuracy', 'precision', 'recall', 'f1']
      results = cross_validate(estimator=model,
                               X=X_train2,
                               y=Y_train2,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
      return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Training Recall scores": results['train_recall'],
              "Mean Training Recall": results['train_recall'].mean(),
              "Training F1 scores": results['train_f1'],
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              "Validation Recall scores": results['test_recall'],
              "Mean Validation Recall": results['test_recall'].mean(),
              "Validation F1 scores": results['test_f1'],
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
# Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='indigo', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='lightblue', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(False)
        plt.show()

def mlp_model():
 	##### MODEL ARCHITECTURE
	mlp_input = layers.Input(shape=(421,), name='mlp_input')
	x = layers.Dense(421, activation='relu', kernel_initializer='he_normal')(mlp_input)
	dense_hidden = Dense(25, activation='relu', name='dense_hidden')(x)
	last_layer = Dense(1, activation='sigmoid')(dense_hidden)
 	model = Model(inputs=structured_input, outputs=last_layer)
	print(model.summary())
	# learning rate decay schedule
	initial_learning_rate = 0.1, lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
	stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode='min', verbose=1)
	best = ModelCheckpoint(filepath='/home/dell11/mlp_t/kfold/best_structured_model_MLP_model_kfold.hdf5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
	# compile the model
	model.compile(optimizer=Adam(learning_rate=lr_schedule, epsilon=1), loss="binary_crossentropy", metrics=['accuracy'])
	results = model.fit(X_train2, Y_train2, epochs=200, callbacks=[stop, best], validation_data=([X_test2, Y_test2]))
	model.save('/home/dell11/mlp_t/kfold/best_structured_model_MLP_model_kfold.h5')
	return model
Kmodel = KerasClassifier(build_fn=lambda:mlp_model(), verbose=1)

mlp_model_result = cross_validation(Kmodel, X_train2, Y_train2, 5)
print(mlp_model_result)

# Plotting Performance metrics
model_name = "MLP"
#Accuracy
plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            mlp_model_result["Training Accuracy scores"],
            mlp_model_result["Validation Accuracy scores"])
# Precision
plot_result(model_name,
            "Precision",
            "Precision scores in 5 Folds",
            mlp_model_result["Training Precision scores"],
            mlp_model_result["Validation Precision scores"])
# Recall
plot_result(model_name,
            "Recall",
            "Recall scores in 5 Folds",
            mlp_model_result["Training Recall scores"],
            mlp_model_result["Validation Recall scores"])
# F1-Score 
plot_result(model_name,
            "F1",
            "F1 Scores in 5 Folds",
            mlp_model_result["Training F1 scores"],
            mlp_model_result["Validation F1 scores"])   
