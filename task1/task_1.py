import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from rdchiral.template_extractor import extract_from_reaction
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Schneider50k dataset
train_data = pd.read_csv('C:\\Users\\123\OneDrive\\桌面\\机器学习大作业\\Project for ML\\schneider50k\\raw_train.csv')
val_data = pd.read_csv('C:\\Users\\123\OneDrive\\桌面\\机器学习大作业\\Project for ML\\schneider50k\\raw_val.csv')
test_data = pd.read_csv('C:\\Users\\123\OneDrive\\桌面\\机器学习大作业\\Project for ML\\schneider50k\\raw_test.csv')

# Combine the training and validation data for preprocessing
data = pd.concat([train_data, val_data])

# Split reactions into multiple reactions with one product
products = []
templates = []
for reaction in data['reactants>reagents>production']:
    reactants, products_str = reaction.split('>>')
    product_list = products_str.split('.')

    for product in product_list:
        products.append(product.strip())
        # Extract reaction template using rdchiral library
        reactant_template = extract_from_reaction({'_id': None, 'reactants': reactants, 'products': product})

        if 'reaction_smarts' in reactant_template.keys():
            templates.append(reactant_template['reaction_smarts'])
        else:
            templates.append(None)

# Create a DataFrame for products and templates
data_df = pd.DataFrame({'product': products, 'template': templates})

# Remove reactions with missing templates
data_df = data_df.dropna()


# Convert product SMILES to Morgan fingerprints
def get_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    on_bits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=bool)
    arr[on_bits] = 1
    return arr


data_df['product_fp'] = data_df['product'].apply(get_morgan_fingerprint)

# Encode templates with label encoder
label_encoder = LabelEncoder()
data_df['template_enc'] = label_encoder.fit_transform(data_df['template'])

# Split the data into training, validation, and test sets
X = np.stack(data_df['product_fp'].values)
y = np.array(data_df['template_enc'].values)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

test_products = []
test_templates = []
for reaction in test_data['reactants>reagents>production']:
    reactants, products_str = reaction.split('>>')
    product_list = products_str.split('.')
    for product in product_list:
        test_products.append(product.strip())
        # Extract reaction template using rdchiral library
        reactant_template = extract_from_reaction({'_id': None, 'reactants': reactants, 'products': product})

        if 'reaction_smarts' in reactant_template.keys():
            test_templates.append(reactant_template['reaction_smarts'])
        else:
            test_templates.append(None)

# Create a DataFrame for products and templates
test_data_df = pd.DataFrame({'product': test_products, 'template': test_templates})

# Remove reactions with missing templates
test_data_df = test_data_df.dropna()

test_data_df['product_fp'] = test_data_df['product'].apply(get_morgan_fingerprint)
test_data_df['template_enc'] = label_encoder.fit_transform(test_data_df['template'])

X_test = np.stack(test_data_df['product_fp'].values)
y_test = np.array(test_data_df['template_enc'].values)

# Build the model
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(2048,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(data_df['template_enc']), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

X_train_gpu = tf.convert_to_tensor(X_train)
y_train_gpu = tf.convert_to_tensor(y_train)
X_val_gpu = tf.convert_to_tensor(X_val)
y_val_gpu = tf.convert_to_tensor(y_val)

# Train the model
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_val, y_val))

X_test_gpu = tf.convert_to_tensor(X_test)
y_test_gpu = tf.convert_to_tensor(y_test)

# Evaluate the model on the test set
_, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)

# Save loss, accuracy, validation loss, and validation accuracy to an Excel file
history_df = pd.DataFrame(history.history)
history_df.to_excel('C:\\Users\\123\\OneDrive\\桌面\\机器学习大作业\\model_history.xlsx', index=False)
