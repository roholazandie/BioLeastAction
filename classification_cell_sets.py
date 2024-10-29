import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
adata = sc.read_h5ad("data/reprogramming_schiebinger_serum_computed.h5ad")

# Extract the PCA features and the cell types
X = adata.obsm['X_pca']  # The PCA features
y = adata.obs['cell_sets'].values  # The cell type labels

# Split the data into training and testing sets with shuffling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Choose the classifier
classifier_type = 'mlp'  # Options: 'random_forest', 'svm', 'mlp'

if classifier_type == 'random_forest':
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
elif classifier_type == 'svm':
    clf = SVC(kernel='linear', random_state=42)
elif classifier_type == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
else:
    raise ValueError("Invalid classifier type. Choose from 'random_forest', 'svm', or 'mlp'.")

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print a detailed classification report
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Get unique cell types as labels for the confusion matrix
labels = np.unique(y)

# Plot the confusion matrix
plot_confusion_matrix(y_test, y_pred, labels)
