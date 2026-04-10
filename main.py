import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.estimator_checks import check_estimator

from reglog import LogisticRegression

models = {}
x_data, y_data = make_moons(n_samples=1000, noise=0.25)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# print(check_estimator(LogisticRegression()))

def plot_decision_boundaries(models_dict, x, y):
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    models_to_plot = [
        'logreg',
        'dt-gini-10',
        'rf-100',
        'svm',
        'vc'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 4))

    for ax, name in zip(axes, models_to_plot):
        model = models_dict[name]

        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(x[:, 0], x[:, 1], c=y, s=10, edgecolor='k', cmap=plt.cm.RdYlBu, alpha=0.6)

        acc = accuracy_score(y, model.predict(x))

        short_name = name.split('(')[0].strip()
        ax.set_title(f"{short_name}\nAcc: {acc:.3f}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

logreg = LogisticRegression(learning_rate=0.5, n_iterations=1000)
print("Training logistic regression...")
logreg.fit(x_train, y_train)

models['logreg'] = logreg

for criteria in ['gini', 'entropy']:
    for depth in [3, 10, None]:
        name = f"dt-{criteria}-{depth if depth else 'unlimited'}"
        dt = DecisionTreeClassifier(criterion=criteria, max_depth=depth, random_state=42)
        print(f"Training {name}...")
        dt.fit(x_train, y_train)
        models[name] = dt

for n_trees in [10, 100, 200]:
    name = f"rf-{n_trees}"
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=5, random_state=42)
    print(f"Training {name}...")
    rf.fit(x_train, y_train)
    models[name] = rf

svm = SVC(probability=True, random_state=42)
print(f"Training SVM...")
svm.fit(x_train, y_train)
models['svm'] = svm

ensemble = VotingClassifier([
    ('lr', models['logreg']),
    ('rf', models['rf-100']),
    ('svm', models['svm'])
], voting='soft')
print(f"Training voting classifier...")
ensemble.fit(x_train, y_train)
models['vc'] = ensemble
print("Completed")

print("\n--- Accuracy Results on Test Set ---")
for name, model in models.items():
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} {accuracy:.4f}")

plot_decision_boundaries(models, x_data, y_data)

