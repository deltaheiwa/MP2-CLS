from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from reglog import LogisticRegression

x_data, y_data = make_moons(n_samples=1000, noise=0.25)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

logreg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
logreg.fit(x_train, y_train)


