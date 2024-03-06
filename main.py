
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


dataset = pd.read_csv('logistic regression dataset.csv')
dataset['Gender'] = dataset['Gender'].map({'Female': 1, 'Male': 0})
print(dataset.head())

X = np.asarray(dataset.drop(['User ID', 'Purchased'], axis=1))
y = dataset['Purchased']

X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.4, random_state=42)
X_cross_val, X_test, y_cross_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cross_val_scaled = scaler.transform(X_cross_val)
X_test_scaled = scaler.transform(X_test)

model1 = LogisticRegression()
model1.fit(X_train_scaled, y_train)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_cross_val_scaled, y_cross_val)
best_C = grid_search.best_params_['C']
print(f"Best C: {best_C}")

model2 = LogisticRegression(C=best_C)
model2.fit(X_train_scaled, y_train)

y_pred_model1 = model1.predict(X_test_scaled)
accuracy_model1 = accuracy_score(y_test, y_pred_model1)
confusion_matrix_model1 = confusion_matrix(y_test, y_pred_model1)
log_loss_model1 = log_loss(y_test, y_pred_model1)

y_pred_model2 = model2.predict(X_test_scaled)
accuracy_model2 = accuracy_score(y_test, y_pred_model2)
confusion_matrix_model2 = confusion_matrix(y_test, y_pred_model2)
log_loss_model2 = log_loss(y_test, y_pred_model2)

print(f"Accuracy\n Model 1: {accuracy_model1} \n Model 2: {accuracy_model2}\n\n"
      f"Confusion matrix\n Model 1:\n {confusion_matrix_model1} \n Model 2:\n {confusion_matrix_model2}\n\n"
      f"Log loss \n Model 1: {log_loss_model1} \n Model 2: {log_loss_model2}")

#За визначеними параметрами передбачень: точність -  вище в Моделі2, матриця помилок (кількість хибно-позитивних
# та ложно-негативних нижче в Моделі2), втрати логарифмічної функції втрат - менше в Моделі2,
# тобто за всі візначені показники краще в Моделі2. Що означає, що Модель2 трохи точніше передбачає значення,
# а також втрати логарифмічної функції втрат вказує на кращу роботу Моделі2.

















