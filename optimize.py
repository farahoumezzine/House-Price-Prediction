from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datapreproce import X_train, X_test, y_train, y_test, scaler
import joblib #to ensure the same scaling is applied during prediction.

# Optimisation de Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
#(Scikit-learn uses negative MSE because it maximizes scores,
#  but we want to minimize MSE, so the negative is used.)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Meilleurs paramètres:", grid_search.best_params_)
print("MSE optimisé:", mean_squared_error(y_test, y_pred))
print("R2 optimisé:", r2_score(y_test, y_pred))

joblib.dump(best_rf, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler as well