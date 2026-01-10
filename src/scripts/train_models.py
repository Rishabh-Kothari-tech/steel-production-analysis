#fitting models

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_results = evaluate_regression(lr, X_test, y_test)
lr_results

# Support Vector Regression (SVR)
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

svr_results = evaluate_regression(svr, X_test, y_test)
svr_results

# KNN Regressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

knn_results = evaluate_regression(knn, X_test, y_test)
knn_results

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_results = evaluate_regression(dt, X_test, y_test)
dt_results 

# Random Forest Regressor (BEST MODEL)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

rf_results = evaluate_regression(rf, X_test, y_test)
rf_results