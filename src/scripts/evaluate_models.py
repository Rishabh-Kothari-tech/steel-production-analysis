# Compare All Models (RESULT TABLE)

results_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "SVR",
        "KNN Regressor",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        lr_results[0],
        svr_results[0],
        knn_results[0],
        dt_results[0],
        rf_results[0]
    ],
    "MSE": [
        lr_results[1],
        svr_results[1],
        knn_results[1],
        dt_results[1],
        rf_results[1]
    ],
    "RMSE": [
        lr_results[2],
        svr_results[2],
        knn_results[2],
        dt_results[2],
        rf_results[2]
    ],
    "R2 Score": [
        lr_results[3],
        svr_results[3],
        knn_results[3],
        dt_results[3],
        rf_results[3]
    ]
})

results_df



# R² Comparison

plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R2 Score", data=results_df)
plt.xticks(rotation=45)
plt.title("Model Comparison using R² Score")
plt.show()



# Actual vs Predicted Plot 

y_pred_rf = rf.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title("Actual vs Predicted (Random Forest)")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")
plt.show()
