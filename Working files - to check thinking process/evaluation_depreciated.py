
def train_and_evaluate_with_pca(model_name, model, train_df, test_df, feature_columns):
    # Apply PCA to reduce dimensionality
    pca = PCA(k=10, inputCol= feature_columns, outputCol="pca_features")
    pca_model = pca.fit(train_df)
    train_df_pca = pca_model.transform(train_df)
    test_df_pca = pca_model.transform(test_df)

    # Start training time
    start_train_time = time.time()

    # Train model
    model_pca = model.fit(train_df_pca)

    # End training time
    end_train_time = time.time()

    # Start evaluation time
    start_eval_time = time.time()

    # Make predictions
    predictions = model_pca.transform(test_df_pca)

    # End evaluation time
    end_eval_time = time.time()

    # Calculate evaluation metrics
    tp = predictions.filter("prediction = 1 AND Class = 1").count()
    fp = predictions.filter("prediction = 1 AND Class = 0").count()
    tn = predictions.filter("prediction = 0 AND Class = 0").count()
    fn = predictions.filter("prediction = 0 AND Class = 1").count()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    confusion_matrix = [[tn, fp], [fn, tp]]

    # Calculate AUC
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Class")
    auc = evaluator.evaluate(predictions)

    # Print performance metrics
    print("\nPerformance Metrics:", model_name, "with PCA")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("Confusion Matrix:", confusion_matrix)
    print("AUC:", auc)

    # Calculate total times
    total_train_time = end_train_time - start_train_time
    total_eval_time = end_eval_time - start_eval_time
    total_time = total_train_time + total_eval_time

    print("Total Train Time:", total_train_time, "secs")
    print("Total Evaluation Time:", total_eval_time, "secs")
    print("Total Time:", total_time, "secs")

    # Plot ROC curve
    roc_data = predictions.select("Class", "probability").rdd.map(lambda row: (float(row["probability"][1]), float(row["Class"])))
    roc_df = spark.createDataFrame(roc_data, ["probability", "Class"])
    roc_auc = BinaryClassificationEvaluator(rawPredictionCol="probability", metricName="areaUnderROC").evaluate(predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": confusion_matrix,
        "auc": auc,
        "total_train_time": total_train_time,
        "total_eval_time": total_eval_time,
        "total_time": total_time,
        "roc_auc": roc_auc
    }