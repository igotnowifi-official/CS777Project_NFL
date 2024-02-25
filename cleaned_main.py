import os
from pyspark.ml.feature import VectorAssembler,  PCA
from pyspark.sql.functions import col, when, lit, udf, expr
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier, NaiveBayes, LinearSVC
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import ArrayType, DoubleType


# Initialize SparkSession
spark = SparkSession.builder \
    .appName("777Project") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .config("fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
    .getOrCreate()

# File paths
tracking_file_paths = [
    "gs://liewnatasyacs777assignment/tracking_week_9.csv",
    "gs://liewnatasyacs777assignment/tracking_week_8.csv",
    "gs://liewnatasyacs777assignment/tracking_week_7.csv",
    "gs://liewnatasyacs777assignment/tracking_week_6.csv",
    "gs://liewnatasyacs777assignment/tracking_week_5.csv",
    "gs://liewnatasyacs777assignment/tracking_week_4.csv",
    "gs://liewnatasyacs777assignment/tracking_week_3.csv",
    "gs://liewnatasyacs777assignment/tracking_week_2.csv",
    "gs://liewnatasyacs777assignment/tracking_week_1.csv"
]

additional_file_paths = [
    "gs://liewnatasyacs777assignment/tackles.csv",
    "gs://liewnatasyacs777assignment/games.csv",
    "gs://liewnatasyacs777assignment/players.csv",
    "gs://liewnatasyacs777assignment/plays.csv"
]

# Read files into DataFrame
tracking_dfs = []
for i, file_path in enumerate(tracking_file_paths, start=1):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df.withColumn("week", lit(i))
    tracking_dfs.append(df)

additional_dfs = {}
for file_path in additional_file_paths:
    df_name = os.path.splitext(os.path.basename(file_path))[0]
    additional_dfs[df_name] = spark.read.csv(file_path, header=True, inferSchema=True)

tackles_df = additional_dfs['tackles']
games_df = additional_dfs['games']
players_df = additional_dfs['players']
plays_df = additional_dfs['plays']



##########################  Step 1: Merging the Dfs  ###############################
# Perform join operations
# Join the combined tracking DataFrame with games.csv through gameId and week column
tracking_combined_df = tracking_dfs[0]  # Start with the first DataFrame
for i, df in enumerate(tracking_dfs[1:], start=2):
    tracking_combined_df = tracking_combined_df.union(df.withColumn("week", lit(i)))
tracking_combined_df.printSchema()

games_tracking_df = tracking_combined_df.join(games_df, ["gameId", "week"], how="full")
games_tracking_df.printSchema()

# Add plays DataFrame to games_tracking_df with a full join using gameId and playId
combined_df = games_tracking_df.join(plays_df, ["gameId", "playId"], how="full")
combined_df.printSchema()

# Join players.csv to tackles.csv through the column nflId
players_tackles_df = players_df.join(tackles_df, "nflId", how="full")
players_tackles_df.printSchema()

# Combine the DataFrames using a full join through gameId, nflId, and playId
final_combined_df = combined_df.join(players_tackles_df, ["gameId", "nflId", "playId"], how="full")
final_combined_df.printSchema()



##########################  Step 2: Preprocessing  ###############################
# Step 1: Remove rows where values of tackle and pff_missedTackle column are not 1
tackle_condition = col("tackle") != 1
missed_tackle_condition = col("pff_missedTackle") != 1
combined_condition = tackle_condition | missed_tackle_condition
filter_condition = when(combined_condition, True).otherwise(False)
filtered_df = final_combined_df.filter(filter_condition)

# Step 2: Create a new column "Class" based on tackle and pff_missedTackle columns
final_combined_df = final_combined_df.withColumn("Class",
                                                 when(final_combined_df["tackle"] == 1, 1)
                                                 .when(final_combined_df["pff_missedTackle"] == 1, 0)
                                                 .otherwise(None))
# Remove tackle and pff_missedTackle columns
final_combined_df = final_combined_df.drop("tackle", "pff_missedTackle")

# Print number of rows and columns after Step 1 and Step 2
print("Number of rows after Step 1 and Step 2:", final_combined_df.count())
print("Number of columns after Step 1 and Step 2:", len(final_combined_df.columns))


################## Small Tester ######################
"""
Research Question:
1. Print the Top 3 teams that are best at tackling.
2. Print the Top 3 teams that missed the most tackles.
3. Print the Top 3 players that are the best at tackling.
4. Print the Top 3 players that missed the most tackles.
"""
# Top 3 teams best at tackling
top_teams_tackling = final_combined_df.filter(final_combined_df['Class'] == 1).groupBy('club').count().orderBy('count', ascending=False).limit(3)
print("Top 3 teams best at tackling:")
top_teams_tackling.show()

# Top 3 teams that missed the most tackles
top_teams_missed_tackles = final_combined_df.filter(final_combined_df['Class'] == 0).groupBy('club').count().orderBy('count', ascending=False).limit(3)
print("Top 3 teams that missed the most tackles:")
top_teams_missed_tackles.show()


# Join final_combined_df with players_df to get displayName
final_with_displayName_tackling = final_combined_df.filter(final_combined_df['Class'] == 1)\
    .groupBy('nflId').count()\
    .join(players_df.select('nflId', 'displayName'), 'nflId', 'left')\
    .orderBy('count', ascending=False).limit(3)\
    .select('nflId', 'displayName', 'count')\
    .orderBy('count', ascending=False)
print("Top 3 players best at tackling:")
final_with_displayName_tackling.show()

final_with_displayName_missed_tackles = final_combined_df.filter(final_combined_df['Class'] == 0)\
    .groupBy('nflId').count()\
    .join(players_df.select('nflId', 'displayName'), 'nflId', 'left')\
    .orderBy('count', ascending=False).limit(3)\
    .select('nflId', 'displayName', 'count')\
    .orderBy('count', ascending=False)
print("Top 3 players that missed the most tackles:")
final_with_displayName_missed_tackles.show()

######################################################

# Drop specified columns
columns_to_drop = [
    "ballCarrierId", "ballCarrierDisplayName", "playDescription", "penaltyYards",
    "foulName1", "foulName2", "foulNFLId1", "foulNFLId2", "displayName", "season",
    "gameDate", "gameTimeEastern", "frameId", "time", "jerseyNumber", "event", "x", "y",
    "s", "a", "passResult", "expectedPoints", "visitorTeamWinProbilityAdded",
    "homeTeamWinProbabilityAdded", "homeTeamAbbr","visitorTeamAbbr","homeFinalScore","visitorFinalScore","collegeName",
    "possessionTeam", "defensiveTeam", "yardlineSide", "club"
]
cleaned_df = final_combined_df.drop(*columns_to_drop)

# Print number of rows and columns after dropping columns
print("Number of rows after dropping columns:", cleaned_df.count())
print("Number of columns after dropping columns:", len(cleaned_df.columns))

# Remove rows where playNullifiedByPenalty == 'Y'
cleaned_df = cleaned_df.filter(col("playNullifiedByPenalty") != "Y")

print("Number of rows after dropping penalty:", cleaned_df.count())
print("Number of columns after dropping penalty:", len(cleaned_df.columns))

# Drop specified columns
columns_to_drop = [
    "birthDate", "height", "weight", "gameClock","gameId","playId","nflId","playNullifiedByPenalty","passProbability","expectedPointsAdded","position","offenseFormation","week","prePenaltyPlayResult","playResult"
]
cleaned_df = cleaned_df.drop(*columns_to_drop)

# Convert columns to float
cleaned_df = cleaned_df.withColumn("o", col("o").cast("float"))
cleaned_df = cleaned_df.withColumn("dir", col("dir").cast("float"))
cleaned_df = cleaned_df.withColumn("passLength", col("passLength").cast("float"))
cleaned_df = cleaned_df.withColumn("defendersInTheBox", col("defendersInTheBox").cast("float"))

# Convert playDirection to 0 if 'left' and 1 if 'right'
cleaned_df = cleaned_df.withColumn("playDirection", when(col("playDirection") == "left", 0).otherwise(1))

# Print number of rows and columns after dropping columns
print("Number of rows after dropping columns round 2:", cleaned_df.count())
print("Number of columns after dropping columns round 2:", len(cleaned_df.columns))

# Step 3: Remove missing values
## Note: weird part, works but sometimes something weird and crash.
# Remove rows with any missing values
cleaned_df2 = cleaned_df.na.drop()

# Print number of rows and columns after removing missing values
print("Number of rows after step 3:", cleaned_df2.count())
print("Number of columns after step 3:", len(cleaned_df2.columns))

print("\nFinal DataFrame Schema:")
cleaned_df2.printSchema()
output_file_path = "gs://liewnatasyacs777assignment/cleaned_df2_output.csv"
cleaned_df2.write.csv(output_file_path, header=True, mode="overwrite")



print("\nFinal DataFrame Schema:")
cleaned_df2.printSchema()
output_file_path = "gs://liewnatasyacs777assignment/cleaned_df2_output.csv"
cleaned_df2.write.csv(output_file_path, header=True, mode="overwrite")

################## Research Quesitons to ask for the Next Section #####################
"""
Research Question:
1. What are the Top 3 features to identify tackles? Which one is the top 3 features to identify Missed Tackles
2. What are the Bottom 3 features to identify tackles? Which one is the top 3 features to identify Missed Tackles
3. Which model and technique is the best to predict a successful Tackle?
4. Use precision, confusion matrix, TPR, TNR, F1-Score, AUC, ROC, and Accuracy of each model using PCA. To note for report: cancel boosting method for hyperparameter tuning. Not enough time.
"""
#################################################



################# Train-Test Split and PCA ############################
# Balancing using Oversampling
class_counts = cleaned_df2.groupBy('Class').count()

if class_counts.count() == 0:
    print("No data found in class_counts DataFrame. Exiting.")
    exit()

minority_class_label = class_counts.orderBy(col('count').asc()).first()[0]
minority_class_count = class_counts.orderBy(col('count').asc()).first()[1]
majority_class_label = class_counts.orderBy(col('count').desc()).first()[0]
majority_class_count = class_counts.orderBy(col('count').desc()).first()[1]

class_count_diff = majority_class_count - minority_class_count
oversampled_minority = cleaned_df2.filter(col('Class') == minority_class_label).sample(True, class_count_diff / minority_class_count, seed=42)
balanced_df = cleaned_df2.union(oversampled_minority)


# Train-Test Split
train_df_raw, test_df_raw = balanced_df.randomSplit([0.8, 0.2], seed=37) # Using last BUID 2 digit for seed

# Assemble features in vectors
features = balanced_df.columns
target = balanced_df["Class"]

# Assemble features into a single vector column
assembler = VectorAssembler(inputCols=features, outputCol="features_vector")
train_df = assembler.transform(train_df_raw)
train_df.printSchema()
test_df = assembler.transform(test_df_raw)
test_df.printSchema()

# PCA
pca = PCA(k=10, inputCol="features_vector", outputCol="pca_features")
pipeline = Pipeline(stages=[pca])
pipeline_model = pipeline.fit(train_df)

# Transform the data using the fitted pipeline
train_df_pca = pipeline_model.transform(train_df)
train_df_pca.printSchema()
test_df_pca = pipeline_model.transform(test_df)
test_df_pca.printSchema()

# Select only the pca_features column
train_df_pca_selected = train_df_pca.select("pca_features")
test_df_pca_selected = test_df_pca.select("pca_features")

# Define a UDF to convert VectorUDT to ArrayType
vector_to_array = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

# Convert struct column to array of doubles
train_df_pca_selected_print = train_df_pca_selected.withColumn("pca_features_array", vector_to_array("pca_features"))
test_df_pca_selected_print = test_df_pca_selected.withColumn("pca_features_array", vector_to_array("pca_features"))

# Convert array of doubles to string representation
train_df_pca_selected_print = train_df_pca_selected_print.withColumn("pca_features_str", expr("to_json(pca_features_array)"))
test_df_pca_selected_print = test_df_pca_selected_print.withColumn("pca_features_str", expr("to_json(pca_features_array)"))

# Select only the string representation column
train_df_pca_selected_print = train_df_pca_selected_print.select("pca_features_str")
test_df_pca_selected_print = test_df_pca_selected_print.select("pca_features_str")

# Define output paths in Google Cloud Storage
output_path_train_df = "gs://liewnatasyacs777assignment/train_df.csv"
output_path_test_df = "gs://liewnatasyacs777assignment/test_df.csv"
output_path_train_df_pca = "gs://liewnatasyacs777assignment/train_df_pca.csv"
output_path_test_df_pca = "gs://liewnatasyacs777assignment/test_df_pca.csv"

# Write DataFrames to CSV files in Google Cloud Storage
train_df_raw.write.csv(output_path_train_df, header=True, mode="overwrite")
test_df_raw.write.csv(output_path_test_df, header=True, mode="overwrite")
train_df_pca_selected_print.write.csv(output_path_train_df_pca, header=True, mode="overwrite")
test_df_pca_selected_print.write.csv(output_path_test_df_pca, header=True, mode="overwrite")



##################### Modelling ###############################

# Restructure DataFrame to have separate columns for features and label
train_df_pca = train_df_pca.select("pca_features", "Class")
test_df_pca = test_df_pca.select("pca_features", "Class")

train_df_pca.printSchema()
test_df_pca.printSchema()
# Define models
models_with_pca = {
    "Logistic Regression with PCA": LogisticRegression(featuresCol="pca_features", labelCol="Class"),
    "Random Forest with PCA": RandomForestClassifier(featuresCol="pca_features", labelCol="Class"),
    "Decision Tree with PCA": DecisionTreeClassifier(featuresCol="pca_features", labelCol="Class"),
    "Linear SVC with PCA": LinearSVC(featuresCol="pca_features", labelCol="Class")
}

evaluation_results = {}

# Train and evaluate models
for model_name, model in models_with_pca.items():
    try:
        # Train model
        model_pca = model.fit(train_df_pca)

        # Make predictions
        predictions = model_pca.transform(test_df_pca)
        predictions.select("prediction", "Class").show(5)
        predictions = predictions.withColumn("prediction", col("prediction").cast("integer"))
        predictions.printSchema()

        # Calculate evaluation metrics
        tp = predictions.filter((predictions["prediction"] == 1) & (predictions.Class == 1)).count()
        fp = predictions.filter((predictions["prediction"] == 1) & (predictions.Class == 0)).count()
        tn = predictions.filter((predictions["prediction"] == 0) & (predictions.Class == 0)).count()
        fn = predictions.filter((predictions["prediction"] == 0) & (predictions.Class == 1)).count()
        print(tp)
        print(fp)
        print(tn)
        print(fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = 2 * precision * recall / (precision + recall)
        tpr = recall
        tnr = tn / (tn + fp)
        print(f1_score)

        # Print evaluation metrics
        print("\nEvaluation Metrics for", model_name, "with PCA")
        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score)
        print("True Positive Rate (TPR):", tpr)
        print("True Negative Rate (TNR):", tnr)
        # print("AUC:", auc)

        # Store evaluation results in dictionary
        evaluation_results[model_name] = {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1 Score": f1_score,
            "True Positive Rate (TPR)": tpr,
            "True Negative Rate (TNR)": tnr,
            # "AUC": auc
        }

        # Extract feature importance
        if hasattr(model_pca, "featureImportances"):
            feature_importance = model_pca.featureImportances.toArray()
            feature_importance = [abs(importance) for importance in feature_importance]

        elif hasattr(model_pca, "coefficients"):
            feature_importance = model_pca.coefficients.toArray()
            feature_importance = [abs(importance) for importance in feature_importance]

        else:
            raise ValueError("Model does not have attribute for feature importance.")

        # Map feature importance to feature names
        feature_importance_map = [(feature, importance) for feature, importance in zip(features, feature_importance)]

        # Sort feature importance in descending order
        sorted_feature_importance = sorted(feature_importance_map, key=lambda x: x[1], reverse=True)

        # Extract top 3 and bottom 3 features
        top_features = sorted_feature_importance[:3]
        bottom_features = sorted_feature_importance[-3:]

        # Print top and bottom features
        print("\nTop 3 features for", model_name, "to identify tackles:")
        for feature, importance in top_features:
            print(feature, ":", importance)

        print("\nBottom 3 features for", model_name, "to identify tackles:")
        for feature, importance in bottom_features:
            print(feature, ":", importance)

    except Exception as e:
        print("An error occurred for model", model_name, ":", e)

# Stop Spark session
spark.stop()