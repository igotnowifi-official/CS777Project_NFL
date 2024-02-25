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

"""
# Output file
output_directory = "./output"
output_file_name = "combined_output_raw.csv"
output_file_path = os.path.join(output_directory, output_file_name)
#final_combined_df.write.csv(output_file_path, header=True, mode="overwrite")
#final_combined_df.write.parquet(output_file_path, mode="overwrite")
print("Final combined DataFrame has been saved to:", output_file_path)
"""

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
# Define UDF to convert height from foot-inches format to cm
## Help from ChatGPT to reformat
@udf("double")
def feet_to_cm(feet_str):
    if feet_str:
        feet, inches = feet_str.split("-")
        return int(feet) * 30.48 + int(inches) * 2.54
    else:
        return None

# Define UDF to convert weight from pounds to kg
@udf("double")
def pounds_to_kg(pounds):
    if pounds:
        return pounds * 0.453592
    else:
        return None

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

"""
I wanted to make age, height, and weight a feature. But based on my attempt to reformat, I guess it's not meant to be.

# Calculate age based on birth year
current_year = 2024
cleaned_df = cleaned_df.withColumn("birthYear", year("birthDate"))
cleaned_df = cleaned_df.withColumn("age", current_year - col("birthYear"))

# Convert height from foot-inches format to cm
cleaned_df = cleaned_df.withColumn("height_cm", feet_to_cm(col("height")))
cleaned_df = cleaned_df.drop("height")

# Convert weight from pounds to kg
cleaned_df = cleaned_df.withColumn("weight_kg", pounds_to_kg(col("weight")))
cleaned_df = cleaned_df.drop("weight")

# Convert gameClock from hour:minute format to minutes
split_time = split(col("gameClock"), ":")
cleaned_df = cleaned_df.withColumn("gameClock_minutes", split_time.getItem(0).cast("int") * 60 + split_time.getItem(1).cast("int"))
cleaned_df = cleaned_df.drop("gameClock")

# Print the number of rows and columns left
print("Number of rows and columns after cleanup:", cleaned_df.count(), len(cleaned_df.columns))
"""
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

"""
There are too many rows and too heavy, let's just remove everything and not worry about imputing

# Step 3: Remove columns with more than 60% missing values
missing_threshold = cleaned_df.count() * 0.6
cleaned_df = cleaned_df.drop(*[
    col_name for col_name in cleaned_df.columns
    if cleaned_df.filter(col(col_name).isNull()).count() > missing_threshold
])

# Print number of rows and columns after Step 3
print("Number of rows after Step 3:", cleaned_df.count())
print("Number of columns after Step 3:", len(cleaned_df.columns))

# Step 4: Use KNN Imputer to impute missing values in remaining columns
imputer = Imputer(strategy="knn", k=5)
cleaned_df = imputer.fit(cleaned_df).transform(cleaned_df)

# Print number of rows and columns after Step 4
print("Number of rows after Step 4:", cleaned_df.count())
print("Number of columns after Step 4:", len(cleaned_df.columns))
"""

# Step 3: Remove missing values
## Note: weird part, works but sometimes something weird and crash.
# Remove rows with any missing values
cleaned_df2 = cleaned_df.na.drop()

# Print number of rows and columns after removing missing values
print("Number of rows after step 3:", cleaned_df2.count())
print("Number of columns after step 3:", len(cleaned_df2.columns))

"""
Remove because too heavy for local env; in PCA we depend.

# Step 4: Remove columns with zero variance
# List of features to ignore during variance thresholding
features_to_ignore = ["Class","playDirection","o","dir","passLength","offenseFormation","defendersInTheBox"]  # Add other features if needed

# Calculate variance for each feature
feature_variances = cleaned_df2.select([variance(col).alias(col) for col in cleaned_df2.columns if col not in features_to_ignore])

# Set a threshold for variance
threshold = 0.0  # Adjust as needed

# Filter out features with variance below the threshold
selected_features = [col for col in feature_variances.columns if feature_variances.select(col).first()[col] > threshold]

# Select only the features with non-zero variance
cleaned_df2 = cleaned_df2.select(selected_features)

# Show the schema of the DataFrame after variance thresholding
cleaned_df2.printSchema()
print("Number of rows after step 4:", cleaned_df2.count())
print("Number of columns after step 4:", len(cleaned_df2.columns))

"""
"""
To decide whether we want to use Vector Assembler for modelling

# Step 4: Vector Assembler (optional if you want to use a vector representation for modeling)
assembler = VectorAssembler(inputCols=[col for col in cleaned_df.columns if col not in ['tackle', 'pff_missedTackle']],
                            outputCol="features")
cleaned_df = assembler.transform(cleaned_df)

"""

"""
To move to be done with modeling

# 4. Perform PCA for dimensionality reduction
assembler = VectorAssembler(inputCols=cleaned_df.columns, outputCol="features")
cleaned_df = assembler.transform(cleaned_df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scalerModel = scaler.fit(cleaned_df)
cleaned_df = scalerModel.transform(cleaned_df)

pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
model = pca.fit(cleaned_df)
cleaned_df = model.transform(cleaned_df)

# Print number of rows and columns after PCA
print("Number of rows after PCA:", cleaned_df.count())
print("Number of columns after PCA:", len(cleaned_df.columns))
 VectorAssembler(inputCols=cleaned_df.columns, outputCol="features")
cleaned_df = assembler.transform(cleaned_df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scalerModel = scaler.fit(cleaned_df)
cleaned_df = scalerModel.transform(cleaned_df)

pca = PCA(k=10, inputCol="scaledFeatures", outputCol="pcaFeatures")
model = pca.fit(cleaned_df)
cleaned_df = model.transform(cleaned_df)

# Print number of rows and columns after PCA
print("Number of rows after PCA:", cleaned_df.count())
print("Number of columns after PCA:", len(cleaned_df.columns))
"""

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

        """
        Refuse to work
        # Define a UDF to extract the probability for the positive class
        get_positive_probability = udf(lambda prob: float(prob[1]), DoubleType())

        # Add a new column to the DataFrame containing the probability for the positive class
        predictions = predictions.withColumn('positive_probability', get_positive_probability('probability'))

        pred_scores = predictions.select('positive_probability', 'Class').rdd.map(
            lambda x: (float(x['positive_probability']), float(x['Class'])))

        # Calculate AUC manually-had help from GPT Almighty
        pred_scores = predictions.select('probability').rdd.map(
            lambda x: (float(x['probability'][1]), float(x['Class'])))
        sorted_pred_scores = pred_scores.sortByKey().collect()
        num_pos = float(sum([x[1] for x in sorted_pred_scores]))
        num_neg = float(len(sorted_pred_scores) - num_pos)
        tpr_list, fpr_list, prev_fpr = [0.0], [0.0], 0.0
        auc = 0.0
        for score in sorted_pred_scores:
            if score[1] == 1:
                tpr_list.append(tpr_list[-1] + 1.0 / num_pos)
                fpr_list.append(fpr_list[-1])
            else:
                tpr_list.append(tpr_list[-1])
                fpr_list.append(fpr_list[-1] + 1.0 / num_neg)
                auc += (fpr_list[-1] - prev_fpr) * tpr_list[-1]
                prev_fpr = fpr_list[-1]
        # Plot ROC curve manually
        plt.figure()
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - ' + model_name)
        plt.legend(loc="lower right")
        plt.show()
        """
        # Print evaluation metrics
        print("\nEvaluation Metrics for", model_name, "with PCA")
        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score)
        print("True Positive Rate (TPR):", tpr)
        print("True Negative Rate (TNR):", tnr)
        #print("AUC:", auc)

        # Store evaluation results in dictionary
        evaluation_results[model_name] = {
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1 Score": f1_score,
            "True Positive Rate (TPR)": tpr,
            "True Negative Rate (TNR)": tnr,
            #"AUC": auc
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