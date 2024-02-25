from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql import SparkSession


# Initialize SparkSession
spark = SparkSession.builder.appName("777Project").getOrCreate()

# File paths
tracking_file_paths = [
    "tracking_week_9.csv",
    "tracking_week_8.csv",
    "tracking_week_7.csv",
    "tracking_week_6.csv",
    "tracking_week_5.csv",
    "tracking_week_4.csv",
    "tracking_week_3.csv",
    "tracking_week_2.csv",
    "tracking_week_1.csv"
]

additional_file_paths = [
    "tackles.csv",
    "games.csv",
    "players.csv",
    "plays.csv"
]


# Read files into DataFrame
tracking_dfs = []
for i, file_path in enumerate(tracking_file_paths, start=1):
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    df = df.withColumn("week", lit(i))
    tracking_dfs.append(df)

additional_dfs = {}
for file_path in additional_file_paths:
    df_name = file_path.split(".")[0]
    additional_dfs[df_name] = spark.read.csv(file_path, header=True, inferSchema=True)

##########################  Step 1: Merging the Dfs  ###############################
# Perform join operations
# Join the combined tracking DataFrame with games.csv through gameId and week column
tracking_combined_df = tracking_dfs[0]  # Start with the first DataFrame
for i, df in enumerate(tracking_dfs[1:], start=2):
    tracking_combined_df = tracking_combined_df.union(df.withColumn("week", lit(i)))
tracking_combined_df.printSchema()

games_tracking_df = tracking_combined_df.join(additional_dfs["games"], ["gameId", "week"], how="full")
games_tracking_df.printSchema()

# Add plays DataFrame to games_tracking_df with a full join using gameId and playId
combined_df = games_tracking_df.join(additional_dfs["plays"], ["gameId", "playId"], how="full")
combined_df.printSchema()

# Join players.csv to tackles.csv through the column nflId
players_tackles_df = additional_dfs["players"].join(additional_dfs["tackles"], "nflId", how="full")
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

# Read players.csv into a DataFrame
players_df = spark.read.csv("players.csv", header=True)

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