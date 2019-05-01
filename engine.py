import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col, split, explode, udf, max, desc
from pyspark.sql.types import ArrayType, DoubleType, StringType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.ml.linalg import SparseVector



spark = SparkSession.builder.appName('app').getOrCreate()

df_original = spark.read.json("static/data/yelp_academic_dataset_business.json")


# df_reviews = spark.read.json("static/data/yelp_academic_dataset_review.json")
# df_reviews = df_reviews.withColumn('review_length', length(col('text')))

# SEE THE BOTTOM oF FILE FOR EXPLANATION ON HOW 
# yelp_academic_dataset_review_tiny.json was created
df_reviews_tiny = spark.read.json("static/data/yelp_academic_dataset_review_tiny.json")

tf_idf_reviews = None


# Filter Out the Businesses that have to do with Food and Drinks
df = df_original.filter(r"lower(categories) like '%food%' OR lower(categories) like '%drink%'" )
state_to_idx = {'AB':0, 'AZ':1, 'IL':2, 'NC':3, 'NV':4, 'OH':5, 'ON':6, 'PA':7, 'QC':8, 'SC':9, 'WI':10}
dummy_food_categories_filter = "lower(cats) != 'restaurants' AND lower(cats) != 'food' AND lower(cats) != 'cafes'"


def get_businesses_at(lat_top, lat_bot, lng_top, lng_bot):
	# Conditions for filtering out the date relevant to the box
	location_filter = "latitude < " + lat_top + " AND latitude > " + lat_bot + " AND " + \
                    "longitude < " + lng_top + " AND longitude > " + lng_bot
	
	local_business_filtered = df.filter(location_filter)
	local_business_df = local_business_filtered.select(["latitude", "longitude", "stars", "business_id", "name"]).sort(col("stars").asc())
	
	#TODO join with reviews to showcase TF-IDF on review
	local_business_count = local_business_df.count()
	
	if (local_business_count == 0):
		return str([0, 0, [[0,0,0]]]) # If there no businesses in the area, return default

	nicest_businesses = [[x[0], x[1]] for x in (local_business_df.sort(desc("stars"))).select("business_id", "name").take(5)]
	review_filter_condition = "business_id in ("
	for i in range(len(nicest_businesses)):
		review_filter_condition = (review_filter_condition + "'" + nicest_businesses[i][0] + "'")
		if (i < (len(nicest_businesses)-1)):
			review_filter_condition = review_filter_condition + ", "
	review_filter_condition = review_filter_condition + ")"

	review_highlights = [x[0] for x in (tf_idf_reviews.filter(review_filter_condition).select("raw_sentence")).take(len(nicest_businesses))]
	review_place_names = [x[1] for x in nicest_businesses]


	local_business_mean_stars = local_business_df.select(mean("stars")).collect()[0][0]	
	local_businesses_locations = [[float(x[0]), float(x[1]), float(x[2])] for x in local_business_df.collect()]
	word_cloud = generate_word_cloud(local_business_filtered)
	local_business_info = [local_business_count, local_business_mean_stars, local_businesses_locations, word_cloud[0], word_cloud[1], review_highlights, review_place_names]

	return local_business_info

def get_stars_distribution():
	stars_mean_val = df.select(mean("stars")).collect()[0][0]
	stars_list = [x[0] for x in df.groupBy("stars").count().sort(col("stars").desc()).collect()]
	stars_count_list = [x[1] for x in df.groupBy("stars").count().sort(col("stars").desc()).collect()]
	stars_count_state = df.groupBy("state").count().filter("`count` >= 30").sort(col("state").asc())
	stars_mean_state = df.groupBy("state").mean("stars").sort(col("state").asc())
	stars_mean_count_join = stars_count_state.join(stars_mean_state, "state")
	stars_state_list = [str(x[0]) for x in stars_mean_count_join.collect()]
	stars_state_mean_list = [x[2] for x in stars_mean_count_join.collect()]
	stars_state_count_list = [x[1] for x in stars_mean_count_join.collect()]
	stars_distribution = [stars_mean_val, stars_list, stars_count_list, stars_state_list, stars_state_mean_list, stars_state_count_list]
	
	return stars_distribution

def get_all_sweet_spots():
	major_state_spots = df.groupBy("state").count().filter("`count` >= 30")
	df_ = df.join(major_state_spots, "state")
	good_quality_businesses_df = df_.filter("stars == 4.0").select(["latitude", "longitude", "state"])
	better_quality_businesses_df = df_.filter("stars == 4.5").select(["latitude", "longitude", "state"])
	best_quality_businesses_df = df_.filter("stars == 5.0").select(["latitude", "longitude", "state"])
	good_quality_businesses_list = [[x[0], x[1], state_to_idx[x[2]]] for x in good_quality_businesses_df.collect()]
	better_quality_businesses_list = [[x[0], x[1], state_to_idx[x[2]]] for x in better_quality_businesses_df.collect()]
	best_quality_businesses_list = [[x[0], x[1], state_to_idx[x[2]]] for x in best_quality_businesses_df.collect()]
	sweet_spots = [good_quality_businesses_list, better_quality_businesses_list, best_quality_businesses_list]
	return sweet_spots


def generate_word_cloud(data_df):
	cloud_size = 100
	do_filter = False
	if (data_df == "ALL"):
		data_df = df_original
	elif (data_df == "FOOD"):
		data_df = df
	else:
		cloud_size = 20
		do_filter = True
	categories_explode = data_df.select(explode(split(col("categories"), ", ")).alias("cats"))
	if (do_filter):
		categories_explode = categories_explode.filter(dummy_food_categories_filter)
	categories_explode = categories_explode.groupBy("cats").count().sort(col("count").desc()).take(cloud_size)
	top_word_values = [x[0] for x in categories_explode]
	top_word_counts = [x[1] for x in categories_explode]
	word_cloud = [top_word_values, top_word_counts]
	return word_cloud


def get_review_length_data():

	# EXPLANATION : COMPUTATION OMITTED DUE TO WEAKNESS OF TESTING COMPUTER

	# reviews_by_length = [["4000-5000", df_reviews.filter("review_length > 4000").count()],\
	#                      ["3000-4000", df_reviews.filter("review_length > 3000 AND review_length <= 4000").count()],\
	#                      ["2000-3000", df_reviews.filter("review_length > 2000 AND review_length <= 3000").count()],\
	#                      ["1000-2000", df_reviews.filter("review_length > 1000 AND review_length <= 2000").count()],\
	#                      ["0-1000",    df_reviews.filter("review_length > 0 AND review_length <= 1000").count()]]

	
	precomputed_reviews_by_length = [["4000-5000", 19211],\
	                                 ["3000-4000", 37628],\
	                                 ["2000-3000", 150570],\
	                                 ["1000-2000", 859920],\
	                                 ["0-1000", 5618571]]

	return precomputed_reviews_by_length

def extract_values_from_vector(vector):
    return vector.values.tolist()

def generate_tf_idf_df():
	global tf_idf_reviews
	ddf = df_reviews_tiny.withColumn("raw_sentence", explode(split(col("text"), r"\.")))

	def extract_values_from_vector(vector):
	    return vector.values.tolist()

	strip_sentences = udf(lambda sent: re.sub(r'[^\w ]', '', sent), StringType())
	sum_sparse_vector = udf(lambda a: sum(a[0:len(a)]), DoubleType())
	extract_values_from_vector_udf = udf(lambda x: extract_values_from_vector(x), ArrayType(DoubleType()))

	ddf = ddf.withColumn("sentence", strip_sentences(col("raw_sentence")))

	tokenizer = RegexTokenizer(inputCol="sentence", outputCol="token_sentence")
	remover = StopWordsRemover(inputCol="token_sentence", outputCol="filtered")

	ddf_tokenized = tokenizer.transform(ddf)
	ddf_clean = remover.transform(ddf_tokenized)

	ddf_clean = ddf_clean.select(["business_id", "raw_sentence", "filtered"])

	hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1000)
	featurizedData = hashingTF.transform(ddf_clean)

	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)


	final = rescaledData.withColumn("vals", extract_values_from_vector_udf(col('features')))
	final = final.withColumn("rank", sum_sparse_vector(col("vals")))
	final = final.filter(final.rank.isNotNull())
	final = final.select(["business_id", "raw_sentence", "rank"])

	max_group = (final.groupBy("business_id").max("rank")).withColumnRenamed("max(rank)", "rank_").withColumnRenamed("business_id", "business_id_")
	tf_idf_reviews = max_group.join(final, (final.rank ==  max_group.rank_) & (final.business_id ==  max_group.business_id_)).drop("business_id_").drop("rank_")




# EXPLANATION ON yelp_academic_dataset_review_tiny.json file
# 
# Download from : link
#
# WHY :
# Computer Used for the Project was MacBook Air (4GB RAM, 1.4 GHz), 
# whose computational capacity leads to 10min delays for spark-local 
# computations on original yelp_academic_dataset_review.json dataset.
# Therefore, the I filtered the file and kept the data necessary for 
# my project (5.6G --> 39MB). The procedure I used is demostrated below.
# 
# HOW :
# df_review = spark.read.json("static/data/yelp_academic_dataset_review.json")
# df_business = spark.read.json("static/data/yelp_academic_dataset_business.json")
# df_business_ids = df_business.filter(r"lower(categories) like '%food%' OR lower(categories) like '%drink%'" ).select("business_id")

# most_useful_reviews = df_review.groupBy('business_id').max('useful')
# most_useful_reviews = most_useful_reviews.select(["business_id", col("max(useful)").alias("useful")])
#
# most_useful_reviews = df.select(["business_id", "text", "useful"])
# most_useful_reviews = most_useful_reviews.withColumn('review_length', length(col('text')))
#
# most_useful_distinct_reviews = most_useful_reviews.join(df, ["business_id", "useful"])
# most_useful_distinct_reviews = most_useful_distinct_reviews.dropDuplicates(["business_id"])
# df_review = df_business_ids.join(most_useful_distinct_reviews, "business_id")








