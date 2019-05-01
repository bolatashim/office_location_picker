from flask import Flask, render_template, request, jsonify
import engine

app = Flask(__name__)

stars_dist = engine.get_stars_distribution()
all_sweet_spots = engine.get_all_sweet_spots()
all_word_cloud = engine.generate_word_cloud("ALL")
food_word_cloud = engine.generate_word_cloud("FOOD")

review_length_data = engine.get_review_length_data()
engine.generate_tf_idf_df()
local_business_info = None


@app.route("/")
def home():
	states_string = ""
	for state in stars_dist[3]:
		states_string = states_string + (state + ",")
	all_word_cloud_string = ""

	for all_word in all_word_cloud[0]:
		all_word_cloud_string = all_word_cloud_string + (all_word + "%")

	food_word_cloud_string = ""
	for food_word in food_word_cloud[0]:
		food_word_cloud_string = food_word_cloud_string + (food_word + "%")

	review_length_label_string = ""
	review_length_label_count = []
	for i in range(len(review_length_data)):
		review_length_label_string = review_length_label_string + (review_length_data[i][0] + "%")
		review_length_label_count.append(review_length_data[i][1])

	print(food_word_cloud_string)
	print(all_word_cloud_string)
	return render_template("home.html", stars_mean_val=str(stars_dist[0]),\
		                                  stars_list=str(stars_dist[1]),\
		                                  stars_count_list=str(stars_dist[2]),\
		                                  stars_state_list=states_string,\
		                                  stars_state_mean_list=str(stars_dist[4]),\
		                                  stars_state_count_list=str(stars_dist[5]),\
		                                  all_word_cloud_words=all_word_cloud_string,\
		                                  all_word_cloud_counts=str(all_word_cloud[1]),\
		                                  food_word_cloud_words=food_word_cloud_string,\
		                                  food_word_cloud_counts=str(food_word_cloud[1]),\
		                                  review_length_label_string=review_length_label_string,\
		                                  review_length_label_count=str(review_length_label_count))

@app.route("/application")
def application():
	return render_template("application.html", all_sweet_spots=str(all_sweet_spots))


@app.route("/action", methods=['POST'])
def action():
	lat_top = str(request.args.get("lat_top"))
	lat_bot = str(request.args.get("lat_bot"))
	lng_top = str(request.args.get("lng_top"))
	lng_bot = str(request.args.get("lng_bot"))
	global local_business_info
	local_business_info = engine.get_businesses_at(lat_top, lat_bot, lng_top, lng_bot)
	return str(local_business_info[:3])

@app.route("/local_word_cloud", methods=['POST'])
def local_word_cloud():
	if (local_business_info is not None):
		word_cloud_string = ""
		for i in range(len(local_business_info[3])):
		 	word_cloud_string = word_cloud_string + (local_business_info[3][i] + "$$" + str(local_business_info[4][i]) + "%")
		return str(word_cloud_string)
	else:
		return "NOTHING"


@app.route("/local_review_highlights", methods=['POST'])
def local_review_highlights():
	if (local_business_info is not None):
		highlight_string = ""
		for highlight in local_business_info[5]:
		 	highlight_string = highlight_string + (highlight + r"%%%")
		highlight_string = highlight_string + "$$$$"
		for place_name in local_business_info[6]:
		 	highlight_string = highlight_string + (place_name + r"%%%")
		return highlight_string
	else:
		return "NOTHING"


if __name__ == "__main__":
	app.run(debug=True)
