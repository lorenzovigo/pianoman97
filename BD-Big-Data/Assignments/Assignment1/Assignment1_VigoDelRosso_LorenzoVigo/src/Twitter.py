from twython import TwythonStreamer
import json
import pymongo


# Enter your keys/secrets as strings in the following fields
credentials = {}
credentials['CONSUMER_KEY'] =
credentials['CONSUMER_SECRET'] =
credentials['ACCESS_TOKEN'] =
credentials['ACCESS_SECRET'] =


# Connection to MongoDB
client = pymongo.MongoClient('mongodb://mongo:27017/')
db = client["twitterdb"]
tweets = db["tweets"]


# Filter out unwanted data
def process_tweet(tweet):
    d = {}
    d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['user'] = tweet['user']
    d['created_at']=tweet['created_at']
    d['geo']=tweet['geo']
    d['reply_count']=tweet['reply_count']
    d['retweet_count']=tweet['retweet_count']
    d['favorite_count']=tweet['favorite_count']
    d['id']=tweet['id_str']
    d['in_reply_to_status_id']=tweet['in_reply_to_status_id_str']
    d['in_reply_to_user_id_str']=tweet['in_reply_to_user_id_str']
    d['text']=tweet['text']
    return d
    
    
# Create a class that inherits TwythonStreamer
class MyStreamer(TwythonStreamer):     

    count=0
    batch = []

    # Received data
    def on_success(self, data):
        # Only collect tweets in English
        if data['lang'] == 'en':
            tweet_data = process_tweet(data)
            self.batch.append(tweet_data)
            self.count+=1
            if(self.count%5==0):
                self.save_to_mongo()
            

    # Problem with the API
    def on_error(self, status_code, data, headers):
        print(status_code, data)
        self.disconnect()
        
    # Save each tweet to a file
    def save_to_file(self, tweet):
        with open('corona_tweet.json', 'a') as fp:
            json.dump(tweet, fp)
            fp.write("\n")

    # Save tweets to mongo in batches of size 20
    def save_to_mongo(self):
        tweets.insert_many(self.batch)
        self.batch = []


# Instantiate from our streaming class
stream = MyStreamer(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'], 
                    credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
# Start the stream
stream.statuses.filter(track='corona')

# Stop execution anytime by stoping the cell execution
