from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import sentiment_mod as s

import json



#        replace mysql.server with "localhost" if you are running via your own server!
#                        server       MySQL username	MySQL pass  Database name.

key_word = input("enter the keyword to be mapped :\n")



#consumer key, consumer secret, access token, access secret.
ckey="c750szg9mvqgZHXqL4k9tnPng"
csecret="sD7W15hKRII4bIT4nGTg3gbx0AZUQN7yFslC7hL0ClAK4iTiry"
atoken="2939876220-K18DMkgJVr5SX4sjtEQj0v6vhaHszoFbXWKf2oJ"
asecret="sg7O6boQNmFZkpapkVawsMWi9HbzA5h9GATnyWvpXRxkp"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        
        tweet = ascii(all_data["text"])
        file_f = open("tweet_out.txt" , "a")
        x = s.sentiment(tweet)
        file_f.write("{}, {}".format(tweet, x))

        file_f.write('\n')
        file_f.close()
        print(tweet, x)
        
        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[key_word])