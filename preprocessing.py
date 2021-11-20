def clean_tweet(tweet):
    #tweet = re.sub(r"@[A-Za-z0-9]+",' ', tweet)
    tweet = re.sub(r"https?://[A-Za-z0-9./]+",' <url> ', tweet)
    tweet = re.sub(r" +", ' ', tweet)
    return tweet
