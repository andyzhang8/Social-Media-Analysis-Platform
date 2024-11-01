import React from "react";

function TweetList({ tweets, keyword }) {
  return (
    <div className="tweet-list">
      <h2>Recent Tweets</h2>
      {tweets
        .filter((tweet) => tweet.keyword.includes(keyword))
        .map((tweet) => (
          <div key={tweet._id} className="tweet">
            <p><strong>{tweet.user_name}</strong> ({tweet.user_location}): {tweet.text}</p>
            <p>Sentiment: {tweet.sentiment}</p>
          </div>
        ))}
    </div>
  );
}

export default TweetList;
