import React, { useState, useEffect } from "react";
import axios from "axios";
import NetworkGraph from "./components/NetworkGraph";
import TweetList from "./components/TweetList";
import SentimentChart from "./components/SentimentChart";
import "./App.css";

const BACKEND_API_URL = "http://127.0.0.1:5000";

function App() {
  const [tweets, setTweets] = useState([]);
  const [keyword, setKeyword] = useState("");
  const [sentimentData, setSentimentData] = useState([]);

  useEffect(() => {
    const fetchTweets = async () => {
      try {
        const response = await axios.get(`${BACKEND_API_URL}/tweets`);
        setTweets(response.data);
        processSentimentData(response.data);
      } catch (error) {
        console.error("Error fetching tweets:", error);
      }
    };
    fetchTweets();
  }, []);

  // Process sentiment data for graphing
  const processSentimentData = (tweets) => {
    const sentimentCounts = { positive: 0, neutral: 0, negative: 0 };
    tweets.forEach((tweet) => sentimentCounts[tweet.sentiment]++);
    setSentimentData(Object.keys(sentimentCounts).map((key) => ({ sentiment: key, count: sentimentCounts[key] })));
  };

  return (
    <div className="App">
      <h1>Social Media Sentiment Dashboard</h1>

      <div className="search-section">
        <input
          type="text"
          placeholder="Filter by keyword"
          value={keyword}
          onChange={(e) => setKeyword(e.target.value)}
        />
      </div>

      <SentimentChart sentimentData={sentimentData} />
      <NetworkGraph />
      <TweetList tweets={tweets} keyword={keyword} />
    </div>
  );
}

export default App;
