import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";

function SentimentChart({ sentimentData }) {
  return (
    <div className="chart-section">
      <h2>Sentiment Distribution</h2>
      <BarChart width={500} height={300} data={sentimentData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="sentiment" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="count" fill="#82ca9d" />
      </BarChart>
    </div>
  );
}

export default SentimentChart;
