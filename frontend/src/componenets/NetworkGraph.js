import React, { useEffect, useState } from "react";
import axios from "axios";
import { ForceGraph2D } from "react-force-graph";

const BACKEND_API_URL = "http://127.0.0.1:5000";

function NetworkGraph() {
  const [networkData, setNetworkData] = useState({ nodes: [], links: [] });

  useEffect(() => {
    const fetchNetworkData = async () => {
      try {
        const response = await axios.get(`${BACKEND_API_URL}/network`);
        const graphData = response.data;

        const nodes = graphData.map(user => ({
          id: user.user_name,
          centrality: user.centrality,
          retweet_count: user.retweet_count,
        }));

        const links = [];
        graphData.forEach(user => {
          user.connections.forEach(connection => {
            links.push({
              source: user.user_name,
              target: connection,
            });
          });
        });

        setNetworkData({ nodes, links });
      } catch (error) {
        console.error("Error fetching network data:", error);
      }
    };
    fetchNetworkData();
  }, []);

  return (
    <div style={{ height: "500px" }}>
      <h2>User Network Graph</h2>
      <ForceGraph2D
        graphData={networkData}
        nodeLabel="id"
        nodeAutoColorBy="centrality"
        linkDirectionalParticles={2}
        linkDirectionalParticleSpeed={d => d.centrality * 0.001}
      />
    </div>
  );
}

export default NetworkGraph;
