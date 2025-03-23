import requests
import json
import faiss
import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import sys
import io
import streamlit as st




# ✅ Initialize Models
   

class Patentanalyzer:
    def __init__(self,googleapikey,serpapikey):
        self.llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=googleapikey,  # ✅ Use google_api_key instead of gemini_api_key
        temperature=0,
        max_tokens=None,
        timeout=None
        )
        self.model= SentenceTransformer("./embeddings_model")
        self.wordcloud=None
        self.SERPAPI_KEY=serpapikey
# ✅ Step 1: Fetch Patents from Google Patents API
    def fetch_patents(self,query):
        API_URL = "https://serpapi.com/search?engine=google_patents"
        three_months_ago = (datetime.today() - timedelta(days=90)).strftime("%Y%m%d")
        params = {"q": query, "after": three_months_ago, "api_key": self.SERPAPI_KEY}

        response = requests.get(API_URL, params=params)
        data = response.json()

        extracted_results = [
            {
                "Title": patent.get("title", "N/A"),
                "Abstract": patent.get("snippet", "N/A"),
                "Assignee": patent.get("assignee", "N/A"),
                "Keywords": patent.get("keywords", []),
                "Publication Date": patent.get("publication_date", "N/A"),
                "Link": patent.get("link", "N/A"),
            }
            for patent in data.get("organic_results", [])
        ]

        json_file_path = "filtered_patents.json"
        with open(json_file_path, "w") as f:
            json.dump(extracted_results, f, indent=4)

        return json_file_path

    # ✅ Step 2: Store Patents in FAISS
    def store_in_faiss(self):
        with open("filtered_patents.json", "r") as f:
            patents = json.load(f)

        abstracts = [patent["Abstract"] for patent in patents]
        embeddings = np.array(self.model.encode(abstracts)).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        patent_metadata = {i: patents[i] for i in range(len(patents))}

        faiss.write_index(index, "patent_index.faiss")
        with open("patent_metadata.json", "w") as f:
            json.dump(patent_metadata, f, indent=4)

        return "FAISS Index and Metadata stored successfully!"

    # ✅ Step 3: Retrieve Top Patents
    def retrieve_top_patents(self,query, top_k=5):
        index = faiss.read_index("patent_index.faiss")
        with open("patent_metadata.json", "r") as f:
            patent_metadata = json.load(f)

        query_embedding = self.model.encode([query]).astype("float32")
        D, I = index.search(query_embedding, top_k)
        return [patent_metadata[str(i)] for i in I[0]]

    # ✅ Step 4: Analyze Trends
    def analyze_trends(self,query):
        top_patents = self.retrieve_top_patents(query)
        text_data = " ".join(
            [patent["Title"] + " " + patent["Abstract"] for patent in top_patents]
        )

        # Step 3: Generate & Plot Word Cloud
        self.wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        prompt = f"""
        Based on the following top 5 patents related to {query}, analyze emerging trends and provide R&D recommendations.

        Patents:
        {json.dumps(top_patents, indent=4)}

        Required Output:
        - Top 5 emerging innovations under this topic (last 3 months)
        - AI-generated trends summary (brief insights)
        - Innovation recommendations for R&D teams
        """
        response = self.llm.invoke(prompt)
        return response.content

    # ✅ Step 5: Initialize Agents
    def initialize_agents(self):
        patent_tool = Tool(
            name="PatentFetcher",
            func=self.fetch_patents,
            description="Fetches latest patents (last 3 months), extracts relevant fields, and saves results to a JSON file."
        )
        trend_tool = Tool(
            name="TrendAnalyzer",
            func=self.analyze_trends,
            description="Analyzes top 5 patents, extracts trends, and generates R&D recommendations."
        )
        agent_fetcher = initialize_agent(tools=[patent_tool], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, llm=self.llm)
        agent_trend_analyzer = initialize_agent(tools=[trend_tool], agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, llm=self.llm)
        return agent_fetcher, agent_trend_analyzer

    # ✅ Step 6: Run Full RAG Pipeline
    def run_rag_pipeline(self, industry, topic):
        agent_fetcher, agent_trend_analyzer = self.initialize_agents()

        output_buffer = io.StringIO()

        with st.spinner(f"🔍 Fetching patents for {topic} in {industry}..."):
            sys.stdout = output_buffer  # Capture logs
            agent_fetcher.run({"input": f"Fetch patents related to {topic} in {industry} industry."})
            sys.stdout = sys.__stdout__  # Reset stdout
        st.success("✅ Patents fetched successfully!")

        with st.spinner("📥 Storing patents in FAISS..."):
            sys.stdout = output_buffer
            self.store_in_faiss()
            sys.stdout = sys.__stdout__
        st.success("✅ FAISS storage completed!")

        trend_query = f"What all new technologies/innovations to focus on, related to {topic} in {industry}?"

        with st.spinner("🔍 Analyzing trends..."):
            sys.stdout = output_buffer
            trend_report = agent_trend_analyzer.run({"input": trend_query})
            sys.stdout = sys.__stdout__
        st.success("✅ Trend analysis completed!")

        # Display captured logs
        terminal_output = output_buffer.getvalue()
        st.write("### Terminal Output")
        st.write("Logs", terminal_output)

        st.write(trend_report)
        return trend_report,self.wordcloud

# ✅ Streamlit UI
st.title("Patent Trend Analyzer")
serpapi_key = st.text_input("Enter SERPAPI Key:", type="password")
gemini_api_key = st.text_input("Enter Gemini API Key:", type="password")
industry = st.text_input("Enter Industry:")
topic = st.text_input("Enter Topic:")


output = st.empty()
if st.button("Generate Insights"):
    if not serpapi_key or not gemini_api_key or not industry or not topic:
        st.error("Please provide all required inputs!")
    else:
        with st.spinner("📥 Patent analysis agent started..."):
            patentanalyzer=Patentanalyzer(googleapikey=gemini_api_key,serpapikey=serpapi_key)

            report,wordcloud=patentanalyzer.run_rag_pipeline(industry,topic)
        st.success("✅ patent analysis completed")
        st.title("Final Insights")

        # Convert WordCloud to an image and display it
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Display the generated report
        st.write(report)


            
