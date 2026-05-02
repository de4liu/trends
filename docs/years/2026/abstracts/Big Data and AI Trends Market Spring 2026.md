  
**Big Data and AI Trends Market, Spring 2026**

***From Data to Action: Agentic AI Systems with Big Data***

April 24, 2026

![][image1]

[Team 1: Clinical Note Intelligence: An Agentic Hybrid Retrieval Framework Combining Structured Search and Retrieval-Augmented Generation](#team-1:-clinical-note-intelligence:-an-agentic-hybrid-retrieval-framework-combining-structured-search-and-retrieval-augmented-generation)

[Team 2: Karen – AI Complaint Assistant](#team-2:-karen-–-ai-complaint-assistant)

[Team 3: PotholeVision–Automating Pothole Detection and GeoMapping](#team-3:-potholevision–automating-pothole-detection-and-geomapping)

[Team 4: From Clicks to Actions: Spark-Powered Funnel Analysis with LLM-Driven Recommendations](#team-4:-from-clicks-to-actions:-spark-powered-funnel-analysis-with-llm-driven-recommendations)

[Team 5: PersonaPath: Personalized Travel & Dining Recommendation Engine (Behavioral Profiling)](#team-5:-personapath:-personalized-travel-&-dining-recommendation-engine-\(behavioral-profiling\))

[Team 6: Data Quality Remediation Assistant: AI-Driven Anomaly Detection & ETL Fix Generation at Scale](#team-6:-data-quality-remediation-assistant:-ai-driven-anomaly-detection-&-etl-fix-generation-at-scale)

[Team 7: Demand Sense: An AI-Backed Driver Nudge System for Demand-Aware Repositioning](#team-7:-demand-sense:-an-ai-backed-driver-nudge-system-for-demand-aware-repositioning)

[Team 8: NFL Contract Prediction and Evaluation with LLM-Based Recommendations](#team-8:-nfl-contract-prediction-and-evaluation-with-llm-based-recommendations)

[Team 9: InsideInsight: Agentic AI for Airbnb Pricing Strategy and Performance Optimization](#team-9:-insideinsight:-agentic-ai-for-airbnb-pricing-strategy-and-performance-optimization)

[Team 10: An AI Copilot for Detecting Delayed Market Reactions to Corporate Disclosures](#team-10:-an-ai-copilot-for-detecting-delayed-market-reactions-to-corporate-disclosures)

[Team 11: Detect Hidden Drug Safety Risks Faster with AI — FDA FAERS Analytics](#team-11:-detect-hidden-drug-safety-risks-faster-with-ai-—-fda-faers-analytics)

[Team 12: TheaterIQ: AI-Driven Scheduling and Promotional Intelligence for Movie Theater Operations](#team-12:-theateriq:-ai-driven-scheduling-and-promotional-intelligence-for-movie-theater-operations)

### **Team 1: Clinical Note Intelligence: An Agentic Hybrid Retrieval Framework Combining Structured Search and Retrieval-Augmented Generation** {#team-1:-clinical-note-intelligence:-an-agentic-hybrid-retrieval-framework-combining-structured-search-and-retrieval-augmented-generation}

**Members**: Ethan Armstrong, Ankit (Ziqi) Cao, Ko Jung Hsu, Cole Johnson, Mashhood Khan, Wenyu Zhong

**Abstract:**  
This project focuses on enabling scalable, data-driven clinical insights through an AI-powered chatbot that leverage Retrieval Augmented Generation (RAG) to analyze large-scale healthcare datasets. As hospitals and research institutions accumulate vast amounts of structured and unstructured patient data, extracting meaningful insights efficiently is a growing problem. By integrating large language models with a vector-based retrieval system, this solution allows healthcare professionals to query aggregate patient data using natural language while ensuring responses are grounded in real clinical data.  
The system combines embedding-based retrieval for relevant cohort and statistical data with generative AI for intuitive, context-aware explanations. Data processing and aggregation are performed using scalable tools such as Apache Spark or Pandas, while the RAG pipeline is implemented using frameworks like LangChain or LlamaIndex. Outputs are delivered through an interactive chatbot interface, enabling rapid exploration of trends such as readmission patterns, treatment outcomes, and population level health metrics.  
By improving accessibility to complex healthcare data, this project contributes to more informed clinical research and operational decision-making. The resulting insights can be utilized by a wide range of stakeholders, including physicians, medical researchers, and hospital administrators, all working toward improving patient outcomes, optimizing resource allocation, and advancing evidence-based healthcare practices.

**Use Cases:**   
**1\) Population level cross-patient pattern discovery (population-level insight)**

**2\) Patient level note summarization**

**DataSet sources:** https://www.physionet.org/content/mimic-iv-note/2.2/

**Tools/Technology used:** 

* Python (Pandas) (Data ingestion, initial processing)  
* Cloud storage (AWS S3)  
* SQL Database  
* Apache Spark  
* RAG pipeline framework   
* Query interpretation agents for AI (translates query into retrieval steps)

**Generative/Agentic AI used (if any):**  

* GPT-4 (Embedding generation, LLM model)  
* LangGraph

**GitHub URL**:  
[https://github.com/Eunggseo/big\_data\_team1](https://github.com/Eunggseo/big_data_team1)  
 

### 

### **Team 2: Karen – AI Complaint Assistant** {#team-2:-karen-–-ai-complaint-assistant}

**Members**: Mohameddeq Ali, Cora Goodwin, Midori Neaton, Raja Sori, Xupei Ye, Kyle Zhu

**Abstract:** 

**Problem \-** Financial institutions receive a massive volume of complaints containing vital signals about operational risks. Employees lack an efficient way to **identify and prioritize** the most urgent issues. This makes it difficult to quickly detect critical problems or track whether high-risk consumer complaints are being effectively addressed.

**Solution \-** We developed **Karen**, an AI-powered assistant designed to transform complaint data into actionable intelligence. Our approach addresses two core challenges:

1. **Data Structuring & Prioritization:** Using **BERTopic** for topic modeling and a custom **Priority Score**—calculated via volume, growth, recency, and "topic danger"—we convert unstructured narratives into ranked, analyzable categories.  
2. **Accessible Insights:** An AI interface allows users to query metrics and generate visualizations through natural language.

**Business Value \-**  By automating the classification and ranking of consumer issues, Karen enables faster, more informed decision-making. This reduces the time required to identify high-risk trends, ensuring financial institutions can address consumer problems with greater precision and speed.

**DataSet sources:** Public CFPB consumer complaint dataset   
[https://catalog.data.gov/dataset/consumer-complaint-database](https://catalog.data.gov/dataset/consumer-complaint-database) 

**Tools/Technology used:**

* **Language Models:** GPT-4o, Faster Whisper  
* **AI Framework:** LangChain  
* **Data Processing:** DuckDB, Parquet, Pandas  
* **NLP/Topics:** BERT, LDA  
* **Front-End:** Streamlit, Plotly  
* **Audio:** OpenAI TTS, Mutagen  
* **Backend:** Python 3.11, WSL2

**Generative/Agentic AI used (if any):**

* **Generative AI:** GPT-4o, OpenAI TTS, and Faster Whisper.  
* **Agentic AI:** LangChain

**GitHub URL**: [https://github.com/ali00418-ship-it/msba-team2-trendsproject](https://github.com/ali00418-ship-it/msba-team2-trendsproject)  
 

### **Team 3: PotholeVision–Automating Pothole Detection and GeoMapping** {#team-3:-potholevision–automating-pothole-detection-and-geomapping}

**Members**: Chunfang Wang, James Pashek, Joseph Sheehan, Madhu Damani, Moses Effah Akoto, Tao Fang

**Abstract:**

The road inspection for cities is highly manual, slow, and expensive. Municipalities desperately need a scalable way to identify road damage faster and prioritize repairs. To solve this, we developed **PotholeVision**, an automated pothole detection and geo-mapping solution. This big data pipeline ingests video and image data in batches, extracts frames, and evaluates them using a ResNet18-based Convolutional Neural Network. It drives direct business value by pushing positive pothole predictions into a dynamic ArcGIS dashboard. Through this dashboard, Public Works planners can view pothole counts, exact locations, and street traffic volumes to prioritize repairs on the busiest roads. Ultimately, PotholeVision drives faster repairs, better allocation of limited maintenance resources, and a significant reduction in vehicle accidents.

**DataSet sources:** [Primary dataset](https://github.com/sekilab/RoadDamageDetector). This is a collection of Pothole data from seven countries              covering varying road texture filmed by different equipment. [Independent benchmark set](https://zenodo.org/records/13334878)  Validates true cross-dataset generalization. It has labels. This will be primarily what our model will be trained and tested on. Downloaded youtube street. We will scrape some sample videos for more data and apply the model on stills taken from videos. [Driving In Downtown Minneapolis, Minnesota | City Drive, City Sounds](https://www.youtube.com/watch?v=HRPQSIfHwEo)

**Tools/Technology used:** Databricks, Pipeline, PySpark, PyTorch, Python CNN, ArcGIS, OpenCV (for video data)

**Generative/Agentic AI used (if any):** Gemini, Claude

**GitHub URL**: [https://github.com/BDAteam3/Pothole\_Project](https://github.com/BDAteam3/Pothole_Project)

### **Team 4: From Clicks to Actions: Spark-Powered Funnel Analysis with LLM-Driven Recommendations** {#team-4:-from-clicks-to-actions:-spark-powered-funnel-analysis-with-llm-driven-recommendations}

**Members**: Shang Chi Hsu, Xiang Li, Ashwini Manokar, Meenakshi Narendra, Isabel O'Grady

**Abstract :**

This project focuses on improving conversion performance in e-commerce platforms by analyzing large-scale clickstream data. Using a scalable Spark-based pipeline, we analyze user journeys at the session level (e.g., view → cart → purchase) to quantify conversion rates and identify key drop-off points across the funnel.

Beyond basic funnel analysis, we examine behavioral patterns to detect friction points associated with failed conversions, such as repeated product views, cart abandonment, and remove-from-cart behavior. We further segment products by traffic volume, conversion efficiency, brand, category, and price range to uncover systematic performance gaps.

To support decision-making, we introduce a prioritization framework that ranks high-impact products and behavioral patterns based on their potential effect on conversion. Additionally, a lightweight LLM component may be incorporated to translate analytical findings into structured, actionable recommendations for brands or categories, helping guide product and UX improvements.

**DataSet sources:**   
[E-commerce Behavior Dataset (Multi-category Store)](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

**Tools/Technology used:**

* Apache Spark / PySpark  
* Python  
* Streamlit  
* Databricks

**Generative/Agentic AI used (if any):**

OpenAI GPT (or equivalent)

   
**GitHub URL**: [https://github.com/SomegHsu/Clicks-To-Actions-Funnel-Analysis](https://github.com/SomegHsu/Clicks-To-Actions-Funnel-Analysis)

### **Team 5: PersonaPath: Personalized Travel & Dining Recommendation Engine (Behavioral Profiling)** {#team-5:-personapath:-personalized-travel-&-dining-recommendation-engine-(behavioral-profiling)}

**Members**: Esther Baumgartner, Hsin Kuei Chang, Saloni Jain, Fu Lee, Dhairya Lunia

**Proposal 1: Traveler Intelligence Platform**

**Abstract:**

This project develops a personalized travel and dining recommendation engine built on the Yelp Open Dataset. While platforms like Yelp already leverage ratings, categories, and basic personalization, they are still limited in how well they capture user intent. In particular, they often fall short in understanding the context behind a user’s preferences, such as occasion, mood, or dining purpose. As a result, recommendations tend to rely on relatively shallow signals and do not fully reflect how and why users choose certain places.

To address this gap, the platform is designed as an analytical prototype that demonstrates how large-scale review data can be used to construct a more holistic, behavior-driven view of users. By leveraging review text to extract latent preferences and map them to real-world contexts, the system enables more precise, context-aware, and user-centric recommendations.

The pipeline consists of four main steps. First, topic modeling is applied to review text to extract latent themes such as ambiance, value, service speed, and cuisine authenticity. Second, these topics are translated into contextual tags that reflect real-world intent and occasions. For users, this includes tags like solo traveler, friend gathering, or family reunion; for businesses, tags such as cozy night bar with great food or sunny brunch spot with a view. Third, these tags are used to construct dual-side profiles, where users are grouped into behavioral personas and businesses into experience-based clusters, both expressed in human-readable, context-driven language. Finally, an LLM-powered RAG pipeline combines the user profile with a natural language query to retrieve and rank the most relevant businesses, with recommendations supported by actual reviewer insights.

For existing users, the system maps their user ID to historical review data to infer personas and contextual preferences. For new users, a short onboarding prompt assigns them to the closest user cluster. Since businesses are already grouped by contextual tags, the system can immediately generate relevant recommendations even without prior interaction history. As new reviews are ingested, both user and business profiles can be continuously updated, and chatbot interactions are used as feedback signals to further improve recommendation quality.

**Use Cases:**

**Intent-based place discovery**

Chat: “I want somewhere cozy for a solo dinner with good vegetarian options”

Retrieval: Businesses tagged with intimate dining, vegetarian-friendly, and solo-appropriate are matched with the user’s persona

Output: Ranked recommendations with explanations grounded in reviewer language

**Occasion-based group recommendations**

Chat: “Looking for a place for a family reunion lunch, something relaxed with lots of space”

Retrieval: Businesses tagged as family-friendly, spacious, and relaxed are matched to the query context

Output: A curated shortlist with reviewer-backed reasoning

**Dataset Sources:**

Yelp Open Dataset: https://www.yelp.com/dataset/download

Includes \~6.9M reviews, \~2.2M users, \~150K businesses

Format: JSON (reviews, users, businesses, check-ins, tips)

Cities: Atlanta, Austin, Boston, Boulder, Columbus, Orlando, Portland, Vancouver

**Tools/Technology Used:**

Python (PySpark, Pandas) — data ingestion and processing

Databricks / Delta Lake — scalable storage and pipeline management

LDA — topic modeling on review text

**Generative/Agentic AI Used:**

OpenAI GPT-4o — natural language understanding and response generation

Streamlit — interactive chatbot interface

Github URL: [https://github.com/DhairyaLunia/Team5\_Big\_data.git](https://github.com/DhairyaLunia/Team5_Big_data.git) 

### **Team 6: Data Quality Remediation Assistant: AI-Driven Anomaly Detection & ETL Fix Generation at Scale** {#team-6:-data-quality-remediation-assistant:-ai-driven-anomaly-detection-&-etl-fix-generation-at-scale}

**Members:** Sean Cabaniss, Yung Hsuan Hsieh, Ching-Fen Hung, Yonghui Kim, Omkar Thombare

**Abstract:**  
Poor data quality costs organizations an estimated $12.9 million per year on average, yet most data pipelines still rely on manual inspection and ad hoc fixes. This project builds a scalable Data Quality Remediation Assistant that automatically detects schema drifts, null anomalies, format inconsistencies, and statistical outliers in large-scale datasets using Apache Spark, then leverages a large language model (LLM) to propose contextualized ETL remediation strategies with confidence scores and plain-English rationale. A human analyst reviews and approves all suggestions through an interactive Streamlit dashboard before any fix is executed, which preserves human oversight while dramatically accelerating the remediation workflow. The system stores all detected issues, LLM suggestions, and remediation decisions in Snowflake for full audit trail and trend analysis. 

**Updated Abstract:**  
Poor data quality costs organizations an estimated $12.9 million per year on average, yet most data pipelines still rely on manual inspection and ad hoc fixes. This project builds a scalable Data Quality Remediation Assistant that automatically detects null anomalies and statistical outliers in large-scale datasets using Apache Spark, then leverages a 2-step LLM agent pipeline to diagnose root causes, assess business impact, and generate runnable PySpark remediation code. A human analyst reviews and approves all suggestions through an interactive Streamlit dashboard before any fix is executed, preserving human oversight while dramatically accelerating the remediation workflow. The system tracks before/after data quality scores and logs all decisions for full auditability.

**DataSet sources:**   
[**Lending Club Loan Data**](https://www.kaggle.com/datasets/wordsforthewise/lending-club)**(\~2.5M rows / 2.5GB)**  
[**Chicago City Payments**](https://data.cityofchicago.org/Administration-Finance/Payments/s4vu-giwb)**(\~5M rows)**  
[**NYSE Historical Prices**](https://www.kaggle.com/datasets/dgawlik/nyse)**(\~3M rows)**

**Tools/Technology used:**  
\- Apache Spark / PySpark   
\- Databricks   
\- OpenAI GPT-4o-mini   
\- Streamlit   
\- Delta Lake

**Generative/Agentic AI used (if any):**  
Model: OpenAI GPT-4o-mini (via API)

Architecture: 2-step agent pipeline   
\- Agent 1 (Diagnosis): identifies root cause, priority score, and business impact  
\- Agent 2 (Remediation): generates runnable PySpark fix code based on Agent 1 output 

Safeguards: Human-in-the-loop design, grounded prompting, full audit trail in Delta Lake

**GitHub URL**: [https://github.com/tom666d/big\_data\_team\_6/tree/main](https://github.com/tom666d/big_data_team_6/tree/main)

### **Team 7: Demand Sense: An AI-Backed Driver Nudge System for Demand-Aware Repositioning** {#team-7:-demand-sense:-an-ai-backed-driver-nudge-system-for-demand-aware-repositioning}

**Members**: Davey Johnson, Hengrui Li, Huiguo Liu, Mansi Malpani, Mounika Polamreddy

**Abstract:**

Ride-hailing platforms frequently experience mismatches between driver availability and rider demand, leading to increased wait times and inefficient driver utilization. While existing dispatch systems provide recommendations, they operate behind the scenes and do not explain the data or reasoning behind them, limiting transparency and driver trust. Our project develops an AI-backed driver nudge system, using 9.4 million NYC taxi trip records to identify high-demand zones across different time periods and provide clear, data-backed insights to drivers.

We use large-scale trip data from the NYC Taxi and Limousine Commission, along with geographic zone mappings, to capture detailed patterns in rider demand. Using Databricks and Apache Spark, we perform data engineering and aggregation, then compute normalized demand scores that compare each zone’s activity to the citywide average. A lightweight ranking layer identifies the highest-opportunity zones within each time window while filtering out low-confidence signals to ensure recommendations remain stable and reliable.

To make results more interpretable, we connect the system to a Large Language Model (LLM) such as OpenAI. The LLM translates structured demand signals into short, plain-language driver nudges that explain where demand is strong and why it may be worthwhile to reposition. Reliability is maintained through confidence filtering, prompt constraints, and automated validation checks that ensure messages are grounded in the underlying data, avoid unsupported claims, and remain concise. Performance is evaluated using a temporal holdout approach, where recommended zones are compared against actual demand patterns in held-out data, achieving results that outperform random selection. 

**DataSet sources:** 

NYC Taxi and Limousine Commission (TLC) Trip Record Data (Jan-Mar 2023\) – [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

NYC Taxi Zone Shapefiles – [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

**Tools/Technology used:**

* *Databricks (for large-scale data processing and pipeline orchestration)*  
* *Apache Spark / PySpark (for data cleaning, aggregation, and demand scoring)*  
* *Delta Lake (for intermediate and final data storage)*  
* *Python (data processing and ranking logic)*  
* *OpenAI API (for generating natural-language driver messages)*  
* *Matplotlib (for visualization and evaluation)*

**Generative/Agentic AI used (if any):**

We use the OpenAI API as a communication layer to generate explainable driver nudges from structured demand data. The system first identifies high-demand zones using a transparent ranking process, then passes these results into the LLM to produce short, plain-language messages that explain where demand is strong and why. The model does not make decisions or predictions; it translates analytical outputs into interpretable guidance for drivers. Reliability is maintained through confidence filtering, structured prompts, and automated checks that ensure outputs remain grounded in the data, avoid unsupported claims, and meet formatting constraints. Message quality is evaluated by verifying alignment with observed demand patterns and ensuring consistency across all generated outputs.

   
**GitHub URL**: [https://github.com/DRJohnson21/team7-driver-nudge-system](https://github.com/DRJohnson21/team7-driver-nudge-system) 

### **Team 8: NFL Contract Prediction and Evaluation with LLM-Based Recommendations** {#team-8:-nfl-contract-prediction-and-evaluation-with-llm-based-recommendations}

### 

**Members**: Adam Getzkin, Mallika Kommera, Jay Pederson, Ariel Zhan, Zhen Zhang

**Abstract:**

One of the best ways to find success in the NFL is getting strong production from players on undervalued contracts (that’s why rookie contracts are so valuable\!). Our project aims to predict NFL contracts and guaranteed values by analyzing player performance statistics along with historical contract data, helping teams, media, and fans identify undervalued or overvalued players.

We will use the nflreadpy Python package to access play-by-play and contract data, focusing on large-scale datasets to capture detailed player contributions. Using Databricks, we will perform necessary data engineering, then build a predictive model with Python libraries like xgboost to estimate contract values.

To make the results more interpretable, we will connect the model to a large language model (LLM) such as OpenAI or LLaMA(free). The LLM will function as a recommendation agent — when a user submits a query such as "find undervalued running backs under $6M", the agent rewrites the query into structured filters, retrieves the relevant player pool, runs predictions on each candidate, compares predicted value against actual market figures, and returns a ranked recommendation list with plain-language justifications. Reliability will be maintained through rule-based validation, performance monitoring, bias checks, and a fallback mechanism that displays raw predictions if the LLM fails. This approach combines predictive analytics with generative AI to provide a practical, transparent tool for evaluating NFL contracts

**DataSet sources:** nflreadpy package in Python ([https://github.com/nflverse/nflreadpy](https://github.com/nflverse/nflreadpy)). Play by play data and contracts data (using other datasets in package as needed)

**Tools/Technology used:**

* Databricks (for data engineering and model building)  
* Python (nflreadpy, pandas, xgboost (or other ML package), etc.)  
* OpenAI API or Llama

**Generative/Agentic AI used (if any):**  
We will use OpenAI API or Llama depending on cost. The LLM functions as a recommendation agent rather than a passive explanation tool. When a user submits a natural language query such as "find undervalued running backs under $6M", the agent rewrites the query into structured filters, retrieves the relevant player pool, calls the XGBoost model to predict contract value for each candidate, compares predicted value against actual market figures, and returns a ranked recommendation list with plain-language justifications. Rule-based checks ensure LLM outputs are consistent with model predictions. Agent recommendation quality will be assessed by evaluating whether recommended players' actual contracts fall within the predicted value range. If the LLM fails or is inconsistent, the system will fall back to displaying the raw model predictions and key statistics.  
**GitHub URL**: [https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project](https://github.com/zhan9921-afk/Big-Data-and-AI-Trend-Project)

### **Team 9: InsideInsight: Agentic AI for Airbnb Pricing Strategy and Performance Optimization**  {#team-9:-insideinsight:-agentic-ai-for-airbnb-pricing-strategy-and-performance-optimization}

**Members**: Bhavisha Chafekar, Jyothirmai Sri Peesapati, Phoenix Ferrari, Stephen Weiler, Tzu-Yu Chen

**Abstract:**  
This project proposes an end-to-end Big Data and Agentic AI system that transforms large-scale Airbnb data into actionable insights for hosts and property managers. Using the Inside Airbnb dataset, which contains millions of listings, calendar records, and reviews, we will build a scalable data pipeline to process, clean, and analyze pricing, availability, and customer feedback at scale. By applying data analysis and natural language processing techniques, we will identify key factors that influence occupancy, pricing, and guest satisfaction across different neighborhoods.

Additionally, we will implement an AI-powered system that converts analytical outputs into clear, actionable recommendations. The final system will help hosts understand their competitive position and make data-driven decisions to improve pricing strategies, listing quality, and overall performance.

**DataSet sources:**  
Inside Airbnb Dataset [https://insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data)  
Contains listings data, calendar data, and reviews data across multiple cities  
Includes structured data (pricing, amenities, availability) and unstructured text data (guest reviews)

**Tools/Technology used:**

* Apache Spark / Databricks  
  Distributed data processing for large-scale datasets  
* Python (PySpark, Pandas)  
  Data cleaning, preprocessing, and feature engineering  
* Storage (Delta Lake / Hive Metastore)  
  Medallion architecture (Bronze → Silver → Gold) for structured data pipelines  
* Natural Language Processing (NLP) (PySpark, Hugging Face, BERTopic)  
  Text preprocessing and sentiment analysis on review data using transformer models  
* Generative/Agentic AI used (if any):  
  We will implement a multi-agent system that translates processed data into actionable recommendations for hosts.

**Workflow:**

* Process and analyze Airbnb data using Spark and NLP  
* Generate structured outputs such as occupancy metrics, sentiment scores, and pricing insights  
* Feed structured outputs into an LLM  
* Generate actionable recommendations and Action Plans

**Model / Tools:**

* OpenAI API or similar LLM for recommendation generation  
* LangGraph or CrewAI for agent orchestration

**Validation**:

* Ground AI outputs in structured results from the processed data  
* Validate pricing model using RMSE and MAE  
* Perform manual spot-checks on sentiment outputs and AI-generated recommendations  
* Ensure outputs are based on retrieved statistics rather than unsupported claims

**Safeguards**:

* Limit AI outputs to retrieved data to reduce hallucinations  
* Apply confidence thresholds and flag low-confidence outputs  
* Avoid use of personally identifiable data  
* Monitor API usage and apply cost controls

GitHub URL:

[https://github.com/stephweil208/big\_data\_team\_9](https://github.com/stephweil208/big_data_team_9)

### **Team 10: An AI Copilot for Detecting Delayed Market Reactions to Corporate Disclosures**  {#team-10:-an-ai-copilot-for-detecting-delayed-market-reactions-to-corporate-disclosures}

**Members**: Kristina Dennise Paraiso, Evelyn Lai, Zhichen Yang, Parul Chaudhary, Shivanshu Dagur

**Abstract:**  
Companies are required to publicly disclose important business events – leadership changes, earnings results, major contracts, and more. While most announcements are quickly reflected in stock prices, some contain information the market initially overlooks, only reacting days or weeks later. This project builds an AI-powered Disclosure Intelligence System that ingests filings from various S\&P 500 companies, scores their importance using a grounded LLM, analyzes post-filing market reactions, and generates a prioritized analyst watchlist.

We define an "underinterpreted disclosure" as a filing with high LLM-assessed importance but weak immediate market reaction, followed by significant delayed price adjustment. These are ranked using a composite Missed Opportunity Score (MOS) and surfaced through an interactive dashboard — flagging what the market missed before it catches up.

**Business Value:**

* Saves analyst time — Replaces hours of manual reading with a ranked, prioritized watchlist  
* Reduces missed opportunities — Systematically flags slow-reaction filings before the broader market catches on  
* Works in real time — New announcements can be scored from text alone, within minutes of release  
* Keeps humans in control — Every flagged filing includes a plain-language explanation tied to the source text, so analysts understand *why* it was flagged and make the final call

**DataSet sources:** 

* SEC EDGAR Filings (10-K, 10-Q, 8-K) [https://www.sec.gov/submit-filings](https://www.sec.gov/submit-filings)  
* Stock Market Data [https://finance.yahoo.com/](https://finance.yahoo.com/)  
  Used to compute post-disclosure returns and reaction patterns

**Tools/Technology used:**

* Python – data processing and modeling  
* Pandas / NumPy – data analysis  
* SQL – data querying and integration  
* NLP Libraries (e.g., HuggingFace, spaCy) – text analysis  
* LLMs \+ RAG – explanation and summarization  
* Streamlit – Dashboard Visualization  
* Machine Learning  
* Vector DB

**Generative/Agentic AI used (if any):**

* LLM used as a **copilot for financial analysis**, not as a decision-maker  
* RAG (Retrieval-Augmented Generation) to ground explanations in:  
  * original filings  
  * similar historical disclosures  
* Generates:  
  * structured summaries  
  * explanation of importance  
  * reasoning for delayed market reaction

**Safeguards:**

* Hallucination Control: LLM outputs are grounded in retrieved filing text; no unsupported claims allowed  
* Traceability: Each output includes references to original filings and historical examples  
* Confidence Scoring: Statistical metrics (e.g., abnormal return patterns, clustering confidence) are provided  
* Human-in-the-Loop: The system provides decision support only; final interpretation is left to analysts

**GitHub URL**: [https://github.com/ShivanshuDagur/not-yet-priced-in.git](https://github.com/ShivanshuDagur/not-yet-priced-in.git)

### **Team 11: Detect Hidden Drug Safety Risks Faster with AI — FDA FAERS Analytics** {#team-11:-detect-hidden-drug-safety-risks-faster-with-ai-—-fda-faers-analytics}

**Members**: Amogha Yalgi, Austin Ganje, Hannah Huang, Hayden Herstrom, Rachel Le 

**Abstract:**

Healthcare providers, pharma companies, and regulators need to identify emerging patterns in adverse drug events, including unexpected drug-event relationships and side-effect trends. The FDA Adverse Event Reporting System (FAERS) contains millions of voluntary reports, but the data is messy, inconsistent, and partially unstructured, making large-scale analysis a significant challenge. In total, approximately 31 million reports exist, with roughly 1 million new reports filed every year (\~250,000 per quarter).

**Business Value: An analyst-facing application that surfaces high-risk drug-event patterns** from millions of messy adverse event reports **in seconds**. Value is delivered across three key audiences:

* **Pharma & Biotech**: Competitive safety intelligence: monitor competitor adverse event trends in real time, improve risk management decisions, and proactively address adverse trends  
* **Analysts & Regulators**: Faster public health risk identification: rapidly surface emerging risks, dramatically reduce manual review effort, and leverage AI-assisted signal summarization  
* **Healthcare Systems**: Adverse event trend awareness: better awareness of adverse event patterns, support clinical decision-making, and accessible to non-clinical staff

Core capabilities include:

* **Smart Search**: Semantic mapping for non-clinically trained users (searching "headache" automatically returns results for cephalalgia, migraine, and related terms)  
* **Risk Detection**: Identify unexpected drug-event relationships and surface emerging safety signals  
* **Signal Summarization**: AI-generated plain-language summaries of adverse event clusters for non-clinical users  
* **Interactive UI**: Drill into drugs, events, and trends — filter and explore in real time

**Pipeline:**   
Ingest quarterly FAERS files → Clean & deduplicate in Spark → Standardize drug & reaction terms → Identify key drugs, manufacturers & reactions → Surface results in a Streamlit dashboard

**DataSet sources:** FDA FAERS public quarterly data extracts (available at [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html))

Each quarter contains reports across multiple relational files (demographics, drugs, reactions, outcomes, therapy dates). Total dataset spans 2004–present and is roughly 15GB.

**Tools/Technology used:**

* **Databricks & Spark**: Data cleaning, deduplication, and aggregation at scale; processes millions of FAERS records without export; quarterly FAERS ingestion in minutes, not days; auto-scaling clusters with zero tuning  
* **Python**: Analysis and visualization  
* **LLM**: Text normalization and summarization layer. Because FAERS is updated quarterly, streaming is not needed, but significant compute is required  
* **Streamlit**: Low-code presentation layer (UI); requires no engineering to operate; drill-down by drug, event, and trend with reviewer-facing explanations for flagged cases

**Generative/Agentic AI used:**

Use an LLM to:

* Normalize messy free-text drug names to standardized identifiers  
* Summarize clusters of related adverse events into readable safety signals  
* Generate reviewer-facing explanations of why a case or signal was flagged

**GitHub URL**: [https://github.com/Hayden1629/bigdataproject](https://github.com/Hayden1629/bigdataproject) 

### **Team 12: TheaterIQ: AI-Driven Scheduling and Promotional Intelligence for Movie Theater Operations** {#team-12:-theateriq:-ai-driven-scheduling-and-promotional-intelligence-for-movie-theater-operations}

**Members:** Sam Benson Devine, Jack Halverson, Tobias Knight, Qiqi Li, Yehan Wang

**Abstract:**

Theater operators — particularly mid-size regional chains and independent multiplexes — make weekly scheduling decisions without reliable data support. The software they use handles ticketing and POS well but tells them what happened, not what to do next. The problem is sharpest mid-week, where blockbusters fill themselves and the real question is which films should run on the remaining screens, for which audiences, and when. TheaterIQ is built to answer that.

The system is not a user-level recommender. It models the fit between a film’s attributes and a theater’s demographic catchment area to score each film-slot-segment combination. Using MovieLens data, we learn which film characteristics — genre, content tags, release patterns — tend to resonate with which demographic segments. This is more like topic modeling and content-based categorization than collaborative filtering: we’re building a profile of each film’s likely audience, not predicting individual behavior. An XGBoost model then layers in theater-level context — screen count, location, simulated local demographic profile — to produce a Movie Match Score (0–100%) for each film-timeslot-segment combination.

Those scores feed an AI copilot built on LangGraph. At the start of the week it takes the full Match Score grid, accounts for screen capacity, and outputs a recommended weekly schedule with a confidence label for each slot. Mid-week, Kafka ingests simulated attendance events and Spark Structured Streaming tracks how each film paces against its prediction. When a film falls more than 20 points below its Match Score, the copilot shifts to promotions mode: it finds which segment has the largest gap between predicted affinity and observed attendance, then generates a targeted promotional brief with a specific channel, message angle, and estimated uplift range. Everything is grounded in the pipeline’s structured outputs. Managers review and approve before anything goes out.

 

**Use Cases:**

**1\) Filling the mid-week slate**

Chat: "We have 4 screens open Tuesday through Thursday — what should we run?"

Retrieval: Match Scores for all available films across the theater’s demographic segments and mid-week slots, filtered against screen availability.

Output: Ranked film-slot assignments for each open screen with confidence labels. The indie drama scores 74% for Adults 35–54 in the 7 PM slot and gets flagged for light promotional support. The kids’ title scores 81% for Families at 4 PM — high confidence, no outreach needed.

Why the system is needed: Mid-week is where independent operators lose the most money to guesswork. Blockbusters fill themselves on weekends; the mid-week slate is where a data-driven call actually moves occupancy.

 

**2\) Responding to mid-week underperformance**

Chat: "The new drama is at 38% occupancy against a predicted 71%. What do we do?"

Retrieval: Film-segment affinity scores, observed attendance by segment, historical uplift data for comparable campaigns.

Output: Target Adults 35–54 — highest affinity, lowest observed attendance. Send a loyalty email with a critic’s pick framing and run a Thursday matinee discount. Historical comparables suggest 12–18% uplift for this segment and film type.

Why the system is needed: Knowing which segment to target isn’t obvious from ticket sales — it requires the affinity signal from the film-segment model. The copilot turns that into a brief the manager can act on immediately.

 

**3\) Screen allocation for a new release**

Chat: "Where should the new franchise sequel go this weekend?"

Retrieval: Match Scores across all segments and available slots, benchmarked against historical opens for comparable releases.

Output: Large-format screen, Friday 7 PM (91%, Young Adults 18–34) and Saturday 2 PM (84%, Families). Sunday 6 PM viable as a third showing. No promotional spend flagged.

Why the system is needed: Managers over-allocate screens to blockbusters out of caution. A calibrated score cuts through the hype and frees up promotional budget for films that actually need it.

 

**Dataset sources:**

•   	**MovieLens 25M / 32M** — film attribute and demographic affinity signals. Used to model which film characteristics resonate with which segments, not to predict individual user behavior. https://grouplens.org/datasets/movielens/

•   	**TMDB / OMDb API** — genre, runtime, budget tier, critic rating joined with MovieLens on title and release year.

•   	**Simulated theater operations data** — synthetic attendance event streams published to Kafka, representing a sample of mid-sized multiplex operators across major U.S. markets. Clearly labeled as simulated throughout.

 

**Tools/Technology used:**

*Data & Storage*

•   	**Databricks** — pipeline orchestration, cluster compute, Delta Lake (Bronze/Silver/Gold)

•   	**Apache Spark / PySpark** — ETL, feature engineering, film-segment affinity modeling

*Modeling*

•   	**Spark MLlib** — Stage 1 film-segment affinity; identifies which film attribute clusters resonate with which demographic segments

•   	**XGBoost / LightGBM** — Stage 2 ranking; produces calibrated Movie Match Score per film-segment-timeslot

•   	**MLflow** — experiment tracking and model versioning

*Streaming*

•   	**Apache Kafka \+ Spark Structured Streaming** — ingests mid-week attendance events; fires MidWeekAlert when occupancy drops 20+ points below Match Score

*AI Layer*

•   	**LangGraph** — orchestrates scheduling and promotions modes

•   	**OpenAI API / Claude** — generates briefs grounded in structured model outputs

•   	**FAISS** — comparable film retrieval for promotions mode

*Visualization*

•   	**Streamlit** — manager dashboard: Match Score heatmap, schedule view, alert panel, promotional briefs

 

**Generative/Agentic AI used (if any):**

The copilot runs on LangGraph in two modes. In scheduling mode it takes the weekly Match Score grid, assigns each screen slot to the highest-confidence film-segment pairing, and labels each slot high-confidence (75%+), needs promotion (50–75%), or consider dropping (below 50%). In promotions mode, triggered by a MidWeekAlert, it identifies the segment with the biggest affinity-attendance gap, selects from a fixed playbook (loyalty email, social brief, matinee discount), and generates a structured brief with channel, message angle, and estimated uplift. A validation script checks every output against the structured input and regenerates if anything doesn’t match. Output quality is evaluated by comparing Match Score predictions against a held-out 20% MovieLens split and benchmarking AI-generated schedules against a naïve box-office-rank baseline. Managers approve before any action is taken.

 

**Safeguards:**

•   	LLM outputs are restricted to data present in the pipeline’s structured results; the copilot cannot reference a film outside the ranking output or a channel outside the predefined playbook.

•   	All agent inputs and outputs are logged to Delta Lake — every recommendation is traceable to the model version and input data that produced it.

•   	No PII used. All modeling is at the demographic segment and film-attribute level.

•   	Simulated data is clearly distinguished from real data in all outputs and evaluation reporting.

•   	API rate limits and cost controls applied to the LLM batch job.

   
**GitHub URL**: [https://github.com/sbendev/big\_data\_team\_12](https://github.com/sbendev/big_data_team_12)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJIAAABGCAYAAAAn1M23AAADwUlEQVR4Xu2c3W0TQRRGJx1QAo8IsVZKoARKoARKoATenPACJVACJbgESkgH4IkyzujYxpPMzP2u7Xuko6BkvXv2Koy9/klKDrm7Wf1V6r0naISDtNZ7T9AIB2mt956gEQ7SWu89QSMcpLXee4JGOEhrvfcEjXCQ1nrvCRrhIK313hM0wkFa670naISDtNZ7T9AIB2mt956gEQ7SWu89QSMcpLXee4JGOEhrvfcEjXCQ1nrvCRrhIK313hM0wkFa670n6IDDHeU6vXvLY7XA/YySxwkGwmGPlsc7BW8/Wh4vGMCPdPuGgx7tOi2/edxjrNPqK28/2nX68JHHDTrhkGfJ4x6Dt5sljxt0cJeWBw54li2rksVqVMvjB6+Eg51t/kVhQyHf3XD72b72QiCo4FCtZEeB21nJjuCFcKBmptU3tihWo2KsSh1wmNZ67wkauE+rLxykwtKzfdz0hz+ztuVCIKhQ3oXQ/PxVbuL3Vf7vQiAAHJ5aD6tRLecVHMDTauTVWJUa4NDCw3JuQYW3uxDvcn5BenwZ5BcH1ev2Kuczvyczn19aNnvf7zEtD5zj1bM3pE69XW3NOs/nCQbDV6P6WWAXq9L2/KrTHfvLFKvSM3vD6XT2/l/qXs/gdzNw/1cJh9Ir91/gdlYee42M2/XK/V8dHEiv3H8hvwjLbWebr0LZUYhVaSAcRq+z99/r7D7u/yoY/T+SD2gfj8FtxLJvxoUAj3HxcAA9HrsL4XZq2ZcZfZf7Pb2/5TEuFp58r9x/gdupZV+B2/XK/V8sPPEej61GGW6rln2FWJVeAU+6V+6/htuqZV8Nt+2V+78oRn+M59QHCLm9WvbVDP8AaFo2PMZFcJ+WT3sn2ymPQbi9WvYRbt/r9qrwJ49x9vAkez21GmV4G7XsI8NXpZvTxzwrRq9G+YMBPMYheDu17DtEfqDM2/V4dqsST0Chx6Zaj31sksNAhR6baj32sUkOAxV6bKr12McmOQxU6LGp1mMfm+QwUKHHplqPfWySw0CFHptqPfaxSQ4DFbIpP9fkSfaxXyGb5DBQIZu8w36FbJLDQIVs8g77FbJJDgMVssk77FfIJjkMVMgm77BfYd2TX8rK7+/KL6vkn0ne18RAhWzyDvsV7lrym+vSsqkvCiQfvmSgQjZ5h/0Kdy1P718q3ytfzV/0ZaBCNnmH/QpLS/5ES91UN5Z/m8BAhWzyDvsV1i2Hvh56/msqDFTIJu+wX+GuZfsYiX/sNB4jnQnsV8im8sfOyl1dEARBEARBEARO+Af/7d5uzdraXgAAAABJRU5ErkJggg==>