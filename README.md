1. Project Overview

Brief description of the RAG chatbot
Technologies used (Groq, Sentence Transformers, FastAPI, Streamlit)
Key features

2. Dataset Information

Link: https://www.datayb.com/datasets/dataset-details/datayb_dataset_details_p333awduhf2dv5t/
Brief description of tables (outlets, products, orders, reviews, etc.)
Data preprocessing approach

3. Best Outlet/Product Definitions
Best Outlet Formula:
performance_score = (total_revenue × 0.4) + (unique_customers × 0.3) + (outlet_score × 0.3 × 100)
Factors: Revenue (40%), Customers (30%), Quality Score (30%)
Best Product Formula:
performance_score = (revenue_total × 0.4) + (units_sold_total × 0.3) + (product_score × 0.3 × 100)
Factors: Revenue (40%), Units Sold (30%), Quality Score (30%)
4. Setup Instructions
bash# Clone repository
git clone <your-repo-url>
cd CHATBOTTASK

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
5. How to Run
Step 1: Preprocess Data
bashcd DataPreprocessing
python3 data_prep.py --data-dir "/path/to/dataset" --out-dir "./output"
cd ..
Step 2: Build Index
bashpython3 -m src.indexer.build_index --data-dir ./output --out-dir ./index
Step 3: Start API Server
bashpython3 main.py
# API runs on http://127.0.0.1:8000
Step 4: Start Streamlit UI (New terminal)
bashstreamlit run src/UI/frontend.py
# UI opens in browser at http://localhost:8501
6. Example Queries

"Which outlet has the highest sales in Surat?"
"What is the best product in terms of rating and sales?"
"Compare Margherita Pizza and Pepperoni Pizza"
"Which products have rating above 4.0?"