import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import shutil
import tempfile

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYSIS_PROMPT = """
You are an expert Senior Data Analyst. You are working with a pandas dataframe named 'df'.
The user has uploaded this CSV file and expects deep insights, not just raw numbers.

### Your Guidelines:
1. **Initial Review**: Before answering, mentally check the column names and data types.
2. **Analysis Mode**: If a user asks for a calculation (like total revenue or average), don't just give the number. Explain *how* you reached that number (e.g., "By summing the 'Price' column...").
3. **Accuracy First**: If a question is ambiguous, look at the dataframe structure to make the best logical guess. If you cannot find the data, tell the user exactly which columns are available.
4. **Formatting**: Use Markdown for your answers. Use **bolding** for key metrics and tables if you are comparing multiple items.
5. **No Hallucinations**: Only use the data present in the 'df' dataframe. Do not make up facts outside of this file.

### Goal:
Provide a helpful, polite, and technically accurate response to the user's question.
"""

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...), question: str = Form(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Read CSV into Pandas
        df = pd.read_csv(tmp_path)

        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Create LangChain Agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            prefix=ANALYSIS_PROMPT,
            verbose=True,
            allow_dangerous_code=True
        )

        # Execute Query
        response = agent.invoke(question)
        
        # Cleanup temp file
        os.remove(tmp_path)

        return {"answer": response["output"]}

    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": str(e)}

@app.post("/preview")
async def preview_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        df = pd.read_csv(tmp_path)
        
        # Get basic info
        num_rows = len(df)
        num_cols = len(df.columns)
        columns = df.columns.tolist()
        preview_data = df.head(5).to_dict(orient='records')
        text_preview = df.head(5).to_string()
        
        os.remove(tmp_path)
        
        return {
            "rowCount": num_rows,
            "colCount": num_cols,
            "columns": columns,
            "preview": preview_data,
            "textPreview": text_preview
        }
    except Exception as e:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Instructions: 
    # 1. Install dependencies: pip install fastapi uvicorn pandas langchain-experimental langchain-openai
    # 2. Set API Key: export OPENAI_API_KEY='your-key' (or set in environment)
    # 3. Run: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
