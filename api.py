from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import our custom modules
from Analytics import BookingAnalytics
from RAG_QA import BookingRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if Google API key is available
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key is None or api_key == "your_api_key_goes_here":
    print("Warning: Google API key is not properly set. RAG functionality may not work.")
    print("Please set a valid API key in the .env file.")

# Initialize the application
app = FastAPI(
    title="Hotel Booking Analytics API",
    description="REST API for hotel booking analytics and question answering",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response schemas
class Question(BaseModel):
    text: str = Field(..., description="The question text to be answered")
    max_results: Optional[int] = Field(5, description="Maximum number of results to consider")
    
class AnalyticsRequest(BaseModel):
    insight_types: List[str] = Field(
        ["revenue_trends", "cancellation_rate", "geographical_distribution", 
         "lead_time_distribution", "additional_insights"],
        description="Types of insights to return. Leave empty to return all."
    )
    format_type: str = Field("json", description="Format type for the response (json or html)")

class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    vector_db_status: bool
    llm_status: bool
    checked_at: str
    execution_time: float

# Global instances of our analytics and RAG classes
analytics_instance = None
rag_instance = None

def get_analytics():
    """Dependency to get or initialize analytics instance"""
    global analytics_instance
    if analytics_instance is None:
        analytics_instance = BookingAnalytics(db_path='data/hotel_bookings.db')
    return analytics_instance

def get_rag():
    """Dependency to get or initialize RAG instance"""
    global rag_instance
    if rag_instance is None:
        rag_instance = BookingRAG(
            db_path='data/hotel_bookings.db',
            vector_db_path='data/vector_db',
            google_api_key=api_key
        )
    return rag_instance

@app.post("/ask", response_model=Dict[str, Any])
async def answer_question(
    question: Question,
    rag: BookingRAG = Depends(get_rag)
):
    """
    Answers a question about the hotel booking data using RAG.
    
    Returns the answer, along with metadata about the request.
    """
    try:
        # Get the answer
        start_time = time.time()
        result = rag.answer_question(question.text)
        
        # Return the results
        return {
            "question": question.text,
            "answer": result["answer"],
            "execution_time": result["execution_time"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )

@app.post("/analytics", response_model=Dict[str, Any])
async def get_insights(
    request: AnalyticsRequest,
    analytics: BookingAnalytics = Depends(get_analytics)
):
    """
    Returns requested analytics insights.
    
    Accepts a list of insight types to return.
    """
    try:
        start_time = time.time()
        
        # Get all insights
        all_insights = analytics.get_all_insights()
        
        # If empty, generate them first
        if not all_insights or any(insight not in all_insights or all_insights[insight] is None for insight in request.insight_types):
            analytics.generate_all_insights()
            all_insights = analytics.get_all_insights()
        
        # Filter to requested insights
        if request.insight_types:
            filtered_insights = {k: v for k, v in all_insights.items() if k in request.insight_types}
        else:
            filtered_insights = all_insights
        
        # Add metadata
        result = {
            "insights": filtered_insights,
            "timestamp": datetime.now().isoformat(),
            "execution_time": time.time() - start_time
        }
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving analytics: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def check_health(
    analytics: BookingAnalytics = Depends(get_analytics),
    rag: BookingRAG = Depends(get_rag)
):
    """
    Checks the health of the system and its dependencies.
    
    Returns status of database connections, vector database, and LLM.
    """
    start_time = time.time()
    
    # Check database connection
    db_connected = False
    try:
        cursor = analytics.conn.cursor()
        cursor.execute("SELECT 1")
        db_connected = cursor.fetchone()[0] == 1
    except:
        db_connected = False
    
    # Check vector database
    vector_db_status = False
    try:
        # Try to get vector collections
        cols = rag.chroma_client.list_collections()
        vector_db_status = True
    except:
        vector_db_status = False
    
    # Check LLM
    llm_status = False
    try:
        # Simple test of Gemini API
        response = rag.model.generate_content("Hello")
        llm_status = response.text is not None
    except:
        llm_status = False
    
    # Determine overall status
    if db_connected and vector_db_status and llm_status:
        status = "healthy"
    elif db_connected and vector_db_status:
        status = "degraded"
    else:
        status = "unhealthy"
    
    execution_time = time.time() - start_time
    
    return HealthResponse(
        status=status,
        database_connected=db_connected,
        vector_db_status=vector_db_status,
        llm_status=llm_status,
        checked_at=datetime.now().isoformat(),
        execution_time=execution_time
    )

@app.get("/query_history", response_model=Dict[str, Any])
async def get_query_history(
    limit: int = 10,
    rag: BookingRAG = Depends(get_rag)
):
    """
    Returns recent query history.
    
    Limit parameter controls how many queries to return.
    """
    try:
        # Query the history table
        query = f"""
        SELECT 
            query_id, query_text, timestamp, execution_time
        FROM 
            {rag.query_history_table_name}
        ORDER BY 
            query_id DESC
        LIMIT {limit}
        """
        
        cursor = rag.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Format the result
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "query": row[1],
                "timestamp": row[2],
                "execution_time": row[3]
            })
        
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving query history: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Initializing services...")
    
    # Pre-initialize the global instances
    analytics = get_analytics()
    rag = get_rag()
    
    # Check if vector DB is set up
    try:
        bookings_count = len(rag.bookings_collection.get()['ids'])
        insights_count = len(rag.insights_collection.get()['ids'])
        
        if bookings_count == 0 or insights_count == 0:
            # If running for the first time, this could take a while
            # For a real production app, this would be better as a background task
            # with a "system initializing" message for API responses
            print("Vector database is empty. This will take some time to set up...")
            print("For production use, run python RAG_QA.py --setup_only first.")
    except Exception as e:
        print(f"Error checking vector database: {str(e)}")
    
    print("API initialized and ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global analytics_instance, rag_instance
    
    if analytics_instance:
        analytics_instance.close()
    
    if rag_instance:
        rag_instance.close()
    
    print("API shutdown complete, all resources released.")

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hotel Booking Analytics API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the API on')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload) 