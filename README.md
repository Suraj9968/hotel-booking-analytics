# LLM-Powered Hotel Booking Analytics System

This project implements a hotel booking analytics system with a retrieval-augmented question answering (RAG) capability. It allows users to extract insights from booking data and ask natural language questions through a REST API.

Here are some screenshots of the App running 
<img width="1710" alt="analytics" src="https://github.com/Suraj9968/hotel-booking-analytics/blob/main/Screenshots/Screenshot%202025-03-26%20182316.png">
<img width="1710" alt="rag_QA" src="https://github.com/Suraj9968/hotel-booking-analytics/blob/main/Screenshots/Screenshot%202025-03-26%20182409.png">
<img width="1710" alt="Query_history" src="https://github.com/Suraj9968/hotel-booking-analytics/blob/main/Screenshots/Screenshot%202025-03-26%20182435.png">
<img width="1710" alt="Health" src="https://github.com/Suraj9968/hotel-booking-analytics/blob/main/Screenshots/Screenshot%202025-03-26%20182453.png">

## Features

- **Data Processing**: Clean and preprocess hotel booking data from CSV files
- **Analytics & Reporting**: Generate insights on revenue trends, cancellation rates, geographical distribution, and more
- **Question Answering**: Retrieve relevant context from the database to answer natural language questions
- **REST API**: Expose analytics and QA functionality through a well-defined API
- **Real-time Updates**: Support for updating the system when new data arrives
- **Health Monitoring**: Endpoint to check the system status and dependencies
- **Interactive Dashboard**: Streamlit-based frontend for data visualization and interaction

## Setup Instructions

### Prerequisites

- Python 3.8+ 
- pip (Python package manager)

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google API key for Gemini:
   - Create a `.env` file in the project root
   - Add the following line: `GOOGLE_API_KEY=your_api_key_here`
   - Replace `your_api_key_here` with your actual Google API key
   - Optionally, add `API_URL=http://localhost:8000` to specify the API endpoint URL

### Running the Application

#### Step 1: Data Preprocessing
```
python Preprocessing.py
```
This will:
- Clean and preprocess the CSV data
- Store the processed data in both CSV and SQLite formats

#### Step 2: Generate Analytics
```
python Analytics.py
```
This will:
- Generate analytics insights from the processed data
- Store insights in the database for quick retrieval
- Create visualization images in the `visualizations` folder

#### Step 3: Set up the Vector Database (Optional but Recommended)
```
python RAG_QA.py --setup_only
```
This will:
- Embed booking records and insights in a vector database
- Set up collections for semantic search
- This step can take some time to complete but only needs to be done once

#### Step 4: Start the API Server
```
python api.py
```
This will:
- Start the FastAPI server on http://0.0.0.0:8000
- Initialize all required components

#### Step 5: Start the Streamlit Frontend (in a new terminal)
```
streamlit run frontend.py
```
This will:
- Start the Streamlit web application
- Connect to the API server
- Open a browser with the interactive dashboard

### API Documentation

After starting the API server, you can access the interactive API documentation at:
```
http://localhost:8000/docs
```

## API Endpoints

### POST /analytics
Returns analytics insights about the hotel booking data.

Example request:
```json
{
  "insight_types": ["revenue_trends", "cancellation_rate"],
  "format_type": "json"
}
```

### POST /ask
Answers a natural language question about the hotel booking data.

Example request:
```json
{
  "text": "What is the average price of a hotel booking?",
  "max_results": 5
}
```

### GET /health
Checks the health of the system and its dependencies.

### GET /query_history
Returns the history of recent queries made to the system.

## Streamlit Frontend

The Streamlit frontend provides an interactive user interface with the following features:

- **Analytics Dashboard**: Visualize key insights with interactive charts
- **Question Answering**: Ask natural language questions about the booking data
- **Query History**: View previously asked questions and their answers
- **System Health**: Check the status of the system and its components

## Sample Test Queries

Here are some example questions you can ask the system:

1. "Show me total revenue for July 2017."
2. "Which locations had the highest booking cancellations?"
3. "What is the average price of a hotel booking?"
4. "What is the average lead time for Resort Hotel bookings?"
5. "Which market segment has the highest cancellation rate?"
6. "What is the distribution of bookings across different countries?"
7. "How long do guests typically stay in City Hotels vs Resort Hotels?"
8. "What is the most common room type booked?"

## Implementation Details

### Architecture

The system consists of four main components:

1. **Data Processing & Analytics Module**: Handles data cleaning, preprocessing, and insight generation
2. **Retrieval-Augmented Question Answering (RAG) Module**: Embeds data in a vector database and uses LLM to answer questions
3. **REST API**: Provides endpoints to access the system's functionality
4. **Web Frontend**: Offers an interactive dashboard for visualizing data and asking questions

### Technologies Used

- **Data Processing**: Pandas, NumPy, SQLite
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **LLM**: Google Gemini
- **API Framework**: FastAPI
- **Frontend**: Streamlit

