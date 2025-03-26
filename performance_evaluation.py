import os
import json
import time
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import load_dotenv
from RAG_QA import BookingRAG

# Load environment variables
load_dotenv()

# Test question sets
EVALUATION_QUESTIONS = [
    # Revenue questions
    "What was the total revenue for July 2017?",
    "Which month had the highest revenue in 2016?",
    "What is the average revenue per booking for Resort Hotels?",
    
    # Cancellation questions
    "Which country has the highest cancellation rate?",
    "What is the overall cancellation rate for all bookings?",
    "Does the City Hotel or Resort Hotel have a higher cancellation rate?",
    
    # Pricing questions
    "What is the average daily rate for a booking?",
    "Which room type has the highest average price?",
    "How does the price vary between weekdays and weekends?",
    
    # Lead time questions
    "What is the average lead time for bookings?",
    "Do customers who book far in advance tend to pay more or less?",
    "Is there a correlation between lead time and cancellation rate?",
    
    # Complex questions
    "What is the most popular time of year for bookings, and does it differ between hotel types?",
    "Which market segment has the lowest cancellation rate but highest average daily rate?",
    "How do booking patterns differ between repeat and non-repeat guests?",
]

# Ground truth answers (would be manually evaluated in a real scenario)
# This is a simplified version for the assignment
GROUND_TRUTH = {
    "What was the total revenue for July 2017?": {
        "keywords": ["revenue", "july", "2017"],
        "contains_number": True
    },
    "Which country has the highest cancellation rate?": {
        "keywords": ["country", "cancellation", "rate", "highest"],
        "contains_country": True
    },
    "What is the average daily rate for a booking?": {
        "keywords": ["average", "daily", "rate", "price"],
        "contains_number": True,
        "contains_currency": True
    }
}

def evaluate_rag_accuracy(rag_instance, questions=EVALUATION_QUESTIONS):
    """
    Evaluate the accuracy of the RAG system using a set of test questions
    """
    results = []
    
    print("Evaluating RAG accuracy...")
    for question in tqdm(questions):
        start_time = time.time()
        response = rag_instance.answer_question(question)
        execution_time = time.time() - start_time
        
        # Check if the answer contains key elements from ground truth
        # This is a simplified check - in a real system, you'd use more sophisticated 
        # methods like semantic similarity, ROUGE, or human evaluation
        accuracy_score = None
        answer_eval = {"has_answer": len(response["answer"]) > 10}
        
        if question in GROUND_TRUTH:
            truth = GROUND_TRUTH[question]
            answer_text = response["answer"].lower()
            
            # Check for required keywords
            keyword_matches = sum(1 for kw in truth["keywords"] if kw.lower() in answer_text)
            keyword_score = keyword_matches / len(truth["keywords"])
            
            # Check for numbers if required
            number_score = 1.0
            if truth.get("contains_number", False):
                import re
                has_number = bool(re.search(r'\d+(\.\d+)?', answer_text))
                number_score = 1.0 if has_number else 0.0
            
            # Check for currency if required
            currency_score = 1.0
            if truth.get("contains_currency", False):
                has_currency = '$' in answer_text or 'usd' in answer_text or 'dollars' in answer_text
                currency_score = 1.0 if has_currency else 0.0
            
            # Check for country names if required
            country_score = 1.0
            if truth.get("contains_country", False):
                # This is a simplified check
                common_countries = ['usa', 'uk', 'portugal', 'spain', 'france', 'germany', 'italy']
                has_country = any(country in answer_text for country in common_countries)
                country_score = 1.0 if has_country else 0.0
            
            # Calculate overall accuracy score
            accuracy_score = (keyword_score + number_score + currency_score + country_score) / 4
            
            answer_eval.update({
                "keyword_score": keyword_score,
                "number_score": number_score,
                "currency_score": currency_score,
                "country_score": country_score,
                "accuracy_score": accuracy_score
            })
        
        # Store result
        results.append({
            "question": question,
            "answer": response["answer"],
            "execution_time": execution_time,
            "context_count": len(response["context_used"]),
            "evaluation": answer_eval,
            "accuracy_score": accuracy_score
        })
    
    return results

def evaluate_api_performance(api_url="http://localhost:8000", n_requests=50):
    """
    Evaluate the performance of the API endpoints
    """
    results = {
        "analytics_endpoint": [],
        "ask_endpoint": [],
        "health_endpoint": []
    }
    
    # Test analytics endpoint
    print(f"Testing analytics endpoint with {n_requests} requests...")
    for _ in tqdm(range(n_requests)):
        payload = {
            "insight_types": ["revenue_trends", "cancellation_rate"],
            "format_type": "json"
        }
        start_time = time.time()
        response = requests.post(f"{api_url}/analytics", json=payload)
        execution_time = time.time() - start_time
        
        results["analytics_endpoint"].append({
            "status_code": response.status_code,
            "execution_time": execution_time,
            "success": response.status_code == 200
        })
    
    # Test ask endpoint with random questions
    print(f"Testing ask endpoint with {n_requests} requests...")
    for i in tqdm(range(n_requests)):
        # Cycle through questions
        question = EVALUATION_QUESTIONS[i % len(EVALUATION_QUESTIONS)]
        payload = {
            "text": question,
            "max_results": 5
        }
        start_time = time.time()
        response = requests.post(f"{api_url}/ask", json=payload)
        execution_time = time.time() - start_time
        
        results["ask_endpoint"].append({
            "status_code": response.status_code,
            "execution_time": execution_time,
            "success": response.status_code == 200,
            "question": question
        })
    
    # Test health endpoint
    print(f"Testing health endpoint with {n_requests} requests...")
    for _ in tqdm(range(n_requests)):
        start_time = time.time()
        response = requests.get(f"{api_url}/health")
        execution_time = time.time() - start_time
        
        results["health_endpoint"].append({
            "status_code": response.status_code,
            "execution_time": execution_time,
            "success": response.status_code == 200
        })
    
    return results

def analyze_performance_results(rag_results, api_results):
    """
    Analyze and visualize performance evaluation results
    """
    # Create results directory if it doesn't exist
    if not os.path.exists("evaluation_results"):
        os.makedirs("evaluation_results")
    
    # 1. Analyze RAG accuracy
    accuracy_scores = [r["accuracy_score"] for r in rag_results if r["accuracy_score"] is not None]
    if accuracy_scores:
        avg_accuracy = np.mean(accuracy_scores)
        print(f"Average accuracy score: {avg_accuracy:.2f}")
        
        plt.figure(figsize=(10, 6))
        plt.hist(accuracy_scores, bins=10, alpha=0.7)
        plt.title("Distribution of Accuracy Scores")
        plt.xlabel("Accuracy Score")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig("evaluation_results/accuracy_distribution.png")
        plt.close()
    
    # 2. Analyze RAG response times
    response_times = [r["execution_time"] for r in rag_results]
    avg_response_time = np.mean(response_times)
    print(f"Average RAG response time: {avg_response_time:.2f} seconds")
    
    plt.figure(figsize=(10, 6))
    plt.hist(response_times, bins=15, alpha=0.7)
    plt.title("Distribution of RAG Response Times")
    plt.xlabel("Response Time (seconds)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig("evaluation_results/rag_response_times.png")
    plt.close()
    
    # 3. Analyze API performance
    api_analytics_times = [r["execution_time"] for r in api_results["analytics_endpoint"] if r["success"]]
    api_ask_times = [r["execution_time"] for r in api_results["ask_endpoint"] if r["success"]]
    api_health_times = [r["execution_time"] for r in api_results["health_endpoint"] if r["success"]]
    
    if api_analytics_times:
        print(f"Average /analytics endpoint response time: {np.mean(api_analytics_times):.4f} seconds")
    if api_ask_times:
        print(f"Average /ask endpoint response time: {np.mean(api_ask_times):.4f} seconds")
    if api_health_times:
        print(f"Average /health endpoint response time: {np.mean(api_health_times):.4f} seconds")
    
    # Plot API response times comparison
    plt.figure(figsize=(12, 7))
    plt.boxplot([api_analytics_times, api_ask_times, api_health_times], 
                labels=['/analytics', '/ask', '/health'])
    plt.title("API Endpoint Response Times")
    plt.ylabel("Response Time (seconds)")
    plt.grid(True, alpha=0.3)
    plt.savefig("evaluation_results/api_response_times.png")
    plt.close()
    
    # Create summary report
    summary = {
        "rag_evaluation": {
            "questions_tested": len(rag_results),
            "questions_with_accuracy_score": len(accuracy_scores),
            "average_accuracy": float(np.mean(accuracy_scores)) if accuracy_scores else None,
            "average_response_time": float(avg_response_time),
            "min_response_time": float(min(response_times)),
            "max_response_time": float(max(response_times))
        },
        "api_evaluation": {
            "analytics_endpoint": {
                "requests": len(api_results["analytics_endpoint"]),
                "success_rate": sum(1 for r in api_results["analytics_endpoint"] if r["success"]) / len(api_results["analytics_endpoint"]),
                "average_response_time": float(np.mean(api_analytics_times)) if api_analytics_times else None
            },
            "ask_endpoint": {
                "requests": len(api_results["ask_endpoint"]),
                "success_rate": sum(1 for r in api_results["ask_endpoint"] if r["success"]) / len(api_results["ask_endpoint"]),
                "average_response_time": float(np.mean(api_ask_times)) if api_ask_times else None
            },
            "health_endpoint": {
                "requests": len(api_results["health_endpoint"]),
                "success_rate": sum(1 for r in api_results["health_endpoint"] if r["success"]) / len(api_results["health_endpoint"]),
                "average_response_time": float(np.mean(api_health_times)) if api_health_times else None
            }
        }
    }
    
    # Save summary as JSON
    with open("evaluation_results/performance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Performance evaluation results saved to evaluation_results/")
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate the performance of the Hotel Booking Analytics system')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', 
                        help='URL of the API server to test')
    parser.add_argument('--rag_only', action='store_true', 
                        help='Only evaluate RAG accuracy, not API performance')
    parser.add_argument('--api_only', action='store_true', 
                        help='Only evaluate API performance, not RAG accuracy')
    parser.add_argument('--n_requests', type=int, default=20, 
                        help='Number of requests to make for API testing')
    parser.add_argument('--db_path', type=str, default='data/hotel_bookings.db', 
                        help='Path to the database for RAG testing')
    
    args = parser.parse_args()
    
    # Check Google API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key is None or api_key == "your_api_key_goes_here":
        print("Warning: Google API key is not properly set. RAG evaluation may fail.")
        print("Please set a valid API key in the .env file.")
    
    rag_results = None
    api_results = None
    
    # Evaluate RAG accuracy
    if not args.api_only:
        print("Initializing RAG system for accuracy evaluation...")
        rag = BookingRAG(
            db_path=args.db_path,
            vector_db_path='data/vector_db',
            google_api_key=api_key
        )
        rag_results = evaluate_rag_accuracy(rag)
        print(f"RAG evaluation complete: tested {len(rag_results)} questions")
    
    # Evaluate API performance
    if not args.rag_only:
        print(f"Testing API performance at {args.api_url}...")
        api_results = evaluate_api_performance(api_url=args.api_url, n_requests=args.n_requests)
        print("API performance evaluation complete")
    
    # Analyze results
    if rag_results or api_results:
        print("Analyzing performance evaluation results...")
        summary = analyze_performance_results(rag_results or [], api_results or {})
        print("Performance analysis complete")
    else:
        print("No evaluation was performed. Please check your arguments.")
    
    print("Done!") 