import os
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from Analytics import BookingAnalytics
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class BookingRAG:
    def __init__(self, 
                 db_path='data/hotel_bookings.db',
                 vector_db_path='data/vector_db',
                 google_api_key=None,
                 model_name="gemini-2.0-flash",
                 embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the Retrieval-Augmented Question Answering system
        
        Args:
            db_path: Path to the SQLite database
            vector_db_path: Path to store the vector database
            google_api_key: API key for Google Gemini
            model_name: Gemini model name to use
            embedding_model: Sentence transformer model for embeddings
        """
        # Initialize SQLite connection
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        
        # Initialize analytics engine to access precomputed insights
        self.analytics = BookingAnalytics(db_path=db_path)
        
        # Setup vector database
        self.vector_db_path = vector_db_path
        if not os.path.exists(vector_db_path):
            os.makedirs(vector_db_path)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
        
        # Set up sentence transformer for embeddings
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Google Gemini
        self.setup_gemini(google_api_key, model_name)
        
        # Define collections
        self.bookings_collection_name = "bookings_data"
        self.insights_collection_name = "insights_data"
        
        # Create or get collections
        self.setup_collections()
        
        # Query history table name
        self.query_history_table_name = 'rag_query_history'
        self.setup_query_history_table()
    
    def setup_gemini(self, api_key, model_name):
        """Set up Google Gemini API"""
        if api_key is None:
            # Try to get from environment
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        if api_key is None:
            raise ValueError("Google API key is required. Please provide it or set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def setup_collections(self):
        """Create or get collections in ChromaDB"""
        # Get list of existing collections
        existing_collections = [col.name for col in self.chroma_client.list_collections()]
        
        # Create or get bookings collection
        if self.bookings_collection_name not in existing_collections:
            self.bookings_collection = self.chroma_client.create_collection(
                name=self.bookings_collection_name,
                metadata={"description": "Hotel booking data embeddings"}
            )
        else:
            self.bookings_collection = self.chroma_client.get_collection(
                name=self.bookings_collection_name
            )
            
        # Create or get insights collection
        if self.insights_collection_name not in existing_collections:
            self.insights_collection = self.chroma_client.create_collection(
                name=self.insights_collection_name,
                metadata={"description": "Precomputed insights embeddings"}
            )
        else:
            self.insights_collection = self.chroma_client.get_collection(
                name=self.insights_collection_name
            )
    
    def setup_query_history_table(self):
        """Create query history table if it doesn't exist"""
        cursor = self.conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.query_history_table_name} (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            response TEXT,
            timestamp TEXT,
            execution_time REAL,
            feedback TEXT
        )
        ''')
        self.conn.commit()
    
    def _log_query(self, query_text, response, execution_time, feedback=None):
        """Log a query to the query history table"""
        timestamp = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute(f'''
        INSERT INTO {self.query_history_table_name} (query_text, response, timestamp, execution_time, feedback)
        VALUES (?, ?, ?, ?, ?)
        ''', (query_text, json.dumps(response), timestamp, execution_time, feedback))
        
        self.conn.commit()
    
    def embed_bookings_data(self, batch_size=1000):
        """
        Embed booking data from the database and store in vector database
        with progress bar
        """
        # Get total count for progress bar
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM bookings_table")
        total_rows = cursor.fetchone()[0]
        
        # Create bookings data chunks for embedding
        print(f"Embedding {total_rows} booking records...")
        
        # Process in batches for memory efficiency
        offset = 0
        with tqdm(total=total_rows, desc="Embedding bookings") as pbar:
            while offset < total_rows:
                # Get batch of data
                query = f"""
                SELECT 
                    booking_id, hotel, is_canceled, lead_time, arrival_date,
                    arrival_date_year, arrival_date_month, country, market_segment, 
                    reserved_room_type, adr, revenue, total_stay_duration
                FROM 
                    (SELECT 
                        rowid as booking_id, *
                    FROM 
                        bookings_table
                    LIMIT {batch_size} OFFSET {offset})
                """
                
                df_batch = pd.read_sql_query(query, self.conn)
                
                # Create text representation of each booking
                texts = []
                ids = []
                metadatas = []
                
                for _, row in df_batch.iterrows():
                    # Create a text representation that captures key information
                    text = f"""
                    Booking ID: {row['booking_id']}
                    Hotel: {row['hotel']}
                    Status: {"Canceled" if row['is_canceled'] == 1 else "Confirmed"}
                    Country: {row['country']}
                    Arrival Date: {row['arrival_date_year']}-{row['arrival_date_month']}
                    Lead Time: {row['lead_time']} days
                    Room Type: {row['reserved_room_type']}
                    Daily Rate: ${row['adr']}
                    Total Revenue: ${row['revenue']}
                    Stay Duration: {row['total_stay_duration']} days
                    Market Segment: {row['market_segment']}
                    """
                    
                    # Clean and normalize text
                    text = text.strip()
                    
                    texts.append(text)
                    ids.append(f"booking_{row['booking_id']}")
                    
                    # Add metadata for more efficient filtering
                    metadatas.append({
                        "hotel": row['hotel'],
                        "year": int(row['arrival_date_year']),
                        "month": row['arrival_date_month'],
                        "country": row['country'],
                        "is_canceled": bool(row['is_canceled']),
                        "revenue": float(row['revenue']),
                        "adr": float(row['adr'])
                    })
                
                # Embed texts
                embeddings = []
                for text in texts:
                    embedding = self.embedding_model.encode(text)
                    embeddings.append(embedding.tolist())
                
                # Add to collection
                self.bookings_collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    ids=ids,
                    metadatas=metadatas
                )
                
                # Update progress bar
                pbar.update(len(df_batch))
                offset += batch_size
        
        print(f"Successfully embedded {total_rows} booking records.")
    
    def embed_insights(self):
        """
        Embed precomputed insights data and store in vector database
        with progress bar
        """
        # Get all insights
        insights = self.analytics.get_all_insights()
        
        if not insights:
            print("No insights found. Please run the analytics first.")
            return
        
        # Create text representations for each insight type
        insight_texts = []
        insight_ids = []
        insight_metadatas = []
        
        # Process revenue trends
        if 'revenue_trends' in insights and insights['revenue_trends']:
            revenue_data = insights['revenue_trends']
            
            # Create texts for each month's revenue
            for i, date in enumerate(revenue_data['dates']):
                revenue = revenue_data['revenues'][i]
                booking_count = revenue_data['booking_counts'][i]
                
                text = f"""
                Revenue Insight:
                Month: {date}
                Total Revenue: ${revenue}
                Number of Bookings: {booking_count}
                Average Revenue per Booking: ${revenue/booking_count if booking_count > 0 else 0}
                """
                
                insight_texts.append(text.strip())
                insight_ids.append(f"revenue_trend_{date}")
                insight_metadatas.append({
                    "insight_type": "revenue_trends",
                    "date": date,
                    "revenue": float(revenue),
                    "booking_count": int(booking_count)
                })
        
        # Process cancellation rates
        if 'cancellation_rate' in insights and insights['cancellation_rate']:
            cancellation_data = insights['cancellation_rate']
            
            # Add overall cancellation rate
            text = f"""
            Cancellation Rate Insight:
            Overall cancellation rate: {cancellation_data['overall']['cancellation_rate']}%
            Total bookings: {cancellation_data['overall']['total_bookings']}
            Canceled bookings: {cancellation_data['overall']['canceled_bookings']}
            """
            
            insight_texts.append(text.strip())
            insight_ids.append("cancellation_rate_overall")
            insight_metadatas.append({
                "insight_type": "cancellation_rate",
                "level": "overall",
                "rate": float(cancellation_data['overall']['cancellation_rate']),
                "total_bookings": int(cancellation_data['overall']['total_bookings']),
                "canceled_bookings": int(cancellation_data['overall']['canceled_bookings'])
            })
            
            # Add hotel-specific cancellation rates
            for i, hotel in enumerate(cancellation_data['by_hotel']['hotels']):
                rate = cancellation_data['by_hotel']['cancellation_rates'][i]
                total = cancellation_data['by_hotel']['total_bookings'][i]
                canceled = cancellation_data['by_hotel']['canceled_bookings'][i]
                
                text = f"""
                Cancellation Rate Insight:
                Hotel: {hotel}
                Cancellation rate: {rate}%
                Total bookings: {total}
                Canceled bookings: {canceled}
                """
                
                insight_texts.append(text.strip())
                insight_ids.append(f"cancellation_rate_{hotel}")
                insight_metadatas.append({
                    "insight_type": "cancellation_rate",
                    "level": "hotel",
                    "hotel": hotel,
                    "rate": float(rate),
                    "total_bookings": int(total),
                    "canceled_bookings": int(canceled)
                })
        
        # Process geographical distribution
        if 'geographical_distribution' in insights and insights['geographical_distribution']:
            geo_data = insights['geographical_distribution']
            
            # Add country-specific booking data
            for i, country in enumerate(geo_data['countries']):
                booking_count = geo_data['booking_counts'][i]
                confirmed = geo_data['confirmed_bookings'][i]
                canceled = geo_data['canceled_bookings'][i]
                revenue = geo_data['total_revenue'][i]
                
                # Only add if there are significant bookings
                if booking_count > 10:
                    text = f"""
                    Geographical Distribution Insight:
                    Country: {country}
                    Total bookings: {booking_count}
                    Confirmed bookings: {confirmed}
                    Canceled bookings: {canceled}
                    Cancellation rate: {(canceled/booking_count)*100 if booking_count > 0 else 0}%
                    Total revenue: ${revenue}
                    """
                    
                    insight_texts.append(text.strip())
                    insight_ids.append(f"geo_dist_{country}")
                    insight_metadatas.append({
                        "insight_type": "geographical_distribution",
                        "country": country,
                        "booking_count": int(booking_count),
                        "confirmed_bookings": int(confirmed),
                        "canceled_bookings": int(canceled),
                        "total_revenue": float(revenue)
                    })
        
        # Process lead time distribution
        if 'lead_time_distribution' in insights and insights['lead_time_distribution']:
            lead_time_data = insights['lead_time_distribution']
            
            # Add lead time statistics
            text = f"""
            Lead Time Distribution Insight:
            Minimum lead time: {lead_time_data['statistics']['min']} days
            Maximum lead time: {lead_time_data['statistics']['max']} days
            Mean lead time: {lead_time_data['statistics']['mean']} days
            Median lead time: {lead_time_data['statistics']['median']} days
            25th percentile: {lead_time_data['statistics']['percentiles']['25']} days
            75th percentile: {lead_time_data['statistics']['percentiles']['75']} days
            90th percentile: {lead_time_data['statistics']['percentiles']['90']} days
            """
            
            insight_texts.append(text.strip())
            insight_ids.append("lead_time_stats")
            insight_metadatas.append({
                "insight_type": "lead_time_distribution",
                "subtype": "statistics",
                "min": float(lead_time_data['statistics']['min']),
                "max": float(lead_time_data['statistics']['max']),
                "mean": float(lead_time_data['statistics']['mean']),
                "median": float(lead_time_data['statistics']['median'])
            })
            
            # Add lead time distribution
            for i, bin_label in enumerate(lead_time_data['distribution']['bins']):
                count = lead_time_data['distribution']['counts'][i]
                
                text = f"""
                Lead Time Distribution Insight:
                Lead time range: {bin_label}
                Number of bookings: {count}
                """
                
                insight_texts.append(text.strip())
                insight_ids.append(f"lead_time_dist_{bin_label}")
                insight_metadatas.append({
                    "insight_type": "lead_time_distribution",
                    "subtype": "distribution",
                    "bin": bin_label,
                    "count": int(count)
                })
        
        # Process additional insights
        if 'additional_insights' in insights and insights['additional_insights']:
            add_insights = insights['additional_insights']
            
            # Process ADR by room type
            if 'adr_by_room_type' in add_insights:
                adr_data = add_insights['adr_by_room_type']
                
                for i in range(len(adr_data['hotel_types'])):
                    hotel = adr_data['hotel_types'][i]
                    room = adr_data['room_types'][i]
                    rate = adr_data['avg_rates'][i]
                    count = adr_data['booking_counts'][i]
                    
                    text = f"""
                    Average Daily Rate Insight:
                    Hotel: {hotel}
                    Room Type: {room}
                    Average Rate: ${rate}
                    Number of Bookings: {count}
                    """
                    
                    insight_texts.append(text.strip())
                    insight_ids.append(f"adr_{hotel}_{room}")
                    insight_metadatas.append({
                        "insight_type": "adr_by_room_type",
                        "hotel": hotel,
                        "room_type": room,
                        "avg_rate": float(rate),
                        "booking_count": int(count)
                    })
            
            # Process market segment analysis
            if 'market_segment_analysis' in add_insights:
                segment_data = add_insights['market_segment_analysis']
                
                for i, segment in enumerate(segment_data['segments']):
                    count = segment_data['booking_counts'][i]
                    rate = segment_data['cancellation_rates'][i]
                    adr = segment_data['avg_rates'][i]
                    duration = segment_data['avg_stay_durations'][i]
                    
                    text = f"""
                    Market Segment Insight:
                    Segment: {segment}
                    Number of Bookings: {count}
                    Cancellation Rate: {rate}%
                    Average Daily Rate: ${adr}
                    Average Stay Duration: {duration} days
                    """
                    
                    insight_texts.append(text.strip())
                    insight_ids.append(f"segment_{segment}")
                    insight_metadatas.append({
                        "insight_type": "market_segment_analysis",
                        "segment": segment,
                        "booking_count": int(count),
                        "cancellation_rate": float(rate),
                        "avg_rate": float(adr),
                        "avg_duration": float(duration)
                    })
            
            # Process room type distribution
            if 'room_type_distribution' in add_insights:
                room_type_data = add_insights['room_type_distribution']
                
                # Add overall summary
                total_bookings = sum(room_type_data['booking_counts'])
                most_common_index = room_type_data['booking_counts'].index(max(room_type_data['booking_counts']))
                most_common_room = room_type_data['room_types'][most_common_index]
                most_common_count = room_type_data['booking_counts'][most_common_index]
                
                text = f"""
                Room Type Distribution Summary:
                Most common room type: {most_common_room}
                Number of bookings for most common room: {most_common_count}
                Percentage of total bookings: {(most_common_count/total_bookings)*100:.1f}%
                Total bookings across all room types: {total_bookings}
                """
                
                insight_texts.append(text.strip())
                insight_ids.append(f"room_type_summary")
                insight_metadatas.append({
                    "insight_type": "room_type_distribution",
                    "subtype": "summary",
                    "most_common_room": most_common_room,
                    "most_common_count": int(most_common_count),
                    "total_bookings": int(total_bookings)
                })
                
                # Add individual room type insights
                for i, room_type in enumerate(room_type_data['room_types']):
                    count = room_type_data['booking_counts'][i]
                    confirmed = room_type_data['confirmed_bookings'][i]
                    canceled = room_type_data['canceled_bookings'][i]
                    rate = room_type_data['avg_rates'][i]
                    duration = room_type_data['avg_stay_durations'][i]
                    
                    text = f"""
                    Room Type Insight:
                    Room Type: {room_type}
                    Total Bookings: {count}
                    Confirmed Bookings: {confirmed}
                    Canceled Bookings: {canceled}
                    Average Daily Rate: ${rate}
                    Average Stay Duration: {duration} days
                    """
                    
                    insight_texts.append(text.strip())
                    insight_ids.append(f"room_type_{room_type}")
                    insight_metadatas.append({
                        "insight_type": "room_type_distribution",
                        "room_type": room_type,
                        "booking_count": int(count),
                        "confirmed_bookings": int(confirmed),
                        "canceled_bookings": int(canceled),
                        "avg_rate": float(rate),
                        "avg_duration": float(duration)
                    })
        
        # Embed and store all insights with a progress bar
        print(f"Embedding {len(insight_texts)} insights...")
        
        embeddings = []
        with tqdm(total=len(insight_texts), desc="Embedding insights") as pbar:
            for text in insight_texts:
                embedding = self.embedding_model.encode(text)
                embeddings.append(embedding.tolist())
                pbar.update(1)
            
        # Add to collection
        self.insights_collection.add(
            documents=insight_texts,
            embeddings=embeddings,
            ids=insight_ids,
            metadatas=insight_metadatas
        )
        
        print(f"Successfully embedded {len(insight_texts)} insights.")
    
    def query_vector_db(self, query, n_results=5):
        """
        Query the vector database for relevant context
        
        Args:
            query: User query text
            n_results: Number of results to retrieve from each collection
            
        Returns:
            List of context strings
        """
        # Encode the query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query both collections
        bookings_results = self.bookings_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        insights_results = self.insights_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Combine results
        context_pieces = []
        
        # Add booking results
        if bookings_results and len(bookings_results['documents']) > 0:
            for doc, score, metadata in zip(
                bookings_results['documents'][0],
                bookings_results['distances'][0],
                bookings_results['metadatas'][0]
            ):
                # Only add if relevance score is good
                if score < 1.0:  # Lower distance means more relevant in vector similarity
                    context_pieces.append(f"BOOKING DATA (Relevance: {1-score:.2f}):\n{doc}")
        
        # Add insight results
        if insights_results and len(insights_results['documents']) > 0:
            for doc, score, metadata in zip(
                insights_results['documents'][0],
                insights_results['distances'][0],
                insights_results['metadatas'][0]
            ):
                # Only add if relevance score is good
                if score < 1.0:  # Lower distance means more relevant
                    context_pieces.append(f"INSIGHT DATA (Relevance: {1-score:.2f}):\n{doc}")
        
        return context_pieces
    
    def get_sql_results(self, query):
        """
        Get SQL query results based on the user query
        Uses pattern matching to identify specific query types
        
        Args:
            query: User query text
            
        Returns:
            SQL results as a string or None if no match
        """
        # Check for revenue by time period
        if ("revenue" in query.lower() and 
            any(month.lower() in query.lower() for month in 
                ["january", "february", "march", "april", "may", "june", 
                 "july", "august", "september", "october", "november", "december"])):
            
            # Extract month and year if present
            months = ["january", "february", "march", "april", "may", "june", 
                      "july", "august", "september", "october", "november", "december"]
            month_num = None
            for i, month in enumerate(months):
                if month.lower() in query.lower():
                    month_num = i + 1
                    break
            
            # Extract year if present
            year = None
            for y in range(2015, 2023):  # Reasonable range of years
                if str(y) in query:
                    year = y
                    break
            
            # Construct SQL query based on what was found
            sql = """
            SELECT 
                SUM(revenue) as total_revenue,
                COUNT(*) as booking_count,
                AVG(adr) as avg_daily_rate
            FROM 
                bookings_table
            WHERE 
                is_canceled = 0
            """
            
            if month_num:
                sql += f" AND arrival_date_month_num = {month_num}"
            
            if year:
                sql += f" AND arrival_date_year = {year}"
            
            df = pd.read_sql_query(sql, self.conn)
            
            return f"""
            SQL Results for Revenue Query:
            Total Revenue: ${df['total_revenue'].iloc[0]:.2f}
            Number of Bookings: {df['booking_count'].iloc[0]}
            Average Daily Rate: ${df['avg_daily_rate'].iloc[0]:.2f}
            """
        
        # Check for cancellation by location
        elif "cancellation" in query.lower() and any(word in query.lower() for word in ["location", "country", "where", "place"]):
            sql = """
            SELECT 
                country,
                COUNT(*) as total_bookings,
                SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
                (SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as cancellation_rate
            FROM 
                bookings_table
            GROUP BY 
                country
            HAVING 
                total_bookings > 10
            ORDER BY 
                cancellation_rate DESC
            LIMIT 10
            """
            
            df = pd.read_sql_query(sql, self.conn)
            
            results = "SQL Results for Cancellations by Location:\n"
            for _, row in df.iterrows():
                results += f"Country: {row['country']}, Cancellation Rate: {row['cancellation_rate']:.2f}%, "
                results += f"Canceled: {row['canceled_bookings']} of {row['total_bookings']} bookings\n"
            
            return results
        
        # Check for average price query
        elif "average" in query.lower() and any(word in query.lower() for word in ["price", "rate", "cost", "adr"]):
            sql = """
            SELECT 
                hotel,
                COUNT(*) as booking_count,
                AVG(adr) as avg_daily_rate
            FROM 
                bookings_table
            WHERE 
                is_canceled = 0
                AND adr > 0
            GROUP BY 
                hotel
            """
            
            df = pd.read_sql_query(sql, self.conn)
            
            overall_avg = df['avg_daily_rate'].mean()
            
            results = f"SQL Results for Average Price Query:\nOverall Average Daily Rate: ${overall_avg:.2f}\n\n"
            results += "Breakdown by Hotel Type:\n"
            
            for _, row in df.iterrows():
                results += f"Hotel: {row['hotel']}, Average Daily Rate: ${row['avg_daily_rate']:.2f} "
                results += f"(Based on {row['booking_count']} bookings)\n"
            
            return results
            
        # Check for room type queries
        elif any(word in query.lower() for word in ["room", "type", "accommodation"]) and any(word in query.lower() for word in ["common", "most", "popular", "frequently", "booked"]):
            sql = """
            SELECT 
                reserved_room_type,
                COUNT(*) as booking_count,
                SUM(CASE WHEN is_canceled = 0 THEN 1 ELSE 0 END) as confirmed_bookings,
                AVG(adr) as avg_daily_rate
            FROM 
                bookings_table
            GROUP BY 
                reserved_room_type
            ORDER BY 
                booking_count DESC
            """
            
            df = pd.read_sql_query(sql, self.conn)
            
            # Get the most common room type
            most_common_room = df.iloc[0]
            
            results = f"SQL Results for Room Type Query:\n"
            results += f"The most common room type booked is '{most_common_room['reserved_room_type']}' with {most_common_room['booking_count']} bookings "
            results += f"({most_common_room['confirmed_bookings']} confirmed) and an average daily rate of ${most_common_room['avg_daily_rate']:.2f}.\n\n"
            
            # Add breakdown of all room types
            results += "Breakdown of room types by popularity:\n"
            
            for _, row in df.iterrows():
                results += f"Room Type: {row['reserved_room_type']}, Bookings: {row['booking_count']}, "
                results += f"Confirmed: {row['confirmed_bookings']}, Avg Rate: ${row['avg_daily_rate']:.2f}\n"
            
            return results
        
        # Check for country distribution queries
        elif any(word in query.lower() for word in ["distribution", "breakdown"]) and any(word in query.lower() for word in ["country", "countries", "geographic", "location", "region"]):
            sql = """
            SELECT 
                country,
                COUNT(*) as total_bookings,
                SUM(CASE WHEN is_canceled = 0 THEN 1 ELSE 0 END) as confirmed_bookings,
                SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
                SUM(revenue) as total_revenue
            FROM 
                bookings_table
            GROUP BY 
                country
            HAVING 
                total_bookings > 5
            ORDER BY 
                total_bookings DESC
            LIMIT 20
            """
            
            df = pd.read_sql_query(sql, self.conn)
            
            # Calculate total bookings across all countries in the results
            total_bookings = df['total_bookings'].sum()
            
            results = "SQL Results for Country Distribution Query:\n"
            results += f"Distribution of bookings across {len(df)} countries (showing top 20):\n\n"
            
            for _, row in df.iterrows():
                percentage = (row['total_bookings'] / total_bookings) * 100
                results += f"Country: {row['country']}, Bookings: {row['total_bookings']} ({percentage:.1f}% of total), "
                results += f"Confirmed: {row['confirmed_bookings']}, Canceled: {row['canceled_bookings']}, "
                results += f"Revenue: ${row['total_revenue']:.2f}\n"
            
            return results
        
        # Check for stay duration comparison queries
        elif "stay" in query.lower() and "hotel" in query.lower() and any(word in query.lower() for word in ["long", "duration", "length"]):
            sql = """
            SELECT 
                hotel,
                COUNT(*) as booking_count,
                AVG(total_stay_duration) as avg_stay_duration,
                MIN(total_stay_duration) as min_stay,
                MAX(total_stay_duration) as max_stay
            FROM 
                bookings_table
            WHERE 
                is_canceled = 0
            GROUP BY 
                hotel
            """
            
            df = pd.read_sql_query(sql, self.conn)
            
            results = "SQL Results for Stay Duration Query:\n"
            
            if len(df) > 1:
                for _, row in df.iterrows():
                    results += f"Hotel Type: {row['hotel']}\n"
                    results += f"  Average Stay Duration: {row['avg_stay_duration']:.1f} days\n"
                    results += f"  Minimum Stay: {row['min_stay']} days\n"
                    results += f"  Maximum Stay: {row['max_stay']} days\n"
                    results += f"  Based on {row['booking_count']} bookings\n\n"
            else:
                # If only one hotel type exists in the data
                row = df.iloc[0]
                results += f"Data only available for {row['hotel']}:\n"
                results += f"  Average Stay Duration: {row['avg_stay_duration']:.1f} days\n"
                results += f"  Minimum Stay: {row['min_stay']} days\n"
                results += f"  Maximum Stay: {row['max_stay']} days\n"
                results += f"  Based on {row['booking_count']} bookings\n\n"
                results += "Note: No data is available for other hotel types in the dataset.\n"
            
            return results
            
        # Check for lead time queries
        elif "lead time" in query.lower() or (("lead" in query.lower() or "advance" in query.lower()) and "booking" in query.lower()):
            # Extract hotel type if present
            hotel_type = None
            if "resort" in query.lower():
                hotel_type = "Resort Hotel"
            elif "city" in query.lower():
                hotel_type = "City Hotel"
            
            sql = """
            SELECT 
                hotel,
                COUNT(*) as booking_count,
                AVG(lead_time) as avg_lead_time,
                MIN(lead_time) as min_lead_time,
                MAX(lead_time) as max_lead_time,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY lead_time) as p25_lead_time,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time) as median_lead_time,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY lead_time) as p75_lead_time
            FROM 
                bookings_table
            """
            
            if hotel_type:
                sql += f" WHERE hotel = '{hotel_type}'"
                
            sql += " GROUP BY hotel"
            
            try:
                df = pd.read_sql_query(sql, self.conn)
            except:
                # If percentile functions are not supported, use a simpler query
                sql = """
                SELECT 
                    hotel,
                    COUNT(*) as booking_count,
                    AVG(lead_time) as avg_lead_time,
                    MIN(lead_time) as min_lead_time,
                    MAX(lead_time) as max_lead_time
                FROM 
                    bookings_table
                """
                
                if hotel_type:
                    sql += f" WHERE hotel = '{hotel_type}'"
                    
                sql += " GROUP BY hotel"
                df = pd.read_sql_query(sql, self.conn)
            
            results = "SQL Results for Lead Time Query:\n"
            
            for _, row in df.iterrows():
                results += f"Hotel Type: {row['hotel']}\n"
                results += f"  Average Lead Time: {row['avg_lead_time']:.1f} days\n"
                results += f"  Minimum Lead Time: {row['min_lead_time']} days\n"
                results += f"  Maximum Lead Time: {row['max_lead_time']} days\n"
                
                if 'median_lead_time' in row:
                    results += f"  Median Lead Time: {row['median_lead_time']:.1f} days\n"
                    results += f"  25th Percentile: {row['p25_lead_time']:.1f} days\n"
                    results += f"  75th Percentile: {row['p75_lead_time']:.1f} days\n"
                
                results += f"  Based on {row['booking_count']} bookings\n\n"
            
            return results
        
        return None
    
    def answer_question(self, question):
        """
        Answer a question using RAG with Google Gemini
        
        Args:
            question: User's question as a string
            
        Returns:
            Answer as a string
        """
        start_time = time.time()
        
        # Step 1: Retrieve relevant context from vector database
        print("Retrieving context from vector database...")
        context = self.query_vector_db(question)
        
        # Step 2: Get any direct SQL results if applicable
        sql_results = self.get_sql_results(question)
        if sql_results:
            context.append(sql_results)
        
        # Step 3: Format the prompt for Gemini
        if context:
            context_text = "\n\n".join(context)
        else:
            context_text = "No specific context found in the database."
        
        prompt = f"""
        You are a hotel booking analytics assistant. Answer the following question based on the provided context.
        Only use information from the context provided. If you can't answer based on the context, acknowledge that.
        
        Context:
        {context_text}
        
        Question: {question}
        
        Answer:
        """
        
        # Step 4: Generate response from Gemini
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
        except Exception as e:
            answer = f"Error generating response: {str(e)}"
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log the query
        self._log_query(question, answer, execution_time)
        
        return {
            "question": question,
            "answer": answer,
            "context_used": context,
            "execution_time": execution_time
        }
    
    def update_vector_db(self):
        """
        Update vector database with any new data
        """
        # Get last ID in vector DB
        try:
            last_id = max([int(id.split('_')[1]) for id in self.bookings_collection.get()['ids'] if id.startswith('booking_')])
        except:
            last_id = 0
        
        # Check if new data exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT MAX(rowid) FROM bookings_table")
        max_id = cursor.fetchone()[0]
        
        if max_id > last_id:
            print(f"Found {max_id - last_id} new records. Updating vector database...")
            
            # Get new data
            query = f"""
            SELECT 
                rowid as booking_id, *
            FROM 
                bookings_table
            WHERE 
                rowid > {last_id}
            """
            
            df = pd.read_sql_query(query, self.conn)
            
            # Create text representation of each booking
            texts = []
            ids = []
            metadatas = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing new bookings"):
                # Create a text representation that captures key information
                text = f"""
                Booking ID: {row['booking_id']}
                Hotel: {row['hotel']}
                Status: {"Canceled" if row['is_canceled'] == 1 else "Confirmed"}
                Country: {row['country']}
                Arrival Date: {row['arrival_date_year']}-{row['arrival_date_month']}
                Lead Time: {row['lead_time']} days
                Room Type: {row['reserved_room_type']}
                Daily Rate: ${row['adr']}
                Total Revenue: ${row['revenue']}
                Stay Duration: {row['total_stay_duration']} days
                Market Segment: {row['market_segment']}
                """
                
                # Clean and normalize text
                text = text.strip()
                
                texts.append(text)
                ids.append(f"booking_{row['booking_id']}")
                
                # Add metadata for more efficient filtering
                metadatas.append({
                    "hotel": row['hotel'],
                    "year": int(row['arrival_date_year']),
                    "month": row['arrival_date_month'],
                    "country": row['country'],
                    "is_canceled": bool(row['is_canceled']),
                    "revenue": float(row['revenue']),
                    "adr": float(row['adr'])
                })
            
            # Embed texts with progress bar
            embeddings = []
            for text in tqdm(texts, desc="Embedding new data"):
                embedding = self.embedding_model.encode(text)
                embeddings.append(embedding.tolist())
            
            # Add to collection
            self.bookings_collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            print(f"Successfully updated vector database with {len(texts)} new records.")
            
            # Update insights
            print("Updating insights...")
            self.analytics.generate_all_insights()
            self.embed_insights()
        else:
            print("No new data found. Vector database is up to date.")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
        self.analytics.close()

    
# When run as a script, set up the vector database
if __name__ == '__main__':
    import argparse
    
    # Make sure environment variables are loaded
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='RAG-based QA system for hotel bookings')
    parser.add_argument('--db_path', type=str, default='data/hotel_bookings.db', help='Path to SQLite database')
    parser.add_argument('--vector_db_path', type=str, default='data/vector_db', help='Path to store vector database')
    parser.add_argument('--api_key', type=str, default=None, help='Google API key')
    parser.add_argument('--setup_only', action='store_true', help='Only set up the vector database, don\'t run the demo')
    parser.add_argument('--update', action='store_true', help='Update vector database with new data')
    
    args = parser.parse_args()
    
    # Check for API key in environment if not provided
    if args.api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None or api_key == "your_api_key_goes_here":
            print("Error: Google API key is required. Please provide it or set GOOGLE_API_KEY environment variable.")
            print("Add your API key to the .env file or pass it with --api_key parameter.")
            exit(1)
    else:
        api_key = args.api_key
    
    # Print confirmation that API key was loaded
    print(f"Using Google API key: {api_key[:5]}...{api_key[-5:]}")
    
    # Initialize RAG system
    print("Initializing BookingRAG...")
    rag = BookingRAG(
        db_path=args.db_path,
        vector_db_path=args.vector_db_path,
        google_api_key=api_key
    )
    
    # Check if vector database exists and is populated
    try:
        bookings_count = len(rag.bookings_collection.get()['ids'])
        insights_count = len(rag.insights_collection.get()['ids'])
        if bookings_count == 0 or insights_count == 0:
            print("Vector database is empty. Setting up...")
            rag.embed_bookings_data()
            rag.embed_insights()
        elif args.update:
            print("Updating vector database...")
            rag.update_vector_db()
        else:
            print(f"Vector database already exists with {bookings_count} booking records " +
                  f"and {insights_count} insight records.")
    except Exception as e:
        print(f"Error checking vector database: {str(e)}")
        print("Creating new vector database...")
        rag.embed_bookings_data()
        rag.embed_insights()
    
    # Run a simple demo if not setup_only
    if not args.setup_only:
        print("\nDemo mode activated. Type 'exit' to quit.")
        print("Example questions:")
        print("- Show me total revenue for July 2017.")
        print("- Which locations had the highest booking cancellations?")
        print("- What is the average price of a hotel booking?")
        
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'exit':
                break
            
            print("Thinking...")
            response = rag.answer_question(question)
            
            print("\nAnswer:")
            print(response["answer"])
            print(f"\nResponse time: {response['execution_time']:.2f} seconds")
    
    rag.close()
    print("Done!") 