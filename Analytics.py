import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class BookingAnalytics:
    def __init__(self, db_path='data/hotel_bookings.db'):
        """Initialize the analytics engine with database path"""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.insights_table_name = 'precomputed_insights'
        self.query_history_table_name = 'query_history'
        
        # Connect to database and create tables if they don't exist
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection and create necessary tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Create table for precomputed insights if it doesn't exist
        self.cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.insights_table_name} (
            insight_id TEXT PRIMARY KEY,
            insight_type TEXT,
            insight_data TEXT,
            timestamp TEXT,
            metadata TEXT
        )
        ''')
        
        # Create table for query history if it doesn't exist (bonus)
        self.cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.query_history_table_name} (
            query_id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            timestamp TEXT,
            response TEXT,
            execution_time REAL
        )
        ''')
        
        self.conn.commit()
    
    def _store_insight(self, insight_id, insight_type, insight_data, metadata=None):
        """Store a precomputed insight in the database"""
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else '{}'
        insight_data_json = json.dumps(insight_data)
        
        # Check if insight exists, update if it does, insert if it doesn't
        self.cursor.execute(f'''
        SELECT insight_id FROM {self.insights_table_name} WHERE insight_id = ?
        ''', (insight_id,))
        
        if self.cursor.fetchone():
            self.cursor.execute(f'''
            UPDATE {self.insights_table_name}
            SET insight_data = ?, timestamp = ?, metadata = ?
            WHERE insight_id = ?
            ''', (insight_data_json, timestamp, metadata_json, insight_id))
        else:
            self.cursor.execute(f'''
            INSERT INTO {self.insights_table_name} (insight_id, insight_type, insight_data, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?)
            ''', (insight_id, insight_type, insight_data_json, timestamp, metadata_json))
        
        self.conn.commit()
    
    def _get_insight(self, insight_id):
        """Retrieve a precomputed insight from the database"""
        self.cursor.execute(f'''
        SELECT insight_data FROM {self.insights_table_name} WHERE insight_id = ?
        ''', (insight_id,))
        
        result = self.cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def _log_query(self, query_text, response, execution_time):
        """Log a query to the query history table (bonus)"""
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute(f'''
        INSERT INTO {self.query_history_table_name} (query_text, timestamp, response, execution_time)
        VALUES (?, ?, ?, ?)
        ''', (query_text, timestamp, json.dumps(response), execution_time))
        
        self.conn.commit()
    
    def generate_all_insights(self):
        """Generate and store all insights"""
        self.generate_revenue_trends()
        self.generate_cancellation_rate()
        self.generate_geographical_distribution()
        self.generate_lead_time_distribution()
        self.generate_additional_insights()
    
    def generate_revenue_trends(self):
        """Generate revenue trends over time"""
        # Query to aggregate revenue by year and month
        query = '''
        SELECT 
            arrival_date_year, 
            arrival_date_month_num, 
            SUM(revenue) as total_revenue,
            COUNT(*) as booking_count
        FROM 
            bookings_table
        WHERE 
            is_canceled = 0
        GROUP BY 
            arrival_date_year, arrival_date_month_num
        ORDER BY 
            arrival_date_year, arrival_date_month_num
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        # Create a proper date column for easier plotting
        df['date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                  df['arrival_date_month_num'].astype(str) + '-01')
        
        # Prepare data for storage
        trend_data = {
            'dates': df['date'].dt.strftime('%Y-%m').tolist(),
            'revenues': df['total_revenue'].tolist(),
            'booking_counts': df['booking_count'].tolist()
        }
        
        # Store the insight
        self._store_insight(
            insight_id='revenue_trends',
            insight_type='time_series',
            insight_data=trend_data,
            metadata={'description': 'Monthly revenue trends over time'}
        )
        
        return trend_data
    
    def generate_cancellation_rate(self):
        """Calculate cancellation rate as percentage of total bookings"""
        query = '''
        SELECT 
            hotel,
            SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
            COUNT(*) as total_bookings,
            (SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as cancellation_rate
        FROM 
            bookings_table
        GROUP BY 
            hotel
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        # Calculate overall cancellation rate
        total_query = '''
        SELECT 
            SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
            COUNT(*) as total_bookings,
            (SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as cancellation_rate
        FROM 
            bookings_table
        '''
        
        total_df = pd.read_sql_query(total_query, self.conn)
        
        # Prepare data for storage
        cancellation_data = {
            'by_hotel': {
                'hotels': df['hotel'].tolist(),
                'cancellation_rates': df['cancellation_rate'].tolist(),
                'canceled_bookings': df['canceled_bookings'].tolist(),
                'total_bookings': df['total_bookings'].tolist()
            },
            'overall': {
                'cancellation_rate': float(total_df['cancellation_rate'].iloc[0]),
                'canceled_bookings': int(total_df['canceled_bookings'].iloc[0]),
                'total_bookings': int(total_df['total_bookings'].iloc[0])
            }
        }
        
        # Store the insight
        self._store_insight(
            insight_id='cancellation_rate',
            insight_type='statistics',
            insight_data=cancellation_data,
            metadata={'description': 'Cancellation rates overall and by hotel type'}
        )
        
        return cancellation_data
    
    def generate_geographical_distribution(self):
        """Analyze geographical distribution of bookings"""
        query = '''
        SELECT 
            country, 
            COUNT(*) as booking_count,
            SUM(CASE WHEN is_canceled = 0 THEN 1 ELSE 0 END) as confirmed_bookings,
            SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
            SUM(CASE WHEN is_canceled = 0 THEN revenue ELSE 0 END) as total_revenue
        FROM 
            bookings_table
        GROUP BY 
            country
        ORDER BY 
            booking_count DESC
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        # Get top 20 countries by booking count
        top_countries = df.head(20)
        
        # Prepare data for storage
        geo_data = {
            'countries': df['country'].tolist(),
            'booking_counts': df['booking_count'].tolist(),
            'confirmed_bookings': df['confirmed_bookings'].tolist(),
            'canceled_bookings': df['canceled_bookings'].tolist(),
            'total_revenue': df['total_revenue'].tolist(),
            'top_20': {
                'countries': top_countries['country'].tolist(),
                'booking_counts': top_countries['booking_count'].tolist()
            }
        }
        
        # Store the insight
        self._store_insight(
            insight_id='geographical_distribution',
            insight_type='geographic',
            insight_data=geo_data,
            metadata={'description': 'Geographical distribution of bookings by country'}
        )
        
        return geo_data
    
    def generate_lead_time_distribution(self):
        """Analyze booking lead time distribution"""
        query = '''
        SELECT 
            lead_time,
            COUNT(*) as booking_count
        FROM 
            bookings_table
        WHERE
            is_canceled = 0
        GROUP BY 
            lead_time
        ORDER BY 
            lead_time
        '''
        
        df = pd.read_sql_query(query, self.conn)
        
        # Calculate percentiles for lead time
        lead_time_stats = {
            'min': int(df['lead_time'].min()),
            'max': int(df['lead_time'].max()),
            'mean': float(np.average(df['lead_time'], weights=df['booking_count'])),
            'median': float(np.median(np.repeat(df['lead_time'], df['booking_count']))),
            'percentiles': {
                '25': float(np.percentile(np.repeat(df['lead_time'], df['booking_count']), 25)),
                '50': float(np.percentile(np.repeat(df['lead_time'], df['booking_count']), 50)),
                '75': float(np.percentile(np.repeat(df['lead_time'], df['booking_count']), 75)),
                '90': float(np.percentile(np.repeat(df['lead_time'], df['booking_count']), 90))
            }
        }
        
        # Group lead times into bins for visualization
        bins = [0, 7, 30, 90, 180, 365, float('inf')]
        labels = ['0-7 days', '8-30 days', '31-90 days', '91-180 days', '181-365 days', '365+ days']
        
        df['lead_time_binned'] = pd.cut(df['lead_time'], bins=bins, labels=labels)
        binned_counts = df.groupby('lead_time_binned')['booking_count'].sum()
        
        # Prepare data for storage
        lead_time_data = {
            'statistics': lead_time_stats,
            'distribution': {
                'bins': labels,
                'counts': binned_counts.tolist()
            }
        }
        
        # Store the insight
        self._store_insight(
            insight_id='lead_time_distribution',
            insight_type='distribution',
            insight_data=lead_time_data,
            metadata={'description': 'Distribution of booking lead times'}
        )
        
        return lead_time_data
    
    def generate_additional_insights(self):
        """Generate additional useful insights"""
        # 1. Average daily rate (ADR) by hotel type and room type
        adr_query = '''
        SELECT 
            hotel,
            reserved_room_type,
            AVG(adr) as avg_daily_rate,
            COUNT(*) as booking_count
        FROM 
            bookings_table
        WHERE 
            is_canceled = 0
            AND adr > 0
        GROUP BY 
            hotel, reserved_room_type
        ORDER BY 
            hotel, avg_daily_rate DESC
        '''
        
        adr_df = pd.read_sql_query(adr_query, self.conn)
        
        # 2. Booking patterns by market segment
        market_query = '''
        SELECT 
            market_segment,
            COUNT(*) as booking_count,
            SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
            (SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) as cancellation_rate,
            AVG(adr) as avg_daily_rate,
            AVG(total_stay_duration) as avg_stay_duration
        FROM 
            bookings_table
        GROUP BY 
            market_segment
        ORDER BY 
            booking_count DESC
        '''
        
        market_df = pd.read_sql_query(market_query, self.conn)
        
        # 3. Booking distribution by day of week
        weekday_query = '''
        SELECT 
            arrival_date_day_of_month,
            COUNT(*) as booking_count
        FROM 
            bookings_table
        WHERE
            is_canceled = 0
        GROUP BY 
            arrival_date_day_of_month
        ORDER BY 
            arrival_date_day_of_month
        '''
        
        weekday_df = pd.read_sql_query(weekday_query, self.conn)
        
        # 4. Room type distribution analysis
        room_type_query = '''
        SELECT 
            reserved_room_type,
            COUNT(*) as booking_count,
            SUM(CASE WHEN is_canceled = 0 THEN 1 ELSE 0 END) as confirmed_bookings,
            SUM(CASE WHEN is_canceled = 1 THEN 1 ELSE 0 END) as canceled_bookings,
            AVG(adr) as avg_daily_rate,
            AVG(total_stay_duration) as avg_stay_duration
        FROM 
            bookings_table
        GROUP BY 
            reserved_room_type
        ORDER BY 
            booking_count DESC
        '''
        
        room_type_df = pd.read_sql_query(room_type_query, self.conn)
        
        # Prepare data for storage
        additional_insights = {
            'adr_by_room_type': {
                'hotel_types': adr_df['hotel'].tolist(),
                'room_types': adr_df['reserved_room_type'].tolist(),
                'avg_rates': adr_df['avg_daily_rate'].tolist(),
                'booking_counts': adr_df['booking_count'].tolist()
            },
            'market_segment_analysis': {
                'segments': market_df['market_segment'].tolist(),
                'booking_counts': market_df['booking_count'].tolist(),
                'cancellation_rates': market_df['cancellation_rate'].tolist(),
                'avg_rates': market_df['avg_daily_rate'].tolist(),
                'avg_stay_durations': market_df['avg_stay_duration'].tolist()
            },
            'day_of_month_distribution': {
                'days': weekday_df['arrival_date_day_of_month'].tolist(),
                'booking_counts': weekday_df['booking_count'].tolist()
            },
            'room_type_distribution': {
                'room_types': room_type_df['reserved_room_type'].tolist(),
                'booking_counts': room_type_df['booking_count'].tolist(),
                'confirmed_bookings': room_type_df['confirmed_bookings'].tolist(),
                'canceled_bookings': room_type_df['canceled_bookings'].tolist(),
                'avg_rates': room_type_df['avg_daily_rate'].tolist(),
                'avg_stay_durations': room_type_df['avg_stay_duration'].tolist()
            }
        }
        
        # Store the insight
        self._store_insight(
            insight_id='additional_insights',
            insight_type='composite',
            insight_data=additional_insights,
            metadata={'description': 'Additional booking insights and patterns'}
        )
        
        return additional_insights
    
    def get_all_insights(self):
        """Retrieve all precomputed insights"""
        insights = {}
        
        for insight_id in [
            'revenue_trends', 
            'cancellation_rate', 
            'geographical_distribution', 
            'lead_time_distribution',
            'additional_insights'
        ]:
            insights[insight_id] = self._get_insight(insight_id)
        
        return insights
    
    def visualize_insights(self, output_dir='visualizations'):
        """Generate visualizations from the insights"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Visualize revenue trends
        revenue_data = self._get_insight('revenue_trends')
        if revenue_data:
            plt.figure(figsize=(12, 6))
            plt.plot(revenue_data['dates'], revenue_data['revenues'], marker='o')
            plt.title('Revenue Trends Over Time')
            plt.xlabel('Month')
            plt.ylabel('Revenue')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/revenue_trends.png")
            plt.close()
        
        # Visualize cancellation rates by hotel
        cancellation_data = self._get_insight('cancellation_rate')
        if cancellation_data:
            plt.figure(figsize=(10, 6))
            hotels = cancellation_data['by_hotel']['hotels']
            rates = cancellation_data['by_hotel']['cancellation_rates']
            
            plt.bar(hotels, rates)
            plt.title('Cancellation Rates by Hotel Type')
            plt.xlabel('Hotel Type')
            plt.ylabel('Cancellation Rate (%)')
            plt.axhline(y=cancellation_data['overall']['cancellation_rate'], 
                        color='r', linestyle='--', label=f"Overall Rate: {cancellation_data['overall']['cancellation_rate']:.2f}%")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/cancellation_rates.png")
            plt.close()
        
        # Visualize geographical distribution (top 20 countries)
        geo_data = self._get_insight('geographical_distribution')
        if geo_data and 'top_20' in geo_data:
            plt.figure(figsize=(14, 8))
            countries = geo_data['top_20']['countries']
            bookings = geo_data['top_20']['booking_counts']
            
            plt.bar(countries, bookings)
            plt.title('Top 20 Countries by Booking Count')
            plt.xlabel('Country')
            plt.ylabel('Number of Bookings')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/geographical_distribution.png")
            plt.close()
        
        # Visualize lead time distribution
        lead_time_data = self._get_insight('lead_time_distribution')
        if lead_time_data and 'distribution' in lead_time_data:
            plt.figure(figsize=(12, 6))
            bins = lead_time_data['distribution']['bins']
            counts = lead_time_data['distribution']['counts']
            
            plt.bar(bins, counts)
            plt.title('Lead Time Distribution')
            plt.xlabel('Lead Time')
            plt.ylabel('Number of Bookings')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/lead_time_distribution.png")
            plt.close()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# When run as a script, generate all insights
if __name__ == '__main__':
    print("Initializing BookingAnalytics...")
    analytics = BookingAnalytics()
    
    print("Generating all insights...")
    analytics.generate_all_insights()
    
    print("Creating visualizations...")
    analytics.visualize_insights()
    
    print("Done! All insights generated and stored in the database.")
    analytics.close() 