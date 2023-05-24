import numpy as np
import pandas as pd
import math
import pyodbc
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve, auc
from datetime import datetime, timedelta
from sqlalchemy import create_engine

server_name = 'parkpal-server.database.windows.net'
database_name = 'ParkingDb'
username = 'parkpal-admin'
password = 'Password123'

connection_string = 'Driver={ODBC Driver 17 for SQL Server};Server=' + server_name + ';Database=' + database_name + ';UID=' + username + ';PWD=' + password + ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'


def calculate_parking_charge():
    # Connect to the database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    # SQL query to fetch necessary data
    select_sql = '''
        SELECT TOP 1 id, temp, time_entry, DATEDIFF(hour, time_entry, GETDATE()) as time_diff
        FROM parking_table
        WHERE time_exit IS NOT NULL
        ORDER BY time_entry DESC
    '''

    cursor.execute(select_sql)
    record = cursor.fetchone()

    # Check if a record was found
    if record is not None:
        # Unpack record values
        id, temp, time_entry, time_diff = record
        day_of_week = time_entry.strftime('%A').lower()

        # Set the base hourly rate and 8+ hour rate depending on the day and temperature
        weekday_rates = (2.50, 10) if temp <= 20 else (3, 12)
        weekend_rates = (4.50, 14) if temp <= 20 else (5.50, 16)

        hourly_rate, day_rate = weekday_rates if day_of_week in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday'] else weekend_rates

        # Calculate the charge
        charge = day_rate if time_diff >= 8 else hourly_rate * math.ceil(time_diff)

        # Update the charge in the charges_table
        update_charge_sql = "UPDATE charges_table SET charge = ? WHERE id = ?"
        cursor.execute(update_charge_sql, (charge, id))

        # Commit the transaction
        conn.commit()

    else:
        print("No records found.")

    # Close the cursor
    cursor.close()

# Call the function
calculate_parking_charge()



##################################

def ML():

    # Execute a SQL query to select all rows from a table
    query = "SELECT * FROM parking_test"
    df = pd.read_sql_query(query, conn)
    print(df.head(1000))
    # Creating Features and Target Variable
    X = df[['time_diff', 'temp', 'humid']]
    y = df['is_parked']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Making Predictions on Test Data
    predictions = model.predict(X_test)

    # Evaluating Model Performance
    score = model.score(X_test, y_test)
    print("Model Score:", score)

    # Prediction 1 - Will a customer spend more than 4 hours on a weekday?
    # Creating Features and Target Variable
    df['weekday'] = pd.to_datetime(df['time_entry']).dt.weekday # Extracting weekday from time_entry
    X = df[['time_diff', 'temp', 'humid', 'weekday']]
    y = (df['time_diff'] >= 4) & (df['weekday'] < 5) # True if time_diff >= 4 and weekday is a weekday (0-4)

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Making Predictions on Test Data
    predictions = model.predict(X_test)

    # Evaluating Model Performance
    score = model.score(X_test, y_test)
    print("Model Score:", score)

    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Likeliness of a customer spending more than 4 hours on a weekday')
    plt.show()

    # Prediction 2 - Will a customer spend more than 4 hours on a weekend?
    # Creating Features and Target Variable
    df['is_weekend'] = df['time_entry'].dt.dayofweek >= 5
    X = df[['time_diff', 'temp', 'humid', 'is_weekend']]
    y = (df['time_diff'] > 4) & (df['is_weekend'])

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Logistic Regression Model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Making Predictions on Test Data
    predictions = model.predict(X_test)

    # Evaluating Model Performance
    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy:", accuracy)

    # Creating Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Likeliness of a customer spending more than 4 hours on a weekend')
    plt.show()

    # Visualise peak usage of the car park
    # Convert time columns to datetime format
    df['time_entry'] = pd.to_datetime(df['time_entry'])
    df['time_exit'] = pd.to_datetime(df['time_exit'])

    # Create new columns for day of the week and hour of the day
    df['day_of_week'] = df['time_entry'].dt.day_name()
    df['hour_of_entry'] = df['time_entry'].dt.hour
    df['hour_of_exit'] = df['time_exit'].dt.hour

    # Create pivot table to count entries for each day of the week and hour of the day
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hour_range = range(0, 24)
    pivot_table = pd.pivot_table(df, values='id', index='hour_of_entry', columns='day_of_week', aggfunc='count').reindex(columns=days_of_week, index=hour_range)
    # Add values from time_exit to pivot table
    pivot_table = pivot_table.add(pd.pivot_table(df, values='id', index='hour_of_exit', columns='day_of_week', aggfunc='count').reindex(columns=days_of_week, index=hour_range), fill_value=0)
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='g')
    plt.title('Peak Times for Each Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Hour of Day')
    plt.show()

    # Set price based on peak usage
    # Convert time_entry and time_exit columns to datetime objects
    df['time_entry'] = pd.to_datetime(df['time_entry'])
    df['time_exit'] = pd.to_datetime(df['time_exit'])

    # Extract day of the week from time_entry
    df['day_of_week'] = df['time_entry'].dt.dayofweek
    df['day_entry'] = df['time_entry'].dt.day_name()
    df['day_exit'] = df['time_exit'].dt.day_name()

    # Create a new column to indicate the parking duration in hours
    df['duration'] = (df['time_exit'] - df['time_entry']) / timedelta(hours=1)


    # Define a function to categorize the pricing based on parking duration and day of week
    def categorize_price(row):
        if row['duration'] >= 1:
            if row['day_entry'] == row['day_exit']:
                if row['time_entry'].hour >= 10 and row['time_entry'].hour <= 20:
                    ticket_type = 'Peak Hours (£7.50)'
                    price_change = 0.75
                else:
                    ticket_type = 'Off Peak Hours (£6)'
                    price_change = 0.25
            else:
                ticket_type = '24 Hour Pass (£12)'
                price_change = 1.0

            base_price = 4.0  # Default price
            if ticket_type == 'Peak Hours (£7.50)' or ticket_type == 'Off Peak Hours (£6)':
                base_price += price_change * base_price

            # Update price_table
            conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                                'Server=parkpal-server.database.windows.net;'
                                'Database=ParkingDb;'
                                'uid=parkpal-admin;'
                                'pwd=Password123')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO price_test (price) VALUES (?)", (base_price,))
            conn.commit()
            conn.close()

            return base_price


    # Categorize the pricing for each row
    df['pricing'] = df.apply(categorize_price, axis=1)

    # Drop rows with missing values in the 'pricing' column
    # Group the data by ticket type
    groups = df.groupby('pricing')

    # Plot a histogram for each group
    for name, group in groups:
        plt.hist(group['pricing'], bins=np.arange(3)-0.5, rwidth=0.8, align='mid', label=name)

    plt.xticks(rotation=45)
    plt.title('Distribution of Pricing Categories by Ticket Type')
    plt.xlabel('Pricing Category')
    plt.ylabel('Number of Cars')
    plt.legend()
    plt.show()

    # Prediction 3 - Is the car park likely to be busy based on temp
    # Define the features (temperature) and the target variable (busy or not)
    X = df['temp'].values.reshape(-1, 1)
    y = (df['time_diff'] > 5).astype(int)

    # Split the data into a training set and a testing set
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Train a logistic regression model on the training set
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluate the performance of the model using a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g')
    plt.title('Is the car park likely to be busy based on temp')
    plt.show()


