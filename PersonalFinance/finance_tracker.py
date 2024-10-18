import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import requests

DATA_FILE = 'data.csv'
CATEGORY_FILE = 'categories.json'

def initialize_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=['Date', 'Category', 'Amount', 'Type'])
        df.to_csv(DATA_FILE, index=False)

    if not os.path.exists(CATEGORY_FILE):
        with open(CATEGORY_FILE, 'w') as file:
            json.dump([], file)

def add_transaction(date, category, amount, trans_type):
    df = pd.read_csv(DATA_FILE)
    df = df.append({'Date': date, 'Category': category, 'Amount': amount, 'Type': trans_type}, ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

def view_transactions():
    df = pd.read_csv(DATA_FILE)
    print(df)

def visualize_expenses():
    df = pd.read_csv(DATA_FILE)
    expense_df = df[df['Type'] == 'Expense']
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Category', y='Amount', data=expense_df.groupby('Category').sum().reset_index())
    plt.title('Expenses by Category')
    plt.xticks(rotation=45)
    plt.ylabel('Amount')
    plt.show()

def budget_report(budget):
    df = pd.read_csv(DATA_FILE)
    total_expenses = df[df['Type'] == 'Expense']['Amount'].sum()
    remaining_budget = budget - total_expenses
    print(f'Total Expenses: ${total_expenses}')
    print(f'Remaining Budget: ${remaining_budget}')

def add_category(category):
    with open(CATEGORY_FILE, 'r') as file:
        categories = json.load(file)
    
    if category not in categories:
        categories.append(category)
        with open(CATEGORY_FILE, 'w') as file:
            json.dump(categories, file)
        print(f'Category "{category}" added.')
    else:
        print(f'Category "{category}" already exists.')

def view_categories():
    with open(CATEGORY_FILE, 'r') as file:
        categories = json.load(file)
    print("Available categories:", categories)

def filter_transactions(start_date=None, end_date=None, category=None):
    df = pd.read_csv(DATA_FILE)
    
    if start_date:
        df = df[df['Date'] >= start_date]
    if end_date:
        df = df[df['Date'] <= end_date]
    if category:
        df = df[df['Category'] == category]
    
    print(df)

def monthly_summary():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')
    summary = df.groupby(['Month', 'Type'])['Amount'].sum().unstack().fillna(0)
    print(summary)

def currency_conversion(amount, from_currency, to_currency):
    api_url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
    response = requests.get(api_url)
    data = response.json()
    conversion_rate = data['rates'][to_currency]
    return amount * conversion_rate
