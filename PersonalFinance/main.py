from user_auth import register, login
from finance_tracker import (
    initialize_data, 
    add_transaction, 
    view_transactions, 
    visualize_expenses, 
    budget_report, 
    add_category, 
    view_categories, 
    filter_transactions, 
    monthly_summary, 
    currency_conversion
)

def main():
    initialize_data()
    
    while True:
        print("\n1. Register")
        print("2. Login")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            username = input("Enter username: ")
            password = input("Enter password: ")
            if register(username, password):
                print("Registration successful!")
            else:
                print("Username already exists.")
        
        elif choice == '2':
            username = input("Enter username: ")
            password = input("Enter password: ")
            if login(username, password):
                print("Login successful!")
                budget = float(input("Enter your monthly budget: "))
                
                while True:
                    print("\n1. Add Transaction")
                    print("2. View Transactions")
                    print("3. Visualize Expenses")
                    print("4. Budget Report")
                    print("5. Add Category")
                    print("6. View Categories")
                    print("7. Filter Transactions")
                    print("8. Monthly Summary")
                    print("9. Currency Conversion")
                    print("10. Logout")
                    option = input("Choose an option: ")

                    if option == '1':
                        date = input("Enter transaction date (YYYY-MM-DD): ")
                        category = input("Enter transaction category: ")
                        amount = float(input("Enter transaction amount: "))
                        trans_type = input("Enter transaction type (Income/Expense): ")
                        add_transaction(date, category, amount, trans_type)
                    
                    elif option == '2':
                        view_transactions()
                    
                    elif option == '3':
                        visualize_expenses()
                    
                    elif option == '4':
                        budget_report(budget)
                    
                    elif option == '5':
                        category = input("Enter category name to add: ")
                        add_category(category)
                    
                    elif option == '6':
                        view_categories()
                    
                    elif option == '7':
                        start_date = input("Enter start date (YYYY-MM-DD) or leave empty: ")
                        end_date = input("Enter end date (YYYY-MM-DD) or leave empty: ")
                        category = input("Enter category to filter by or leave empty: ")
                        filter_transactions(start_date if start_date else None, end_date if end_date else None, category if category else None)

                    elif option == '8':
                        monthly_summary()
                    
                    elif option == '9':
                        amount = float(input("Enter amount: "))
                        from_currency = input("Enter from currency (e.g., USD): ")
                        to_currency = input("Enter to currency (e.g., EUR): ")
                        converted_amount = currency_conversion(amount, from_currency, to_currency)
                        print(f"{amount} {from_currency} is {converted_amount:.2f} {to_currency}.")
                    
                    elif option == '10':
                        print("Logging out...")
                        break
            
            else:
                print("Login failed. Check your username and password.")

        elif choice == '3':
            print("Exiting the application.")
            break

if __name__ == '__main__':
    main()
