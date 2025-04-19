import random
import os

def calculate_sector_return(sector, investment):
    multipliers = {
        "Technology": random.uniform(-0.2, 0.20),
        "Agriculture": random.uniform(-0.1, 0.2),
        "Energy": random.uniform(-0.05, 0.15),
        "Health": random.uniform(-0.08, 0.12),
        "Services": random.uniform(-0.1, 0.15),
    }
    return investment * multipliers[sector]

def save_list_to_file(file_path, list_of_lists, delimiter=","):
    try:
        with open(file_path, 'w+') as file:
            for sublist in list_of_lists:
                line = delimiter.join(map(str, sublist))
                file.write(line + "\n")
        print(f"Data successfully saved to: {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

class Company:
    def __init__(self, name):
        self.name = name
        self.money = 1_000_000  # Initial capital
        self.reputation = 50    # Initial reputation
        self.sectors = {
            "Technology": 0,
            "Agriculture": 0,
            "Energy": 0,
            "Health": 0,
            "Services": 0
        }
        self.profit = 0
        self.turn = 1
        self.data_x = []
        self.data_y = []

    def show_status(self):
        print(f"\nYear: {2025 + self.turn - 1}")
        print(f"Company: {self.name}")
        print(f"Capital: ${self.money:,.2f}")
        print(f"Reputation: {self.reputation}%")
        print(f"Profit: {self.profit}")
        print("Investments by sector:")
        for sector, investment in self.sectors.items():
            print(f"  {sector}: ${investment:,.2f}")

    def invest(self):
        print("\nAvailable sectors for investment:")
        available_sectors = list(self.sectors.keys())
        for i, sector in enumerate(available_sectors, start=1):
            print(f"{i}. {sector}")

        choice = int(input("Choose a sector to invest in (number): ")) - 1

        self.data_y.append([choice])
        if 0 <= choice < len(available_sectors):
            chosen_sector = available_sectors[choice]
            percent = float(input(f"What percentage of capital do you want to invest in {chosen_sector}? "))

            amount = percent * self.money

            if amount > 0 and amount <= self.money:
                self.sectors[chosen_sector] += amount
                self.money -= amount
                self.reputation = min(100, self.reputation + 2)  # Gain reputation
                print(f"Investment of ${amount:,.2f} made in {chosen_sector}!")
            else:
                print("Invalid investment. Check available funds.")
        else:
            print("Invalid choice.")

    def next_turn(self):
        print("\nAdvancing to the next turn...")
        self.turn += 1
        turn_profit = 0
        print("\n--- Sector Returns ---")
        for sector, investment in self.sectors.items():
            return_ = calculate_sector_return(sector, investment)
            turn_profit += return_
            print(f"{sector}: return of ${return_:,.2f}")
        
        self.money += turn_profit
        self.profit += turn_profit

        # Fixed operational cost
        fixed_cost = 50000 + (self.turn * 2000)
        self.money = max(0, self.money - fixed_cost)
        self.profit -= fixed_cost

        print(f"\nOperational cost of the year: ${fixed_cost:,.2f}")
        print(f"Total annual profit: ${turn_profit - fixed_cost:,.2f}")

        # Random event
        if random.random() < 0.3:
            affected_sector = random.choice(list(self.sectors.keys()))
            impact = random.uniform(0.1, 0.5)
            loss = self.sectors[affected_sector] * impact
            self.money = max(0, self.money - loss)
            self.profit -= loss
            print(f"\n⚠️ Unexpected event! A crisis in the {affected_sector} sector caused a loss of ${loss:,.2f}.")

    def play(self):
        turns_with_no_money = 0
        turns_with_loss = 0

        for i in range(100):
            self.show_status()
            self.invest()
            
            values = [investment for investment in self.sectors.values()]
            values.append(self.reputation)
            self.data_x.append(values.copy())

            money_before = self.money
            profit_before = self.profit

            self.next_turn()

            # Check if out of money
            if self.money <= 0:
                turns_with_no_money += 1
            else:
                turns_with_no_money = 0

            # Check if profit decreased
            if self.profit < profit_before:
                turns_with_loss += 1
            else:
                turns_with_loss = 0

            # Critical reputation
            if self.reputation <= 5:
                print("\n❌ Your reputation has dropped too low. No one wants to do business with you.")
                print("💀 Game over!")
                break

            # Bankruptcy due to no money
            if turns_with_no_money >= 2:
                print("\n❌ Your company has been out of capital for too long.")
                print("💀 Bankruptcy declared. Game over!")
                break

            # Bankruptcy due to consecutive losses
            if turns_with_loss >= 3:
                print("\n❌ Your company had losses for 3 consecutive turns.")
                print("💀 Bankruptcy declared. Game over!")
                break
            
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(current_dir, 'datas')
        X_train_path = os.path.join(data_dir, 'x_train.dll')
        Y_train_path = os.path.join(data_dir, 'y_train.dll')

        # Save training data if the game lasted
        save_list_to_file(X_train_path, self.data_x)
        save_list_to_file(Y_train_path, self.data_y)

company_name = input("Enter your company name: ")
game = Company(company_name)
game.play()
