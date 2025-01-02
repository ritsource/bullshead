from typing import Union

class Wallet:
    def __init__(self, initial_balance: Union[int, float] = 0):
        self.balance = initial_balance

    def add_balance(self, amount: Union[int, float]) -> Union[int, float]:
        if amount < 0:
            raise ValueError("Amount to add must be positive")
        self.balance += amount
        return self.balance

    def sub_balance(self, amount: Union[int, float]) -> Union[int, float]:
        if amount < 0:
            raise ValueError("Amount to subtract must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        return self.balance

    def get_balance(self) -> Union[int, float]:
        return self.balance

    def update_balance(self, new_balance: Union[int, float]) -> Union[int, float]:
        if new_balance < 0:
            raise ValueError("Balance cannot be negative")
        self.balance = new_balance
        return self.balance
