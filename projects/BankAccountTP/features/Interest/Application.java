package main;
class Application {

	void nextDay() {
		original();
		account.interest += account.calculateInterest();
	}

	void nextYear() {
		original();
		account.balance += account.interest;
		account.interest = 0;
	}

}
