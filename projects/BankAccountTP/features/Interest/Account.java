package main;
class Account {

	final static int INTEREST_RATE = 2;

	int interest = 0;


	int calculateInterest() {
		return balance * INTEREST_RATE / 36500;
	}

}
