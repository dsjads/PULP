package main;
class Account {

	final static int DAILY_LIMIT = -1000;
	
	int withdraw = 0;


	boolean update(int x) {
		int newWithdraw = withdraw;
		if (x < 0)  {
			newWithdraw += x;
			if (newWithdraw < DAILY_LIMIT) 
				return false;
		}
		if (!original(x))
			return false;
		withdraw = newWithdraw;
		return true;
	}
	

	boolean undoUpdate(int x) {
		int newWithdraw = withdraw;
		if (x < 0)  {
			newWithdraw -= x;
			if (newWithdraw < DAILY_LIMIT) 
				return false;
		}
		if (!original(x))
			return false;
		withdraw = newWithdraw;
		return true;
	}
		
}
