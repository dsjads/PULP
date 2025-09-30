package javachat.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ChatWindow {

	void initComponents() {

		JButton clearButton = new JButton();
		clearButton.setText("Clear");

		clearButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				jTextAreaChat.setText("");
			}
		});

		midHorizontal.addComponent(clearButton);
		midVertical.addComponent(clearButton);

		original();
	}
}