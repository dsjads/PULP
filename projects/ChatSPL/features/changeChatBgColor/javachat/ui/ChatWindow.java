package javachat.ui;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JColorChooser;

public class ChatWindow {

	void initComponents() {

		JButton bgColorButton = new JButton();
		bgColorButton.setText("Chat BG-Color");

		bgColorButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				Color initialBackground = jTextAreaChat.getBackground();
				Color background = JColorChooser.showDialog(null, "Change Button Background", initialBackground);
				if (background != null) {
					jTextAreaChat.setBackground(background);
				}
			}
		});

		midHorizontal.addComponent(bgColorButton);
		midVertical.addComponent(bgColorButton);

		original();
	}
}