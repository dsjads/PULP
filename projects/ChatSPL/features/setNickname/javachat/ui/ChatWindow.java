package javachat.ui;

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JColorChooser;

import javachat.JavaChat;
import javachat.network.Client;

public class ChatWindow {
	
	public String name = "Unknown";

	void initComponents() {

		JButton nicknameButton = new JButton();
		nicknameButton.setText("Update Name");

		nicknameButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				UpdateNameAction(e);
			}
		});

		topHorizontal.addComponent(nicknameButton);
		topVertical.addComponent(nicknameButton);

		original();
	}
	
	private void UpdateNameAction(java.awt.event.ActionEvent evt) {
		Client client = JavaChat.getClient();
		String comp = jTextFieldName.getText();
		if (!name.equals(comp)){
			if (client != null) {
				name = jTextFieldName.getText();
				client.setName(name);
			}
		}
	}
}