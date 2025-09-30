package javachat.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JOptionPane;

import javachat.JavaChat;
import javachat.network.Server;

/**
 * Information about, who is online.
 */
public class ChatWindow {
	public String DialogOnline;

	JButton onlineButton = new JButton();

	void initComponents() {
		onlineButton.setText("Clients Online");

		onlineButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				getClientsNames();
				JOptionPane.showMessageDialog(null, DialogOnline, "Clients Online", JOptionPane.PLAIN_MESSAGE);
			}
		});

		botHorizontal.addComponent(onlineButton);
		botVertical.addComponent(onlineButton);
		
		onlineButton.setEnabled(false);
		onlineButton.setVisible(false);

		original();
	}

	private void getClientsNames() {
		javachat.JavaChat.refreshList();
		String sb = javachat.network.Server.sb;
		sb = sb.replaceAll("\\s", "\n");
		DialogOnline = "Users: \n" + sb;
	}

	private void jToggleButtonOnlineActionPerformed(java.awt.event.ActionEvent evt) {
		original(evt);
		if (jToggleButtonOnline.isSelected()) {
			onlineButton.setEnabled(true);
			if (jRadioButtonServer.isSelected()) {
				onlineButton.setVisible(true);
			}
		} else {
			onlineButton.setEnabled(false);
		}

	}

}