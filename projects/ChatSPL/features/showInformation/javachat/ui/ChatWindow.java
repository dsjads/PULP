package javachat.ui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JOptionPane;

/**
 * Informations over the Server.
 */
public class ChatWindow {
	public String Host = "Host: " + javachat.network.util.IPUtil.getHostName();
	public String InternalIPAddress = "Internal (local) clients should use the address: "
			+ javachat.network.util.IPUtil.getInternalIPAddress();
	public String ExternalIPAddress = "External (internet) clients should use the address: "
			+ javachat.network.util.IPUtil.getExternalIPAddress();
	public String Dialog = Host + "\n" + InternalIPAddress + "\n" + ExternalIPAddress;

	JButton informationButton = new JButton();

	void initComponents() {
		informationButton.setText("Server Information");

		informationButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				JOptionPane.showMessageDialog(null, Dialog, "Server Information", JOptionPane.PLAIN_MESSAGE);
			}
		});

		botHorizontal.addComponent(informationButton);
		botVertical.addComponent(informationButton);

		informationButton.setEnabled(false);

		original();
	}

	private void jToggleButtonOnlineActionPerformed(java.awt.event.ActionEvent evt) {
		original(evt);
		if (jToggleButtonOnline.isSelected()) {
			informationButton.setEnabled(true);
		} else {
			informationButton.setEnabled(false);
		}

	}

}