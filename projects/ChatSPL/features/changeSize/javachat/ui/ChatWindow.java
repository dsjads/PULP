package javachat.ui;

import java.awt.Font;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;

public class ChatWindow {

	void initComponents() {

		JComboBox fontsizeComboBox = new JComboBox(this.getFontsizeComboBoxList());

		fontsizeComboBox.addActionListener(new ActionListener() {

			public void actionPerformed(java.awt.event.ActionEvent evt) {
				// Change Fonttype to chosen font
				JComboBox cb = (JComboBox) evt.getSource();
				int fontSize = Integer.parseInt((String) cb.getSelectedItem());
				String fontName = jTextAreaChat.getFont().getName();
				int fontType = jTextAreaChat.getFont().getStyle();

				// get the correct TextField
				jTextAreaChat.setFont(new Font(fontName, fontType, fontSize));
			}

		});

		midHorizontal.addComponent(fontsizeComboBox);
		midVertical.addComponent(fontsizeComboBox);

		original();
	}

	private String[] getFontsizeComboBoxList() {
		String num[] = new String[100];
		for (int i = 0; i < 100; i++) {
			num[i] = Integer.toString(i);
		}
		return num;
	}
}