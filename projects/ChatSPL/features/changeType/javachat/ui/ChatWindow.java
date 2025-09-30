package javachat.ui;

import java.awt.Font;
import java.awt.GraphicsEnvironment;
import java.awt.event.ActionListener;

import javax.swing.JComboBox;

public class ChatWindow {

	void initComponents() {

		JComboBox fonttypeComboBox = new JComboBox(this.getfonttypeComboBoxList());

		fonttypeComboBox.addActionListener(new ActionListener() {

			public void actionPerformed(java.awt.event.ActionEvent evt) {
				// Change Fonttype to chosen font
				JComboBox cb = (JComboBox) evt.getSource();
				String fontName = (String) cb.getSelectedItem();
				int fontSize = jTextAreaChat.getFont().getSize();
				int fontType = jTextAreaChat.getFont().getStyle();

				// get the correct TextField
				jTextAreaChat.setFont(new Font(fontName, fontType, fontSize));
			}

		});

		midHorizontal.addComponent(fonttypeComboBox);
		midVertical.addComponent(fonttypeComboBox);

		original();
	}

	private String[] getfonttypeComboBoxList() {
		return GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
	}
}