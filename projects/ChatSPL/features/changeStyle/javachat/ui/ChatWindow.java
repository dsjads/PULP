package javachat.ui;

import java.awt.Font;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class ChatWindow {
	
	void initComponents(){
		
		JButton boldButton = new JButton();
		JButton italicButton = new JButton();
		boldButton.setText("Bold");		
		italicButton.setText("Italic");

		boldButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				int fontSize = jTextAreaChat.getFont().getSize();
				String fontName = jTextAreaChat.getFont().getName();
				
				if (!jTextAreaChat.getFont().isBold()){
					jTextAreaChat.setFont(new Font(fontName, Font.BOLD, fontSize));
				} else {
					jTextAreaChat.setFont(new Font(fontName, Font.PLAIN, fontSize));
				}				
			}
		});
		
		italicButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				int fontSize = jTextAreaChat.getFont().getSize();
				String fontName = jTextAreaChat.getFont().getName();
				
				if (!jTextAreaChat.getFont().isItalic()){
					jTextAreaChat.setFont(new Font(fontName, Font.ITALIC, fontSize));
				} else {
					jTextAreaChat.setFont(new Font(fontName, Font.PLAIN, fontSize));
				}			
			}
		});

		midHorizontal.addComponent(boldButton);
		midHorizontal.addComponent(italicButton);
		midVertical.addComponent(boldButton);
		midVertical.addComponent(italicButton);
		
		original();
	}
}