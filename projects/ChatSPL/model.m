chat_spl : Base :: _chat_spl ;

Base : [Client] [Server] [GUI] [Messaging] :: _Base ;

Client : [MessageAPI] [Customization] :: _Client ;

MessageAPI : [deleteMessage] :: _MessageAPI ;

Customization : [setNickname] [setAbout] :: _Customization ;

Server : [Filesharing] :: _Server ;

GUI : changeFont* [searchMessage] [clearChat] [changeChatBgColor] [showServer] :: _GUI ;

changeFont : changeSize
	| changeStyle
	| changeType ;

showServer : [showOnline] [showInformation] :: _showServer ;

Messaging : [DeliveryStatus] :: _Messaging ;

%%

showOnline implies setAbout ;

