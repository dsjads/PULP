// non-grammar constraints
Notepad : [FullStyled] [Print] [Format] [Find] [Clipboard] [Undo] [File] [MenuBar] [ToolBar] Text Base :: _Notepad ;

Format : FormatRaw
	| FormatStyled ;

Text : TextRaw
	| TextStyled ;

%%

FormatRaw implies TextRaw ;
FormatStyled implies TextStyled ;

