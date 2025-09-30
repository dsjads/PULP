SnakeFOP : [Points] Entities Gamefield Level+ :: _SnakeFOP ;

Points : [Highscore] :: _Points ;

Entities : Snake Enemies+ :: _Entities ;

Snake : Automatic
	| Manual ;

Enemies : Bug
	| Centipede
	| Fly
	| Mouse
	| Slug ;

Gamefield : Bordered
	| Torus ;

Level : L1
	| L2
	| L3
	| L4 ;

