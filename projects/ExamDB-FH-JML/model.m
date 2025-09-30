ExamDB : [BonusPoints] [BackOut] [Statistics] Derivatives :: _ExamDB ;

Derivatives : [BonusPointsBackOut] [BonusPointsStatistics] [BackOutStatistics] [BonusPointsBackOutStatistics] :: _Derivatives ;

%%

BonusPointsBackOut iff BonusPoints and BackOut ;
BonusPointsStatistics iff BonusPoints and Statistics ;
BackOutStatistics iff BackOut and Statistics ;
BonusPointsBackOutStatistics iff BonusPoints and BackOut and Statistics ;

##

Derivatives { hidden } 
