GPL : [MainGpl] :: _GPL ;

MainGpl : HiddenGtp TestProg Alg+ Src HiddenWgt Wgt Gtp Implementation Base :: _MainGpl ;

HiddenGtp : DirectedWithEdges
	| DirectedWithNeighbors
	| DirectedOnlyVertices
	| UndirectedWithEdges
	| UndirectedWithNeighbors
	| UndirectedOnlyVertices ;

Alg : Number
	| Connected
	| StronglyConnected Transpose :: StrongC
	| Cycle
	| MSTPrim
	| MSTKruskal ;

Src : BFS
	| DFS ;

HiddenWgt : WeightOptions :: _HiddenWgt ;

WeightOptions : [WeightedWithEdges] [WeightedWithNeighbors] [WeightedOnlyVertices] :: _WeightOptions ;

Wgt : Weighted
	| Unweighted ;

Gtp : Directed
	| Undirected ;

Implementation : OnlyVertices
	| WithNeighbors
	| WithEdges ;

%%

Number implies Gtp and Src ;
Connected implies Undirected and Src ;
StrongC implies Directed and DFS ;
Cycle implies Gtp and DFS ;
MSTKruskal or MSTPrim implies Undirected and Weighted ;
MSTKruskal or MSTPrim implies not (MSTKruskal and MSTPrim) ;
MSTKruskal implies WithEdges ;
OnlyVertices and Weighted iff WeightedOnlyVertices ;
WithNeighbors and Weighted iff WeightedWithNeighbors ;
WithEdges and Weighted iff WeightedWithEdges ;
OnlyVertices and Directed iff DirectedOnlyVertices ;
WithNeighbors and Directed iff DirectedWithNeighbors ;
WithEdges and Directed iff DirectedWithEdges ;
OnlyVertices and Undirected iff UndirectedOnlyVertices ;
WithNeighbors and Undirected iff UndirectedWithNeighbors ;
WithEdges and Undirected iff UndirectedWithEdges ;

##

HiddenGtp { hidden } 
HiddenWgt { hidden } 
