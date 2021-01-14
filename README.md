# 8-puzzle-Search-Algorithms
## Artificial Intelligence
### Implements Multiple Search Algorithms: 
* *Breadth First Search*
* *Depth Limited Search*
* *Iterative Deepening Search*
* *A\* search*

The A\* search is implemented in two ways with each having a different admissible heursitic function: **Misplaced Tiles** & **Manhattan Distance**. 

#### Sample Input and Output
~~~~~~~~~~~~~~~~~~~~
{cslinux2:~/4365.501/HW1} python3 homework1.py bfs \* 1 3 4 2 5 7 8 6
8 Puzzle
Breadth-First Search
--------------------
Initial state

*  1  3  
4  2  5  
7  8  6  

1  *  3  
4  2  5  
7  8  6  

1  2  3  
4  *  5  
7  8  6  

1  2  3  
4  5  *  
7  8  6  

1  2  3  
4  5  6  
7  8  *  

Goal state
--------------------
Number of moves =  4
Number of states enqueued =  28
~~~~~~~~~~~~~~~~~~~~
#### Provide a short comparative analysis of two heuristics used for A*
~~~~~~~~~~~~~~~~~~~~
- The two heuristics used were for the two different functions Astar1 & Astar2.
- Astar1: Misplaced Tiles Heuristic (calculated how many tiles were misplaced)
- Astar2: Manhattan Distance (calculated manhattan distance from current state to goal state for each tile)
- The best way to compare the two heuristics is to set up a way to compare their performance for different depths (d).
- We can set this up by starting at the goal node and moving the empty space 1 step at a time. 
- We can do this up to 10 times since we are limited by a depth of 10.
Goal State: 1 2 3 
	    4 5 6 
 	    7 8 *
----------------------------------------------------------------------
Example initial state for a playthrough with a depth of one:   1 2 3 
					                       4 5 6 
					                       7 * 8
----------------------------------------------------------------------
Example initial state for a playthrough with a depth of two:   1 2 3 
					                       4 5 6 
					                       * 7 8
----------------------------------------------------------------------
Example initial state for a playthrough with a depth of three: 1 2 3 
					                       * 5 6 
					                       4 7 8
----------------------------------------------------------------------
- And so on and so on.......

NoM = Number of Moves
NoSE = Number of States Enqueued

   Depth|NoM for Astar1|NoSE for Astar1|NoM for Astar2|NoSE for Astar2|Space Separated Input 
---------------------------------------------------------------------------------------------------------
       1|             1|              4|             1|     	     4| 1 2 3 4 5 6 7 * 8
---------------------------------------------------------------------------------------------------------
       2|             2|              5|    	     2|   	     5| 1 2 3 4 5 6 * 7 8
---------------------------------------------------------------------------------------------------------
       3|    	      3|              7|             3|              7| 1 2 3 * 5 6 4 7 8
---------------------------------------------------------------------------------------------------------
       4|    	      4|             10|             4|             10| 1 2 3 5 * 6 4 7 8
---------------------------------------------------------------------------------------------------------
       5|    	      5|             12|             5|             12| 1 2 3 5 6 * 4 7 8
---------------------------------------------------------------------------------------------------------
       6|    	      6|             13|             6|             13| 1 2 * 5 6 3 4 7 8
---------------------------------------------------------------------------------------------------------
       7|    	      7|             15|             7|             15| 1 * 2 5 6 3 4 7 8 
---------------------------------------------------------------------------------------------------------
       8|    	      8|             16|             8|             16| * 1 2 5 6 3 4 7 8
---------------------------------------------------------------------------------------------------------
       9|    	      9|             33|             9|             21| 5 1 2 * 6 3 4 7 8
---------------------------------------------------------------------------------------------------------
      10|    	     10|             58|            10|             30| 5 1 2 4 6 3 * 7 8
---------------------------------------------------------------------------------------------------------
Analysis:
- As we can see the performance of the two heuristsics are basically the same (comparable) up until a depth of 8.
- Going past this depth we can clearly see that the Astar2 heuristic enqueues less and less nodes as we progress towards the depth limit.
- Of course we could go further to see the true disparity, but at least by depth 10 Astar2 enqueues half the states Astar1 enqueues.
- From this we can clearly state that the Astar2 heuristic dominates the Astar1 heuristic.
~~~~~~~~~~~~~~~~~~~~
