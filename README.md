# AIVD 2018 challenge 17
An attempt on building a solver for puzzle 17 from the Dutch AIVD Christmas
puzzle challenge.

The solver uses a set of strategies to attempt to progressively solve the input
puzzle. The implemented strategies are heavily inspired by well known [sudoku
solving strategies][strategy-families], along with some self conceived
strategies.

## Example output
```
Input:
                          A B C A A A C A A A F B
                          F D F H E E D D E A K C
                          I D H H F F E E E I K E
                          I E I J F M I G H N O E
                          I I K N G M I G L O P F
                          J J M O L N L O Q Q Q I
                          K M R S O O L R T R R I
                          L N T T O P R S W R T R
                          R O X U P Q T S X R T T
                          S Q X U T T V T X S V T
                          X R X V V X X U Y V W U
                          X T X X W X X Y Z X W V
                          Z V Z X W Z X Z Z X Y X

A E H I N R T T W X X Z   . . . . . . . . . . . .
A D E I N S T T V X X X   . . . . . . . . . . . .
D E J M N O R T T X X X   . . . . . . . . . . . .
E E I K M R S S T W X X   . . . . . . . . . . . .
D D E O O P R R T X X X   . . . . . . . . . . . .
A A E G I K K Q R R S Z   . . . . . . . . . . . .
C F H L M M Q T W X Y Z   . . . . . . . . . . . .
A A C F H Q S U V W W Y   . . . . . . . . . . . .
F I I J L O O R T V V X   . . . . . . . . . . . .
A C F G H I J L O Q X Y   . . . . . . . . . . . .
A B B F F F L L O P R X   . . . . . . . . . . . .
E E I I O Q R T U U Z Z   . . . . . . . . . . . .
G I K N P T U V V V X Z   . . . . . . . . . . . .

Start solving...
State after pass #0:
                          A . C A A A C A A A F .
                          F D F H E E D D E A K C
                          I D H H F F E E E I K E
                          I E I J F M I G H N O E
                          I I K N G M I G L O P F
                          J J M O L N L O Q Q Q I
                          K M R S O O L R T R R I
                          L N T T O P R S W R T R
                          R O X U P Q T S X R T T
                          S Q X U T T V T X S V T
                          X R X V V X X U Y V W U
                          X T X X W X X Y Z X W V
                          Z V Z X W Z X Z Z X Y X

A E H I N R T T W X X Z   . . . . . . . . . . . .
A D E I N S T T V X X X   . . . . . . . . . . . .
D E J M N O R T T X X X   . . . . . . . . . . . .
E E I K M R S S T W X X   . . . . . . . . . . . .
D D E O O P R R T X X X   . . . . . . . . . . . .
A A E G I K K Q R R S Z   . . . . . . . . . . . .
C F H L M M Q T W X Y Z   . . . . . . . . . . . .
A A C F H Q S U V W W Y   . . . . . . . . . . . .
F I I J L O O R T V V X   . . . . . . . . . . . .
A C F G H I J L O Q X Y   . . . . . . . . . . . .
A . . F F F L L O P R X   . B . . . . . . . . . B
E E I I O Q R T U U Z Z   . . . . . . . . . . . .
G I K N P T U V V V X Z   . . . . . . . . . . . .

Solving stalled, could not progress any further
```

## License
This project is released under the GNU GPL-3.0 license.
Check out the [LICENSE](LICENSE) file for more information. 

[strategy-families]: http://www.sudokuwiki.org/Strategy_Families
