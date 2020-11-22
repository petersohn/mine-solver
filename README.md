# Minesweeper solver

Try to solve a minesweeper map. Usage:

```
python3 mine-solver.py -f <filename> -x <start-x> -y <start-y>
```

The input file is read line by line, `x` means mine, `o` means not mine, and
everything else is ignored.

## Algorithm

In the following pseudo-code, coordinates are represented as `{x, y}`.

The algorithm consists of the following subroutines:

### Attempt

```python
attempt(p)
```

Query the number of mines surrounding the attempted cell. This is one step in
the game.

### Neighbor calculation

```python
neighbors(p)
```

It returns the remaining number of mines near the cell, and a list of
coordinates of the free cells nearby. Example:

```
x 3 ,
2 , ,
1 , ,
```

In this case, `neighbors({1, 0})` returns `(2, [{2, 0}, {1, 1}, {2, 1}])`.

### Grind

```python
grind()
```

Try to solve brute-force as much as possible. It does the following until it
gets stuck:

* If a cell has the same number of remaining mines as it has free neighbors, mark
  all free neighbors as mine.
* If a cell has no remaining mines, attempt all free neighbors.

When the grinding is stuck, it returns a list of `problematic` cells. These are
the cells that are already attempted, but have remaining mines and have more
free neighbors than remaining mines.

### Eliminate

```python
eliminate(problematic)
```

For each problematic cell, collect each possible substitution for its free
neighbors. Check each of these substitutions for consistency. For each free
neighbor:

* If it is a mine in all consistent substitutions, mark it as a mine.
* If it is not a mine in all consistent substitutions, attempt it.

If there are changes for a problematic cell, return.

For example:

```
0 1 ,
0 2 ,
0 2 ,
0 1 ,
```

In this case, problematic cells are the ones in column 1. For cell `{1, 1}`,
the possible substitutions are (these cells marked as mine): `[{2, 0}, {2, 1}],
[{2, 0}, {2, 2}], [{2, 1}, {2, 2}]`. The first of these is inconsistent,
because cell `{1, 0}` would have 2 bombs near it, but it has a value of only 1.
The cell `{2, 2}` is mine in all consistent substitutions, so it will be marked
as mine.

### Consistency check

Check the possible implications of a substitution without marking or attempting
any cells. Return whether the substitution is consistent or not.

Maintain the following sets:

* `mines`: the cells marked as mine.
* `not_mines`: the cells marked as not mine.

For each problematic cell, repeat until there are no more changes or an
inconsistency is found:

* If there are more neighbors in `mines` than remaining mines, the substitution
  is inconsistent.
* If there are more remaining mines than free neighbors not in `not_mines`, the
  substitution is inconsistent.
* If the number of neighbors in `mines` equals the number of remaining mines,
  put all neighbors not in `mines` into `not_mines`.
* If the number of remaining mines equals the number of free neighbors not in
  `not_mines`, put all neighbors not in `not_mines` into `mines`.

Additionally, if at any point the length of `mines` exceeds the total remaining
mines, the substitution is inconsistent.

### Solve

```python
solve(start)
```

First, attempt the start point. Then, repeatedly grind and eliminate until all
fields are marked or attempted. If stuck, then the game is not solvable.
