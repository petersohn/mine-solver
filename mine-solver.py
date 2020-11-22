from typing import Iterable, List, NamedTuple, Optional, Tuple
import argparse


class Point(NamedTuple):
    x: int
    y: int


class Table:
    def attempt(self, p: Point) -> int:
        raise NotImplementedError

    def get_size(self) -> Point:
        raise NotImplementedError


def check_size(p: Point, size: Point) -> None:
    assert p.x >= 0
    assert p.y >= 0
    assert p.x < size.x
    assert p.y < size.y


class SimpleTable(Table):
    def __init__(self, size: Point, mines: Iterable[Point]):
        self.size = size
        for m in mines:
            check_size(m, size)
        self.mines = set(mines)

    def get_size(self) -> Point:
        return self.size

    def attempt(self, p: Point) -> int:
        check_size(p, self.size)
        if p in self.mines:
            return -1
        num = 0
        for y in range(p.y - 1, p.y + 2):
            for x in range(p.x - 1, p.x + 2):
                if x < 0 or x >= self.size.x or y < 0 or y >= self.size.y:
                    continue
                pp = Point(x, y)
                if pp != p and pp in self.mines:
                    num += 1
        return num


class Solver:
    def __init__(self, table: Table):
        self.table = table
        self.known: List[int] = []
        self.size = Point(0, 0)

    def at(self, p: Point) -> Optional[int]:
        if p.x < 0 or p.x >= self.size.x or p.y < 0 or p.y >= self.size.y:
            return None
        return self.known[p.y * self.size.x + p.x]

    def set(self, p: Point, value: int) -> None:
        self.known[p.y * self.size.x + p.x] = value

    def print(self) -> None:
        def symbol(x: int, y: int) -> str:
            n = self.at(Point(x, y))
            if n == -3:
                return '!'
            if n == -2:
                return ','
            if n == -1:
                return 'X'
            return str(n)

        print('\n'.join(
            ' '.join(symbol(x, y) for x in range(self.size.x))
            for y in range(self.size.y)))
        print()

    def attempt(self, p: Point) -> int:
        result = self.table.attempt(p)
        if result == -1:
            self.set(p, -3)
            self.print()
            raise Exception('Stepped on mine')
        return result

    def neighbors(self, p: Point) -> Optional[Tuple[int, List[Point]]]:
        mines = 0
        unknown: List[Point] = []
        value = self.at(p)
        if value is None or value < 0:
            return None
        for y in range(p.y - 1, p.y + 2):
            for x in range(p.x - 1, p.x + 2):
                pp = Point(x, y)
                neighbor = self.at(pp)
                if neighbor == -2:
                    unknown.append(pp)
                elif neighbor == -1:
                    mines += 1
        return value - mines, unknown

    def grind(self) -> Optional[List[Point]]:
        changed = False
        problematic: List[Point] = []
        for i in range(len(self.known)):
            if self.known[i] < 0:
                continue
            p = Point(i % self.size.x, i // self.size.x)
            neighbors = self.neighbors(p)
            assert neighbors is not None
            mines, unknown = neighbors
            if not unknown:
                continue
            if mines == 0:
                for pp in unknown:
                    self.set(pp, self.attempt(pp))
                changed = True
            elif mines == len(unknown):
                for pp in unknown:
                    self.set(pp, -1)
                changed = True
            elif not changed:
                problematic.append(p)
        if changed:
            return None
        return problematic

    def solve(self, start: Point) -> None:
        self.size = self.table.get_size()
        self.known = [-2 for i in range(self.size.x * self.size.y)]
        self.set(start, self.attempt(start))

        while True:
            while True:
                problematic = self.grind()
                if problematic is not None:
                    break
            self.print()
            if not problematic:
                break
            raise Exception('Cannot solve')
            self.print()
            print('-----')
        assert not any(v == -2 for v in self.known)


def load_table(filename: str) -> Table:
    width = 0
    mines: List[Point] = []
    y = 0
    with open(filename) as f:
        for line in f:
            x = 0
            for c in line:
                if c == 'x':
                    mines.append(Point(x, y))
                elif c != 'o':
                    continue
                x += 1
            width = max(width, x)
            y += 1
    return SimpleTable(Point(width, y), mines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)
    parser.add_argument('-x', '--startx', type=int, required=True)
    parser.add_argument('-y', '--starty', type=int, required=True)
    args = parser.parse_args()
    table = load_table(args.file)
    solver = Solver(table)
    solver.solve(Point(args.startx, args.starty))
