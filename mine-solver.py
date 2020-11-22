from typing import Iterable, Iterator, List, NamedTuple, Optional, \
    Set, Tuple
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

    def neighbors(self, p: Point) -> Tuple[int, List[Point]]:
        num_mines = 0
        unknown: List[Point] = []
        value = self.at(p)
        assert value is not None
        assert value >= 0
        for y in range(p.y - 1, p.y + 2):
            for x in range(p.x - 1, p.x + 2):
                pp = Point(x, y)
                neighbor = self.at(pp)
                if neighbor == -2:
                    unknown.append(pp)
                elif neighbor == -1:
                    num_mines += 1
        return value - num_mines, unknown

    def grind_step(self) -> Optional[List[Point]]:
        changed = False
        problematic: List[Point] = []
        for i in range(len(self.known)):
            if self.known[i] < 0:
                continue
            p = Point(i % self.size.x, i // self.size.x)
            num_mines, unknown = self.neighbors(p)
            if not unknown:
                continue
            if num_mines == 0:
                for pp in unknown:
                    self.set(pp, self.attempt(pp))
                changed = True
            elif num_mines == len(unknown):
                for pp in unknown:
                    self.set(pp, -1)
                changed = True
            elif not changed:
                problematic.append(p)
        if changed:
            return None
        return problematic

    def grind(self) -> List[Point]:
        problematic = None
        while problematic is None:
            problematic = self.grind_step()
        return problematic

    def find_possibilities_inner(
            self, points: List[Point],
            num: int, result: Set[Point]) -> Iterator[Set[Point]]:
        if num == 0:
            yield result
            return
        if num > len(points):
            return
        remaining = points[1:]
        yield from self.find_possibilities_inner(remaining, num, result)
        yield from self.find_possibilities_inner(
            remaining, num - 1, result | set([points[0]]))

    def find_possibilities(
            self, points: List[Point], num: int) -> Iterator[Set[Point]]:
        return self.find_possibilities_inner(points, num, set())

    def is_consistent(
            self, problematic: List[Point], mines: Set[Point]) -> bool:
        not_mines: Set[Point] = set()
        processed: Set[Point] = set()
        minp = Point(
            min(p.x for p in mines),
            min(p.y for p in mines))
        maxp = Point(
            max(p.x for p in mines),
            max(p.y for p in mines))

        changed = True
        while changed:
            changed = False
            for p in problematic:
                if p.x < minp.x - 1 or p.x > maxp.x + 1 \
                        or p.y < minp.y - 1 or p.y > maxp.y + 1 \
                        or p in processed:
                    continue
                num_mines, unknown = self.neighbors(p)
                unknown_set = set(unknown)
                current_not_mines = unknown_set & not_mines
                current_mines = unknown_set & mines
                print('  {}: u{} +{} -{}'.format(
                    p, unknown, current_mines, current_not_mines))
                if num_mines < len(current_mines) or \
                        num_mines > len(unknown) - len(current_not_mines):
                    return False
                if num_mines == len(current_mines):
                    processed.add(p)
                    new_not_mines = unknown_set - current_mines
                    print('   -{}'.format(new_not_mines))
                    before = len(not_mines)
                    not_mines |= new_not_mines
                    if len(not_mines) != before:
                        changed = True
                        for pp in new_not_mines:
                            minp = Point(min(minp.x, pp.x), min(minp.y, pp.y))
                            maxp = Point(max(maxp.x, pp.x), max(maxp.y, pp.y))
                if num_mines == len(unknown) - len(current_not_mines):
                    processed.add(p)
                    new_mines = unknown_set - current_not_mines
                    print('   +{}'.format(new_mines))
                    before = len(mines)
                    mines |= new_mines
                    if len(mines) != before:
                        changed = True
                        for pp in new_mines:
                            minp = Point(min(minp.x, pp.x), min(minp.y, pp.y))
                            maxp = Point(max(maxp.x, pp.x), max(maxp.y, pp.y))
            print('  --')

        return True

    def eliminate(self, problematic: List[Point]) -> bool:
        for p in problematic:
            print(p)
            num_mines, unknown = self.neighbors(p)
            resolution = {pp: [False, False] for pp in unknown}
            for possibility in self.find_possibilities(unknown, num_mines):
                print(' {}'.format(possibility))
                if not self.is_consistent(problematic, possibility):
                    print('not consistent')
                    continue
                for pp in unknown:
                    resolution[pp][int(pp in possibility)] = True
            changed = False
            for pp, (no, yes) in resolution.items():
                if no and yes:
                    continue
                if no:
                    self.set(pp, self.attempt(pp))
                elif yes:
                    self.set(pp, -1)
                else:
                    self.print()
                    raise Exception('Something is inconsistent')
                changed = True
            if changed:
                return True
        return False

    def solve(self, start: Point) -> None:
        self.size = self.table.get_size()
        self.known = [-2 for i in range(self.size.x * self.size.y)]
        self.set(start, self.attempt(start))

        while True:
            problematic = self.grind()
            self.print()
            if not problematic:
                break
            if not self.eliminate(problematic):
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
