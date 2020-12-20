from typing import Any, Iterable, Iterator, List, NamedTuple, Optional, \
    Set, Tuple
import argparse
import sys
import traceback


class Point(NamedTuple):
    x: int
    y: int


class CannotSolve(Exception):
    pass


class Table:
    def attempt(self, p: Point) -> int:
        raise NotImplementedError

    def get_size(self) -> Point:
        raise NotImplementedError

    def total_mines(self) -> int:
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

    def total_mines(self) -> int:
        return len(self.mines)


class InteractiveTable(Table):
    def __init__(self, size: Point, mines: int):
        self.size = size
        self.mines = mines

    def get_size(self) -> Point:
        return self.size

    def attempt(self, p: Point) -> int:
        while True:
            try:
                sys.stdout.write('[0-8|m] ? ')
                sys.stdout.flush()
                line = sys.stdin.readline()
                if line == 'm':
                    return -1
                value = int(line)
                assert value >= 0
                assert value <= 8
                return value
            except KeyboardInterrupt:
                raise
            except Exception:
                traceback.print_exc()

    def total_mines(self) -> int:
        return self.mines


class Solver:
    def __init__(self, table: Table, interactive: bool):
        self.table = table
        self.known: List[int] = []
        self.size = Point(0, 0)
        self.remaining_mines = 0
        self.interactive = interactive

    def index_to_point(self, index: int) -> Point:
        return Point(index % self.size.x, index // self.size.x)

    def at(self, p: Point) -> Optional[int]:
        if p.x < 0 or p.x >= self.size.x or p.y < 0 or p.y >= self.size.y:
            return None
        return self.known[p.y * self.size.x + p.x]

    def set_index(self, index: int, value: int) -> None:
        if self.known[index] != -1 and value == -1:
            self.remaining_mines -= 1
        self.known[index] = value

    def set(self, p: Point, value: int) -> None:
        self.set_index(p.y * self.size.x + p.x, value)

    def print(self) -> None:
        def symbol(x: int, y: int) -> str:
            n = self.at(Point(x, y))
            if n == -4:
                return '?'
            if n == -3:
                return '!'
            if n == -2:
                return ','
            if n == -1:
                return 'X'
            if n == 0:
                return '.'
            return str(n)

        print('\n'.join(
            ' '.join(symbol(x, y) for x in range(self.size.x))
            for y in range(self.size.y)))
        print()

    def attempt(self, p: Point) -> None:
        if self.interactive:
            self.set(p, -4)
            self.print()
        result = self.table.attempt(p)
        if result == -1:
            self.set(p, -3)
            raise CannotSolve('Stepped on mine')
        self.set(p, result)

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
            p = self.index_to_point(i)
            num_mines, unknown = self.neighbors(p)
            if not unknown:
                continue
            if num_mines == 0:
                for pp in unknown:
                    self.attempt(pp)
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
            self, problematic: List[Point], points: List[Point], num: int,
            result: List[Point], mines: Set[Point],
            not_mines: Set[Point]) -> Iterator[List[Point]]:
        if num > len(points):
            return
        if mines and not self.is_consistent(problematic, mines, not_mines):
            return
        if num == 0:
            yield result
            return

        p = points[0]
        remaining = points[1:]
        if p not in mines:
            yield from self.find_possibilities_inner(
                problematic, remaining, num, result,
                set(mines), set(not_mines))

        if p not in not_mines:
            yield from self.find_possibilities_inner(
                problematic, remaining, num - 1, result + [p],
                mines | set([p]), not_mines)

    def find_possibilities(
            self, problematic: List[Point], points: List[Point],
            num: int) -> Iterator[List[Point]]:
        return self.find_possibilities_inner(problematic, points, num,
            [], set(), set())

    def is_consistent(
            self, problematic: List[Point], mines: Set[Point],
            not_mines: Set[Point]) -> bool:
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

            if self.remaining_mines < len(mines):
                return False

            for p in problematic:
                if p.x < minp.x - 1 or p.x > maxp.x + 1 \
                        or p.y < minp.y - 1 or p.y > maxp.y + 1 \
                        or p in processed:
                    continue
                num_mines, unknown = self.neighbors(p)
                unknown_set = set(unknown)
                current_not_mines = unknown_set & not_mines
                current_mines = unknown_set & mines
                if num_mines < len(current_mines) or \
                        num_mines > len(unknown) - len(current_not_mines):
                    return False
                if num_mines == len(current_mines):
                    processed.add(p)
                    new_not_mines = unknown_set - current_mines
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
                    before = len(mines)
                    mines |= new_mines
                    if len(mines) != before:
                        changed = True
                        for pp in new_mines:
                            minp = Point(min(minp.x, pp.x), min(minp.y, pp.y))
                            maxp = Point(max(maxp.x, pp.x), max(maxp.y, pp.y))

        return True

    def eliminate(self, problematic: List[Point]) -> bool:
        for p in problematic:
            num_mines, unknown = self.neighbors(p)
            resolution = {pp: [False, False] for pp in unknown}
            for possibility in self.find_possibilities(problematic, unknown, num_mines):
                for pp in unknown:
                    resolution[pp][int(pp in possibility)] = True
            changed = False
            for pp, (no, yes) in resolution.items():
                if no and yes:
                    continue
                if no:
                    self.attempt(pp)
                elif yes:
                    self.set(pp, -1)
                else:
                    self.print()
                    raise RuntimeError('Something is inconsistent')
                changed = True
            if changed:
                return True
        return False

    def solve(self, start: Point) -> None:
        self.size = self.table.get_size()
        self.remaining_mines = self.table.total_mines()
        self.known = [-2 for i in range(self.size.x * self.size.y)]
        self.attempt(start)

        while True:
            problematic = self.grind()
            if not self.interactive:
                self.print()
            if not problematic:
                break
            if not self.eliminate(problematic):
                raise CannotSolve('Cannot solve')
            if not self.interactive:
                self.print()
                print('-----')

        num_unsolved = sum(v == -2 for v in self.known)
        if num_unsolved == self.remaining_mines:
            for i in range(len(self.known)):
                if self.known[i] == -2:
                    self.set_index(i, -1)
        elif self.remaining_mines == 0:
            for i in range(len(self.known)):
                if self.known[i] == -2:
                    self.attempt(self.index_to_point(i))
        else:
            raise CannotSolve('Field has unreachable part')

        print()
        print('-' * self.size.x * 2)
        self.print()
        print('-' * self.size.x * 2)


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
            if x != 0:
                width = max(width, x)
                y += 1
    return SimpleTable(Point(width, y), mines)


def get_file_solver(args: 'Any') -> Solver:
    table = load_table(args.file)
    solver = Solver(table, interactive=False)
    if args.startx is None and args.starty is None:
        with open(args.file) as f:
            poss = f.readline().split(' ')
            args.startx = int(poss[0])
            args.starty = int(poss[1])
    return solver


def get_interactive_solver(args: 'Any') -> Solver:
    return Solver(
        InteractiveTable(Point(args.width, args.height), args.mines),
        interactive=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--startx', type=int)
    parser.add_argument('-y', '--starty', type=int)
    subparsers = parser.add_subparsers()

    file_parser = subparsers.add_parser('file', aliases=['f'])
    file_parser.add_argument('file')
    file_parser.set_defaults(func=get_file_solver)

    interactive_parser = subparsers.add_parser('interactive', aliases=['i'])
    interactive_parser.add_argument('-w', '--width', type=int, required=True)
    interactive_parser.add_argument('-hg', '--height', type=int, required=True)
    interactive_parser.add_argument('-m', '--mines', type=int, required=True)
    interactive_parser.set_defaults(func=get_interactive_solver)

    args = parser.parse_args()
    solver = args.func(args)

    if args.startx is None or args.starty is None:
        raise RuntimeError('Starting position is not given')

    try:
        solver.solve(Point(args.startx, args.starty))
    except CannotSolve as e:
        solver.print()
        print(e.args[0])
        sys.exit(2)
