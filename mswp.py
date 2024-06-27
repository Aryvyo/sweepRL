import numpy as np
import random


class Minesweeper:
    def __init__(self, size=10, num_mines=10):
        self.size = size
        self.num_mines = num_mines
        self.grid = np.zeros((size, size), dtype=int)
        self.revealed = np.zeros((size, size), dtype=bool)
        self.game_over = False
        self.place_mines()
        self.calculate_numbers()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.revealed = np.zeros((self.size, self.size), dtype=bool)
        self.game_over = False
        self.place_mines()
        self.calculate_numbers()
        return self.get_visible_grid()

    def place_mines(self):
        mines_placed = 0
        while mines_placed < self.num_mines:
            x, y = random.randint(
                0, self.size-1), random.randint(0, self.size-1)
            if self.grid[x, y] != -1:
                self.grid[x, y] = -1
                mines_placed += 1

    def calculate_numbers(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] != -1:
                    self.grid[i, j] = self.count_adjacent_mines(i, j)

    def count_adjacent_mines(self, x, y):
        count = 0
        for i in range(max(0, x-1), min(self.size, x+2)):
            for j in range(max(0, y-1), min(self.size, y+2)):
                if self.grid[i, j] == -1:
                    count += 1
        return count

    def reveal(self, x, y):
        if self.game_over or self.revealed[x, y]:
            return

        self.revealed[x, y] = True

        if self.grid[x, y] == -1:
            self.game_over = True
            return True
        elif self.grid[x, y] == 0:
            for i in range(max(0, x-1), min(self.size, x+2)):
                for j in range(max(0, y-1), min(self.size, y+2)):
                    if not self.revealed[i, j]:
                        self.reveal(i, j)
                        return False

        if np.count_nonzero(self.revealed) == self.size**2 - self.num_mines:
            self.game_over = True
            print("Congratulations! You've won!")

    def get_grid(self):
        return self.grid.copy()

    def get_visible_grid(self):
        visible = np.full((self.size, self.size), -2, dtype=int)
        visible[self.revealed] = self.grid[self.revealed]
        return visible

    def print_grid(self):
        visible = self.get_visible_grid()
        for row in visible:
            print(' '.join([str(cell) if cell != -2 else '.' for cell in row]))
