import torch

class VectorizedMinesweeper:
    """
    Vectorized batch Minesweeper environment for rectangular boards.
    All batch operations are done on torch tensors.
    """
    def __init__(self, batch_size=256, height=8, width=8, num_mines=10, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, switching to CPU.")
            device = 'cpu'
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.reset()

    def reset(self):
        B, H, W = self.batch_size, self.height, self.width
        self.board = torch.zeros((B, H, W), dtype=torch.int8, device=self.device)  # -1 for mine, 0-8 for clues
        self.revealed = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)
        self.flagged = torch.zeros((B, H, W), dtype=torch.bool, device=self.device)
        self.game_over = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.win = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.mines_placed = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(B, dtype=torch.long, device=self.device)
        self.mines_left = torch.full((B,), self.num_mines, dtype=torch.long, device=self.device)

    def _place_mines(self, first_click_r, first_click_c, indices=None):
        B, H, W = self.batch_size, self.height, self.width
        if indices is None:
            indices = range(B)
        for idx, fr, fc in zip(indices, first_click_r, first_click_c):
            if self.mines_placed[idx]:
                continue
            safe_zone = set()
            for r in range(max(0, fr-1), min(H, fr+2)):
                for c in range(max(0, fc-1), min(W, fc+2)):
                    safe_zone.add((r, c))
            available = [(r, c) for r in range(H) for c in range(W) if (r, c) not in safe_zone]
            if len(available) < self.num_mines:
                raise ValueError(f"Not enough cells to place mines for batch {idx}.")
            rand_idx = torch.randperm(len(available), device=self.device)[:self.num_mines]
            for midx in rand_idx:
                r, c = available[int(midx)]
                self.board[idx, r, c] = -1
            self.mines_placed[idx] = True
        self._calculate_numbers(indices)

    def _calculate_numbers(self, indices=None):
        B, H, W = self.batch_size, self.height, self.width
        if indices is None:
            indices = range(B)
        for idx in indices:
            for r in range(H):
                for c in range(W):
                    if self.board[idx, r, c] == -1:
                        continue
                    count = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and self.board[idx, nr, nc] == -1:
                                count += 1
                    self.board[idx, r, c] = count

    def reveal(self, r, c, indices=None):
        # r, c: lists or tensors of length N (N <= batch_size)
        if indices is None:
            indices = range(self.batch_size)
        for i, ri, ci in zip(indices, r, c):
            if self.game_over[i] or self.revealed[i, ri, ci] or self.flagged[i, ri, ci]:
                continue
            if not self.mines_placed[i]:
                self._place_mines([ri], [ci], [i])
            if self.board[i, ri, ci] == -1:
                self.revealed[i, ri, ci] = True
                self.game_over[i] = True
                self.win[i] = False
            else:
                self._reveal_cell(i, ri, ci)
                if self._check_win(i):
                    self.game_over[i] = True
                    self.win[i] = True

    def _reveal_cell(self, i, r, c):
        H, W = self.height, self.width
        stack = [(int(r), int(c))]
        while stack:
            rr, cc = stack.pop()
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if self.revealed[i, rr, cc]:
                continue
            self.revealed[i, rr, cc] = True
            if self.board[i, rr, cc] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = rr + dr, cc + dc
                        if (dr != 0 or dc != 0) and 0 <= nr < H and 0 <= nc < W:
                            stack.append((nr, nc))

    def flag(self, r, c, indices=None):
        if indices is None:
            indices = range(self.batch_size)
        for i, ri, ci in zip(indices, r, c):
            if self.game_over[i] or self.revealed[i, ri, ci]:
                continue
            self.flagged[i, ri, ci] = not self.flagged[i, ri, ci]

    def _check_win(self, i):
        H, W = self.height, self.width
        return torch.sum(self.revealed[i]) == H * W - self.num_mines

    def get_state(self):
        return {
            'board': self.board.clone(),
            'revealed': self.revealed.clone(),
            'flagged': self.flagged.clone(),
            'game_over': self.game_over.clone(),
            'win': self.win.clone()
        }

    def batch_done(self):
        return self.game_over.clone()

    def batch_won(self):
        return self.win.clone()
