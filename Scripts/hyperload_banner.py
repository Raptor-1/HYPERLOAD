"""
hyperload_banner.py
Print the HYPERLOAD banner in the terminal.
Import and call print_banner() at the top of any suite's main().
"""

RESET  = "\033[0m"
BOLD   = "\033[1m"

# Colours (ANSI 256-colour foreground)
BLUE   = "\033[38;5;39m"
GREEN  = "\033[38;5;85m"
PINK   = "\033[38;5;213m"
ORANGE = "\033[38;5;208m"
PURPLE = "\033[38;5;141m"
CYAN   = "\033[38;5;81m"
LIME   = "\033[38;5;118m"
YELLOW = "\033[38;5;226m"
RED    = "\033[38;5;203m"
DIM    = "\033[38;5;240m"
WHITE  = "\033[38;5;255m"

# Each letter is 7 rows × variable columns of symbols
# Symbol key:
#   H = ★  Y = ✦  P = ☽  E = ⬡  R = ◉  L = ⊹  O = ✪  A = ☄  D = ♃

H = [
    "★ ★",
    "★ ★",
    "★ ★",
    "★★★",
    "★ ★",
    "★ ★",
    "★ ★",
]

Y = [
    "✦   ✦",
    " ✦ ✦ ",
    "  ✦  ",
    "  ✦  ",
    "  ✦  ",
    "  ✦  ",
    "  ✦  ",
]

P = [
    "☽☽☽ ",
    "☽  ☽",
    "☽  ☽",
    "☽☽☽ ",
    "☽   ",
    "☽   ",
    "☽   ",
]

E = [
    "⬡⬡⬡",
    "⬡   ",
    "⬡   ",
    "⬡⬡⬡",
    "⬡   ",
    "⬡   ",
    "⬡⬡⬡",
]

R = [
    "◉◉◉ ",
    "◉  ◉",
    "◉  ◉",
    "◉◉◉ ",
    "◉ ◉ ",
    "◉  ◉",
    "◉  ◉",
]

L = [
    "⊹   ",
    "⊹   ",
    "⊹   ",
    "⊹   ",
    "⊹   ",
    "⊹   ",
    "⊹⊹⊹",
]

O = [
    " ✪✪✪ ",
    "✪   ✪",
    "✪   ✪",
    "✪   ✪",
    "✪   ✪",
    "✪   ✪",
    " ✪✪✪ ",
]

A = [
    "  ☄  ",
    " ☄ ☄ ",
    "☄   ☄",
    "☄☄☄☄☄",
    "☄   ☄",
    "☄   ☄",
    "☄   ☄",
]

D = [
    "♃♃♃ ",
    "♃  ♃",
    "♃  ♃",
    "♃  ♃",
    "♃  ♃",
    "♃  ♃",
    "♃♃♃ ",
]

LETTERS = [
    (H, BLUE),
    (Y, GREEN),
    (P, PINK),
    (E, ORANGE),
    (R, PURPLE),
    (L, CYAN),
    (O, LIME),
    (A, YELLOW),
    (D, RED),
]

LEGEND = (
    f"{DIM}  ★=stars  ✦=nebulae  ☽=moons  ⬡=galaxies  ◉=black holes"
    f"  ⊹=supernovae  ✪=pulsars  ☄=comets  ♃=planets{RESET}"
)


def print_banner():
    gap = "  "
    rows = []
    for row_idx in range(7):
        line = gap
        for letter_rows, color in LETTERS:
            line += BOLD + color + letter_rows[row_idx] + RESET + gap
        rows.append(line)

    width = 72
    border_top    = DIM + "  " + "·" * width + RESET
    border_bottom = DIM + "  " + "·" * width + RESET

    byline = (
        f"{DIM}  {'advanced siril python processing suite':^{width}}{RESET}"
    )
    author = (
        f"{WHITE}{BOLD}  {'by  N. Marcell M.':^{width}}{RESET}"
    )

    print()
    print(border_top)
    print()
    for row in rows:
        print(row)
    print()
    print(byline)
    print(author)
    print()
    print(LEGEND)
    print()
    print(border_bottom)
    print()


if __name__ == "__main__":
    print_banner()
