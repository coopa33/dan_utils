# Globals
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
def green_str(s: str) -> str:
    return f"{GREEN}{s}{RESET}"
def red_str(s: str) -> str:
    return f"{RED}{s}{RESET}"
def yellow_str(s: str) -> str:
    return f"{YELLOW}{s}{RESET}"
def cyan_str(s: str) -> str:
    return f"{CYAN}{s}{RESET}"





def print_begin(title: str) -> None:
    title_str = f"+++ {title} +++"
    title_str_col = green_str(title_str)
    print(title_str_col)
    return len(title_str)

def print_end(title_length: int) -> None:
    print(f"{green_str('+' * title_length)}")
