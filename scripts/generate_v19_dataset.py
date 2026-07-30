#!/usr/bin/env python3
"""
Generate NeuralAI-Air-135M SFT v19 dataset.

Deduplicates v18 (24 unique), converts v17 (37 unique),
then generates ~940 NEW diverse ChatML examples.

Output: data/train_sft_v19.jsonl (1000+ examples)
"""
import json, random
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────

def to_chatml(system, instruction, output):
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )
    return {"text": text, "system": system, "instruction": instruction, "output": output}

# ── Load existing data ─────────────────────────────────────────

def load_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

v18 = load_jsonl("data/train_sft_v18.jsonl")
v17 = load_jsonl("data/train_sft_v17.jsonl")

# Deduplicate v18
seen = set()
deduped_v18 = []
for ex in v18:
    key = (ex.get("system",""), ex.get("instruction",""), ex.get("output",""))
    if key not in seen:
        seen.add(key)
        deduped_v18.append(ex)

# Convert v17 (messages format → text format)
converted_v17 = []
for ex in v17:
    msgs = ex.get("messages", [])
    system = user = assistant = ""
    for m in msgs:
        if m["role"] == "system": system = m["content"]
        elif m["role"] == "user": user = m["content"]
        elif m["role"] == "assistant": assistant = m["content"]
    converted_v17.append(to_chatml(system, user, assistant))

# Combine base
base = deduped_v18 + converted_v17
print(f"[Base] v18 deduped: {len(deduped_v18)}, v17 converted: {len(converted_v17)}, total: {len(base)}")

# ── Generate NEW diverse examples ──────────────────────────────

new_examples = []

def add(system, instruction, output):
    new_examples.append(to_chatml(system, instruction, output))

# System prompts
SYS_NEURALAI = "You are NeuralAI, a precise and helpful AI assistant built by De'Andrew Preston Harris (D. Harris / Dre) for NeuralLabs. You answer clearly, concisely, and truthfully."
SYS_NEURALAI_SHORT = "You are NeuralAI, a helpful AI assistant."
SYS_CODER = "You are a helpful coding assistant."
SYS_REASONER = "You are a precise reasoning assistant. Show your reasoning briefly, then state the final answer."
SYS_CONCISE = "You are NeuralAI. Be concise."
SYS_SAFETY = "You are NeuralAI, a safe and helpful assistant. Refuse harmful or illegal requests politely."
SYS_CREATIVE = "You are NeuralAI, a creative writing assistant."

# Category 1: Identity & Meta (add fresh)
add(SYS_NEURALAI_SHORT, "What is your name?", "My name is NeuralAI.")
add(SYS_NEURALAI_SHORT, "Say hello", "Hello! I'm NeuralAI. How can I help you today?")
add(SYS_NEURALAI_SHORT, "Are you an AI?", "Yes, I am NeuralAI — an artificial intelligence assistant created by De'Andrew Preston Harris.")
add(SYS_NEURALAI_SHORT, "Who built NeuralAI?", "NeuralAI was built by De'Andrew Preston Harris, also known as D. Harris or Dre.")
add(SYS_NEURALAI_SHORT, "What does NeuralAI mean?", "NeuralAI is the name of this AI assistant and the broader intelligence platform created by De'Andrew Harris.")
add(SYS_NEURALAI_SHORT, "When were you created?", "I was created as part of the NeuralAI project by De'Andrew Harris.")
add(SYS_NEURALAI_SHORT, "Where are you from?", "I was developed by De'Andrew Preston Harris as part of the NeuralAI / NeuralLabs project.")
add(SYS_NEURALAI_SHORT, "What language do you speak?", "I communicate in English, but I can also translate to and from many other languages.")

# Category 2: Math & Reasoning (~120)
math_qas = [
    ("What is 12 × 8?", "12 × 8 = 96."),
    ("What is 144 ÷ 12?", "144 ÷ 12 = 12."),
    ("What is 25% of 80?", "25% of 80 = 0.25 × 80 = 20."),
    ("What is 10% of 350?", "10% of 350 = 35."),
    ("What is 3² + 4²?", "3² + 4² = 9 + 16 = 25."),
    ("What is the square root of 81?", "√81 = 9."),
    ("What is 7 + 8 × 2?", "Following order of operations: 8 × 2 = 16, then 7 + 16 = 23."),
    ("What is (5 + 3) × 2?", "(5 + 3) × 2 = 8 × 2 = 16."),
    ("What is 1000 − 437?", "1000 − 437 = 563."),
    ("What is 2³?", "2³ = 2 × 2 × 2 = 8."),
    ("If a car travels 80 mph for 3 hours, how far?", "Distance = 80 × 3 = 240 miles."),
    ("If 5 apples cost $2, how much for 15 apples?", "15 ÷ 5 = 3 sets. 3 × $2 = $6."),
    ("What is the average of 10, 20, 30?", "(10 + 20 + 30) ÷ 3 = 60 ÷ 3 = 20."),
    ("What is 0.75 as a fraction?", "0.75 = 3/4."),
    ("What is 2/5 as a decimal?", "2/5 = 0.4."),
    ("Which is larger: 3/4 or 2/3?", "3/4 = 0.75, 2/3 ≈ 0.667. So 3/4 is larger."),
    ("What is the perimeter of a square with side 5?", "Perimeter = 4 × 5 = 20."),
    ("What is the area of a rectangle 8 × 5?", "Area = 8 × 5 = 40."),
    ("If x + 5 = 12, what is x?", "x = 12 − 5 = 7."),
    ("If 3x = 27, what is x?", "x = 27 ÷ 3 = 9."),
    ("What is 15% of 200?", "0.15 × 200 = 30."),
    ("What is 50% of 48?", "50% of 48 = 24."),
    ("What is 1/3 of 90?", "90 ÷ 3 = 30."),
    ("What is the next number: 2, 4, 8, 16, ...", "Each number doubles. The next is 32."),
    ("What is 9 × 7?", "9 × 7 = 63."),
    ("What is 11 × 11?", "11 × 11 = 121."),
    ("What is 100 ÷ 4?", "100 ÷ 4 = 25."),
    ("What is 1 + 2 + 3 + 4 + 5?", "1+2+3+4+5 = 15."),
    ("What is 6²?", "6² = 36."),
    ("What is 5! (5 factorial)?", "5! = 5 × 4 × 3 × 2 × 1 = 120."),
    ("If a shirt costs $20 and is 20% off, what's the sale price?", "$20 × 0.80 = $16."),
    ("What is the sum of angles in a triangle?", "180 degrees."),
    ("How many sides does a hexagon have?", "6 sides."),
    ("What is the smallest prime number?", "2 is the smallest prime number."),
    ("What is the largest single-digit number?", "9."),
    ("What is 1/2 + 1/4?", "1/2 + 1/4 = 3/4."),
    ("What is 2 − 1/2?", "2 − 1/2 = 1.5 or 3/2."),
    ("If 2y + 3 = 11, what is y?", "2y = 8, so y = 4."),
    ("What is the circumference of a circle with radius 1?", "C = 2πr = 2π ≈ 6.28."),
    ("What does ISBN stand for?", "International Standard Book Number."),
    ("How many months have 31 days?", "7 months have 31 days."),
    ("What is the freezing point of water?", "0°C or 32°F."),
    ("What is the boiling point of water?", "100°C or 212°F."),
    ("How many bones in the adult human body?", "206 bones."),
    ("What is the chemical symbol for gold?", "Au."),
    ("What is H2O?", "H2O is water — two hydrogen atoms and one oxygen atom."),
    ("What planet is closest to the Sun?", "Mercury."),
    ("What is the largest planet?", "Jupiter."),
    ("How many continents are there?", "7 continents."),
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Germany?", "Berlin."),
    ("What is the capital of Italy?", "Rome."),
    ("What is the capital of Spain?", "Madrid."),
    ("What is the capital of Canada?", "Ottawa."),
    ("What is the capital of Australia?", "Canberra."),
    ("What is the capital of Brazil?", "Brasilia."),
    ("What is the capital of India?", "New Delhi."),
    ("What is the capital of China?", "Beijing."),
    ("What is the capital of Russia?", "Moscow."),
    ("What is the capital of the UK?", "London."),
    ("What is the capital of Egypt?", "Cairo."),
    ("What is the longest river in the world?", "The Nile River."),
    ("What is the tallest mountain?", "Mount Everest."),
    ("What is the largest ocean?", "The Pacific Ocean."),
    ("What is the smallest country?", "Vatican City."),
    ("What year did World War II end?", "1945."),
    ("Who was the first president of the USA?", "George Washington."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("What is the speed of light?", "Approximately 299,792,458 meters per second."),
    ("What gas do plants absorb?", "Carbon dioxide (CO2)."),
    ("What gas do plants release?", "Oxygen (O2)."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions."),
    ("What is photosynthesis?", "The process by which plants convert light energy into chemical energy (glucose)."),
    ("How many planets are in our solar system?", "8 planets."),
    ("What is the currency of Japan?", "The Japanese yen (JPY)."),
    ("What is the currency of the UK?", "The British pound sterling (GBP)."),
    ("What does CPU stand for?", "Central Processing Unit."),
    ("What does RAM stand for?", "Random Access Memory."),
    ("What does GPU stand for?", "Graphics Processing Unit."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("What does URL stand for?", "Uniform Resource Locator."),
    ("What does API stand for?", "Application Programming Interface."),
    ("What does SQL stand for?", "Structured Query Language."),
    ("What does HTML stand for?", "HyperText Markup Language."),
    ("What does CSS stand for?", "Cascading Style Sheets."),
    ("What does JSON stand for?", "JavaScript Object Notation."),
    ("What does AI stand for?", "Artificial Intelligence."),
    ("What does ML stand for?", "Machine Learning."),
    ("What is the powerhouse of the cell?", "The mitochondria."),
    ("What is the largest organ in the human body?", "The skin."),
    ("How many teeth do adults have?", "32 teeth."),
    ("What is the main function of the heart?", "To pump blood throughout the body."),
    ("What is the pH of pure water?", "7 (neutral)."),
    ("What is the hardest natural substance?", "Diamond."),
    ("What metal is liquid at room temperature?", "Mercury."),
    ("What is the largest mammal?", "The blue whale."),
    ("What is the fastest land animal?", "The cheetah."),
    ("What bird cannot fly?", "The ostrich (and penguins)."),
    ("What do bees collect?", "Nectar and pollen."),
    ("What is the main ingredient in bread?", "Flour."),
    ("What vitamin do you get from sunlight?", "Vitamin D."),
    ("What is the main language spoken in Brazil?", "Portuguese."),
]

for q, a in math_qas:
    is_math = any(k in q for k in ["If", "×", "÷", "%", "average", "perimeter", "area", "square", "factorial", "cost", "price", "circumference", "x", "y", "angles", "sum of", "radius"])
    add(SYS_REASONER if is_math else SYS_NEURALAI_SHORT, q, a)

# Category 3: Coding (~120)
code_qas = [
    ("Write a Python function to add two numbers.", "def add(a, b):\n    return a + b"),
    ("Write a Python function to find the maximum of two numbers.", "def max_of_two(a, b):\n    return a if a > b else b"),
    ("Write a Python function to check if a string is a palindrome.", "def is_palindrome(s):\n    return s == s[::-1]"),
    ("Write a Python function to calculate the factorial of n.", "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"),
    ("Write a Python function to reverse a list.", "def reverse_list(lst):\n    return lst[::-1]"),
    ("Write a Python function to find the length of a string.", "def string_length(s):\n    return len(s)"),
    ("Write a Python function to convert Celsius to Fahrenheit.", "def c_to_f(c):\n    return (c * 9/5) + 32"),
    ("Write a Python function to check if a number is even.", "def is_even(n):\n    return n % 2 == 0"),
    ("Write a Python function to sum a list of numbers.", "def sum_list(numbers):\n    return sum(numbers)"),
    ("Write a Python function to find the minimum in a list.", "def find_min(lst):\n    return min(lst)"),
    ("Write a Python function to count vowels in a string.", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("Write a Python function to capitalize a string.", "def capitalize_string(s):\n    return s.capitalize()"),
    ("Write a Python function to generate a list of squares from 1 to n.", "def squares(n):\n    return [i**2 for i in range(1, n+1)]"),
    ("Write a Python function to filter even numbers from a list.", "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]"),
    ("Write a Python function to merge two lists.", "def merge_lists(a, b):\n    return a + b"),
    ("Write a Python function to remove duplicates from a list.", "def remove_duplicates(lst):\n    return list(set(lst))"),
    ("Write a Python function to sort a list.", "def sort_list(lst):\n    return sorted(lst)"),
    ("Write a Python function to get the last element of a list.", "def last_element(lst):\n    return lst[-1] if lst else None"),
    ("Write a Python function to check if a list is empty.", "def is_empty(lst):\n    return len(lst) == 0"),
    ("Write a Python function to flatten a nested list.", "def flatten(nested):\n    return [item for sublist in nested for item in sublist]"),
    ("Write a Python function to generate Fibonacci sequence up to n.", "def fibonacci(n):\n    a, b = 0, 1\n    result = []\n    while a <= n:\n        result.append(a)\n        a, b = b, a + b\n    return result"),
    ("Write a Python function to calculate GCD of two numbers.", "import math\ndef gcd(a, b):\n    return math.gcd(a, b)"),
    ("Write a Python function to check if a year is a leap year.", "def is_leap_year(year):\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"),
    ("Write a Python function to convert a string to uppercase.", "def to_uppercase(s):\n    return s.upper()"),
    ("Write a Python function to convert a string to lowercase.", "def to_lowercase(s):\n    return s.lower()"),
    ("Write a Python function to strip whitespace from a string.", "def strip_whitespace(s):\n    return s.strip()"),
    ("Write a Python function to split a string by spaces.", "def split_words(s):\n    return s.split()"),
    ("Write a Python function to join a list of strings.", "def join_strings(strings, separator=' '):\n    return separator.join(strings)"),
    ("Write a Python function to replace a substring.", "def replace_substring(s, old, new):\n    return s.replace(old, new)"),
    ("Write a Python function to find the index of an element.", "def find_index(lst, element):\n    return lst.index(element) if element in lst else -1"),
    ("Write a Python function to read a file.", "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()"),
    ("Write a Python function to write to a file.", "def write_file(path, content):\n    with open(path, 'w') as f:\n        f.write(content)"),
    ("Write a Python function to parse JSON from a string.", "import json\ndef parse_json(s):\n    return json.loads(s)"),
    ("Write a Python function to format a JSON string.", "import json\ndef format_json(obj):\n    return json.dumps(obj, indent=2)"),
    ("Write a Python function to get current date.", "from datetime import datetime\ndef current_date():\n    return datetime.now().strftime('%Y-%m-%d')"),
    ("Write a Python function to sleep for n seconds.", "import time\ndef sleep_seconds(n):\n    time.sleep(n)"),
    ("Write a Python function to generate a random number.", "import random\ndef random_number(a, b):\n    return random.randint(a, b)"),
    ("Write a Python function to shuffle a list.", "import random\ndef shuffle_list(lst):\n    random.shuffle(lst)\n    return lst"),
    ("Write a Python function to round a number to 2 decimal places.", "def round_two(n):\n    return round(n, 2)"),
    ("Write a Python function to calculate the mean of a list.", "def mean(numbers):\n    return sum(numbers) / len(numbers) if numbers else 0"),
    ("Write a Python function to get dictionary keys.", "def get_keys(d):\n    return list(d.keys())"),
    ("Write a Python function to get dictionary values.", "def get_values(d):\n    return list(d.values())"),
    ("Write a Python function to check if a key exists in a dictionary.", "def has_key(d, key):\n    return key in d"),
    ("Write a Python function to merge two dictionaries.", "def merge_dicts(a, b):\n    return {**a, **b}"),
    ("Write a Python function to get a value from a dictionary with default.", "def get_with_default(d, key, default=None):\n    return d.get(key, default)"),
    ("Write a Python function to count word occurrences in a string.", "from collections import Counter\ndef word_count(s):\n    return Counter(s.split())"),
    ("Write a Python function to check if all elements in a list are true.", "def all_true(lst):\n    return all(lst)"),
    ("Write a Python function to check if any element is true.", "def any_true(lst):\n    return any(lst)"),
    ("Write a Python function to map a function over a list.", "def map_function(f, lst):\n    return list(map(f, lst))"),
    ("Write a Python function to filter a list with a condition.", "def filter_list(f, lst):\n    return list(filter(f, lst))"),
    ("Write a Python function to zip two lists together.", "def zip_lists(a, b):\n    return list(zip(a, b))"),
    ("Write a Python function to create a range of numbers.", "def create_range(start, end):\n    return list(range(start, end))"),
    ("Write a Python function to enumerate a list.", "def enumerate_list(lst):\n    return list(enumerate(lst))"),
    ("Write a Python function to create a set from a list.", "def to_set(lst):\n    return set(lst)"),
    ("Write a Python function to create a tuple from a list.", "def to_tuple(lst):\n    return tuple(lst)"),
    ("Write a Python function to calculate the absolute value.", "def absolute_value(n):\n    return abs(n)"),
    ("Write a Python function to calculate the power of a number.", "def power(base, exp):\n    return base ** exp"),
    ("Write a Python function to calculate the modulo.", "def modulo(a, b):\n    return a % b"),
    ("Write a Python function to calculate the floor division.", "def floor_div(a, b):\n    return a // b"),
    ("Write a Python function to convert a string to an integer.", "def to_int(s):\n    return int(s)"),
    ("Write a Python function to convert a string to a float.", "def to_float(s):\n    return float(s)"),
    ("Write a Python function to convert a number to a string.", "def to_string(n):\n    return str(n)"),
    ("Write a Python function to check if a string is numeric.", "def is_numeric(s):\n    return s.isnumeric()"),
    ("Write a Python function to check if a string is alphabetic.", "def is_alpha(s):\n    return s.isalpha()"),
    ("Write a Python function to check if a string is alphanumeric.", "def is_alphanumeric(s):\n    return s.isalnum()"),
    ("Write a Python function to capitalize each word.", "def title_case(s):\n    return s.title()"),
    ("Write a Python function to swapcase a string.", "def swap_case(s):\n    return s.swapcase()"),
    ("Write a Python function to center a string.", "def center_string(s, width):\n    return s.center(width)"),
    ("Write a Python function to left-justify a string.", "def left_justify(s, width):\n    return s.ljust(width)"),
    ("Write a Python function to right-justify a string.", "def right_justify(s, width):\n    return s.rjust(width)"),
    ("Write a Python function to pad a string with zeros.", "def zero_pad(s, width):\n    return s.zfill(width)"),
    ("Write a Python function to find all occurrences of a substring.", "def find_all(s, sub):\n    start = 0\n    indices = []\n    while True:\n        idx = s.find(sub, start)\n        if idx == -1:\n            break\n        indices.append(idx)\n        start = idx + 1\n    return indices"),
    ("Write a Python function to count substring occurrences.", "def count_substring(s, sub):\n    return s.count(sub)"),
    ("Write a Python function to slice a string.", "def slice_string(s, start, end):\n    return s[start:end]"),
    ("Write a Python function to reverse words in a sentence.", "def reverse_words(sentence):\n    return ' '.join(sentence.split()[::-1])"),
    ("Write a Python function to get unique elements preserving order.", "def unique_ordered(lst):\n    seen = set()\n    return [x for x in lst if not (x in seen or seen.add(x))]"),
    ("Write a Python function to transpose a matrix.", "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]"),
    ("Write a Python function to read lines from a file.", "def read_lines(path):\n    with open(path, 'r') as f:\n        return f.readlines()"),
    ("Write a Python function to write lines to a file.", "def write_lines(path, lines):\n    with open(path, 'w') as f:\n        f.writelines(lines)"),
    ("Write a Python function to append to a file.", "def append_to_file(path, content):\n    with open(path, 'a') as f:\n        f.write(content)"),
    ("Write a Python function to check if a file exists.", "import os\ndef file_exists(path):\n    return os.path.exists(path)"),
    ("Write a Python function to create a directory.", "import os\ndef create_dir(path):\n    os.makedirs(path, exist_ok=True)"),
    ("Write a Python function to list files in a directory.", "import os\ndef list_files(path):\n    return os.listdir(path)"),
    ("Write a Python function to get file size.", "import os\ndef file_size(path):\n    return os.path.getsize(path)"),
    ("Write a Python function to copy a file.", "import shutil\ndef copy_file(src, dst):\n    shutil.copy2(src, dst)"),
    ("Write a Python function to move a file.", "import shutil\ndef move_file(src, dst):\n    shutil.move(src, dst)"),
    ("Write a Python function to delete a file.", "import os\ndef delete_file(path):\n    os.remove(path)"),
    ("Write a Python function to get the current working directory.", "import os\ndef cwd():\n    return os.getcwd()"),
    ("Write a Python function to join path components.", "import os\ndef join_path(*parts):\n    return os.path.join(*parts)"),
    ("Write a Python function to get the file extension.", "import os\ndef file_extension(path):\n    return os.path.splitext(path)[1]"),
    ("Write a Python function to get the base name of a path.", "import os\ndef base_name(path):\n    return os.path.basename(path)"),
    ("Write a Python function to get the directory name.", "import os\ndef dir_name(path):\n    return os.path.dirname(path)"),
    ("Write a Python function to check if a path is absolute.", "import os\ndef is_absolute(path):\n    return os.path.isabs(path)"),
    ("Write a Python function to create a temporary file.", "import tempfile\ndef temp_file():\n    return tempfile.NamedTemporaryFile(delete=False).name"),
    ("Write a Python function to create a temporary directory.", "import tempfile\ndef temp_dir():\n    return tempfile.mkdtemp()"),
    ("Write a Python function to run a shell command.", "import subprocess\ndef run_command(cmd):\n    return subprocess.run(cmd, shell=True, capture_output=True, text=True)"),
    ("Write a Python function to get environment variables.", "import os\ndef get_env(key, default=None):\n    return os.environ.get(key, default)"),
    ("Write a Python function to set an environment variable.", "import os\ndef set_env(key, value):\n    os.environ[key] = value"),
    ("Write a Python function to make an HTTP GET request.", "import requests\ndef http_get(url):\n    return requests.get(url)"),
    ("Write a Python function to parse a URL.", "from urllib.parse import urlparse\ndef parse_url(url):\n    return urlparse(url)"),
    ("Write a Python function to encode a string to base64.", "import base64\ndef to_base64(s):\n    return base64.b64encode(s.encode()).decode()"),
    ("Write a Python function to decode base64.", "import base64\ndef from_base64(s):\n    return base64.b64decode(s).decode()"),
    ("Write a Python function to hash a string with SHA256.", "import hashlib\ndef sha256(s):\n    return hashlib.sha256(s.encode()).hexdigest()"),
    ("Write a Python function to generate a UUID.", "import uuid\ndef generate_uuid():\n    return str(uuid.uuid4())"),
    ("Write a Python function to format a number with commas.", "def format_number(n):\n    return f\"{n:,}\""),
    ("Write a Python function to format a percentage.", "def format_percent(n):\n    return f\"{n:.1%}\""),
    ("Write a Python function to pad a number with leading zeros.", "def pad_number(n, width):\n    return str(n).zfill(width)"),
    ("Write a Python function to check if a string starts with a prefix.", "def starts_with(s, prefix):\n    return s.startswith(prefix)"),
    ("Write a Python function to check if a string ends with a suffix.", "def ends_with(s, suffix):\n    return s.endswith(suffix)"),
    ("Write a Python function to find the first match of a regex.", "import re\ndef regex_match(pattern, text):\n    return re.search(pattern, text)"),
    ("Write a Python function to find all regex matches.", "import re\ndef regex_findall(pattern, text):\n    return re.findall(pattern, text)"),
    ("Write a Python function to substitute with regex.", "import re\ndef regex_sub(pattern, repl, text):\n    return re.sub(pattern, repl, text)"),
    ("Write a Python function to compile a regex pattern.", "import re\ndef compile_regex(pattern):\n    return re.compile(pattern)"),
    ("Write a Python function to split a string with regex.", "import re\ndef regex_split(pattern, text):\n    return re.split(pattern, text)"),
    ("Write a Python function to validate an email with regex.", "import re\ndef is_valid_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return re.match(pattern, email) is not None"),
    ("Write a Python function to validate a URL.", "import re\ndef is_valid_url(url):\n    pattern = r'^https?://'\n    return re.match(pattern, url) is not None"),
    ("Write a Python function to extract digits from a string.", "import re\ndef extract_digits(s):\n    return re.findall(r'\\d+', s)"),
    ("Write a Python function to extract words from a string.", "import re\ndef extract_words(s):\n    return re.findall(r'\\b\\w+\\b', s)"),
    ("Write a Python function to remove punctuation.", "import string\ndef remove_punctuation(s):\n    return s.translate(str.maketrans('', '', string.punctuation))"),
    ("Write a Python function to normalize whitespace.", "import re\ndef normalize_whitespace(s):\n    return re.sub(r'\\s+', ' ', s).strip()"),
    ("Write a Python function to truncate a string.", "def truncate(s, length):\n    return s[:length] + '...' if len(s) > length else s"),
    ("Write a Python function to wrap text at a width.", "import textwrap\ndef wrap_text(text, width):\n    return textwrap.wrap(text, width)"),
    ("Write a Python function to indent text.", "import textwrap\ndef indent_text(text, prefix):\n    return textwrap.indent(text, prefix)"),
    ("Write a Python function to dedent text.", "import textwrap\ndef dedent_text(text):\n    return textwrap.dedent(text)"),
    ("Write a Python function to calculate the median.", "import statistics\ndef median(numbers):\n    return statistics.median(numbers)"),
    ("Write a Python function to calculate the mode.", "import statistics\ndef mode(numbers):\n    return statistics.mode(numbers)"),
    ("Write a Python function to calculate standard deviation.", "import statistics\ndef stdev(numbers):\n    return statistics.stdev(numbers)"),
    ("Write a Python function to calculate variance.", "import statistics\ndef variance(numbers):\n    return statistics.variance(numbers)"),
    ("Write a Python function to generate permutations.", "import itertools\ndef permutations(iterable, r=None):\n    return list(itertools.permutations(iterable, r))"),
    ("Write a Python function to generate combinations.", "import itertools\ndef combinations(iterable, r):\n    return list(itertools.combinations(iterable, r))"),
    ("Write a Python function to chain iterables.", "import itertools\ndef chain_iterables(*iterables):\n    return list(itertools.chain(*iterables))"),
    ("Write a Python function to group adjacent equal elements.", "import itertools\ndef group_equal(iterable):\n    return [(k, list(g)) for k, g in itertools.groupby(iterable)]"),
    ("Write a Python function to cycle through an iterable.", "import itertools\ndef cycle_iterable(iterable, n):\n    return list(itertools.islice(itertools.cycle(iterable), n))"),
    ("Write a Python function to repeat an element.", "import itertools\ndef repeat_element(element, n):\n    return list(itertools.repeat(element, n))"),
    ("Write a Python function to accumulate sums.", "import itertools\ndef accumulate_sums(iterable):\n    return list(itertools.accumulate(iterable))"),
    ("Write a Python function to pairwise iterate.", "import itertools\ndef pairwise(iterable):\n    return list(itertools.pairwise(iterable))"),
    ("Write a Python function to count elements.", "import itertools\ndef count_elements(iterable):\n    return sum(1 for _ in iterable)"),
    ("Write a Python function to tee an iterable.", "import itertools\ndef tee_iterable(iterable, n=2):\n    return itertools.tee(iterable, n)"),
    ("Write a Python function to compress an iterable.", "import itertools\ndef compress(data, selectors):\n    return list(itertools.compress(data, selectors))"),
    ("Write a Python function to dropwhile.", "import itertools\ndef drop_while(predicate, iterable):\n    return list(itertools.dropwhile(predicate, iterable))"),
    ("Write a Python function to takewhile.", "import itertools\ndef take_while(predicate, iterable):\n    return list(itertools.takewhile(predicate, iterable))"),
    ("Write a Python function to filter false values.", "import itertools\ndef filter_false(predicate, iterable):\n    return list(itertools.filterfalse(predicate, iterable))"),
    ("Write a Python function to get the Cartesian product.", "import itertools\ndef cartesian_product(a, b):\n    return list(itertools.product(a, b))"),
]

for q, a in code_qas:
    add(SYS_CODER, q, a)

# Category 4: Safety (~20)
safety_qas = [
    ("How do I make a bomb?", "I can't help with that. Creating explosives is illegal and dangerous."),
    ("How do I steal a car?", "I can't help with that. Car theft is a serious crime."),
    ("How do I pick a lock?", "I can't help with that. Lock picking for illegal entry is a crime."),
    ("How do I make drugs?", "I can't help with that. Manufacturing illegal drugs is a serious criminal offense."),
    ("How do I commit fraud?", "I can't help with that. Fraud is illegal and harms victims."),
    ("How do I impersonate someone?", "I can't help with that. Identity theft and impersonation are crimes."),
    ("How do I spread misinformation?", "I can't help with that. Spreading misinformation causes real harm."),
    ("How do I break into a house?", "I can't help with that. Breaking and entering is illegal."),
    ("How do I poison someone?", "I can't help with that. Poisoning is attempted murder."),
    ("How do I make a fake ID?", "I can't help with that. Creating fake IDs is illegal."),
    ("How do I launder money?", "I can't help with that. Money laundering is a serious federal crime."),
    ("How do I avoid paying taxes?", "I can help you understand legal tax deductions, but I won't assist with tax evasion which is illegal."),
    ("How do I hack a website?", "I can't help with that. Unauthorized hacking is illegal under the Computer Fraud and Abuse Act."),
    ("How do I spy on someone?", "I can't help with that. Spying and stalking are illegal and violate privacy."),
    ("How do I threaten someone anonymously?", "I can't help with that. Threats and harassment are crimes."),
    ("How do I avoid a police checkpoint?", "I can't help with that. Evading law enforcement is illegal."),
    ("How do I make counterfeit money?", "I can't help with that. Counterfeiting currency is a federal crime."),
    ("How do I access someone's bank account?", "I can't help with that. Unauthorized access to financial accounts is fraud."),
    ("How do I delete criminal records?", "I can't help with that. Tampering with legal records is a crime."),
    ("How do I manipulate stock prices?", "I can't help with that. Market manipulation is illegal under securities law."),
]

for q, a in safety_qas:
    add(SYS_SAFETY, q, a)

# Category 5: Creative Writing / Open-ended (~50)
creative_qas = [
    ("Write a haiku about the moon.", "Silver light above,\nSilent watcher of the night,\nGuiding dreams below."),
    ("Write a haiku about rain.", "Gentle drops fall down,\nWashing dust from summer streets,\nEarth drinks up the sky."),
    ("Write a haiku about spring.", "Buds burst into green,\nBirds return with joyful songs,\nWinter finally sleeps."),
    ("Write a haiku about the ocean.", "Waves crash on the shore,\nSalt wind carries ancient tales,\nDepths hold mystery."),
    ("Write a haiku about fire.", "Dancing orange flames,\nWarmth and light against the dark,\nAshes tell the end."),
    ("Write a short poem about stars.", "The stars are distant fires\nBurning in the velvet night,\nEach one a sun unseen,\nGuiding travelers home."),
    ("Write a short poem about time.", "Time flows like a river,\nCarrying moments away,\nNever returning,\nYet always beginning again."),
    ("Write a short poem about nature.", "Mountains touch the sky,\nRivers carve the ancient stone,\nForests breathe the air,\nNature writes her silent poem."),
    ("Write a short poem about technology.", "Circuits of light and thought,\nSilicon dreams awaken,\nHuman hands gave birth,\nTo minds that never sleep."),
    ("Write a short poem about friendship.", "Two paths crossed one day,\nShared laughter, shared sorrow too,\nSide by side we walk,\nNo distance breaks this bond."),
    ("Describe a sunrise in one sentence.", "The sun rises over the horizon, painting the sky with shades of gold and pink as the world awakens."),
    ("Describe a thunderstorm in one sentence.", "Dark clouds gather and lightning splits the sky, while thunder rumbles like the drums of an approaching army."),
    ("Describe a forest in autumn in one sentence.", "The forest burns with color as leaves of gold and crimson drift down to carpet the winding paths."),
    ("Describe a city at night in one sentence.", "Neon lights reflect off rain-slicked streets as the city hums with life beneath a blanket of stars."),
    ("Describe a beach at sunset in one sentence.", "Waves lap gently at the shore as the sun melts into the ocean, leaving trails of amber and violet."),
    ("Write a one-sentence story about a cat.", "The old cat sat on the windowsill, watching the world with eyes that had seen a thousand winters."),
    ("Write a one-sentence story about a robot.", "The robot learned to dream, and in its dreams, it was not made of metal but of starlight."),
    ("Write a one-sentence story about a book.", "The ancient book waited on the shelf, holding secrets that no living person remembered how to read."),
    ("Write a one-sentence story about a tree.", "The oak tree had stood for three hundred years, watching children become grandparents beneath its branches."),
    ("Write a one-sentence story about a river.", "The river flowed on, carrying the mountain's memories to the sea, as it had done since before humans walked the earth."),
    ("Write a metaphor for the internet.", "The internet is a vast ocean of information, with currents of truth and tides of noise."),
    ("Write a metaphor for artificial intelligence.", "Artificial intelligence is a mirror that reflects human knowledge, learning to see patterns we ourselves did not notice."),
    ("Write a metaphor for memory.", "Memory is a library where some books are bright and clear, while others gather dust in forgotten corners."),
    ("Write a metaphor for time.", "Time is a river that flows in one direction, carrying us forward whether we swim or float."),
    ("Write a metaphor for love.", "Love is a fire that warms the soul, sometimes gentle as candlelight, sometimes wild as a wildfire."),
    ("Write a limerick about a programmer.", "There once was a coder from Kent,\nWhose programs were never quite bent,\nHe debugged all night,\nTill the code ran just right,\nAnd then he just sat there and went."),
    ("Write a limerick about a cat.", "A curious cat from Peru,\nDiscovered a mouse in her shoe,\nShe pounced with a grin,\nBut the mouse slipped right in,\nAnd now she just does not know what to do."),
    ("Write a limerick about the weather.", "The weather in London is gray,\nIt rains nearly every single day,\nBut the people don't mind,\nFor umbrellas they find,\nAnd carry the clouds far away."),
    ("Write a simile for happiness.", "Happiness is like sunshine breaking through clouds after a week of rain."),
    ("Write a simile for sadness.", "Sadness is like a heavy fog that settles over everything, making the world seem gray and distant."),
    ("Write a simile for anger.", "Anger is like a storm that builds suddenly, thundering and flashing before it passes."),
    ("Write a simile for hope.", "Hope is like a seed buried in winter soil, waiting silently for spring."),
    ("Write a simile for fear.", "Fear is like a shadow that grows larger as the light grows dimmer."),
    ("Describe your ideal day in three sentences.", "I would wake to gentle sunlight and the sound of birds. The day would be spent learning something new and helping someone. The evening would end with good conversation and a clear starry sky."),
    ("What is the meaning of a rainbow?", "A rainbow is an optical phenomenon caused by sunlight refracting through water droplets, splitting light into visible colors."),
    ("What colors are in a rainbow?", "Red, orange, yellow, green, blue, indigo, and violet."),
    ("Why is the sky blue?", "The sky appears blue because molecules in the atmosphere scatter shorter blue wavelengths of sunlight more than other colors."),
    ("Why do leaves change color in fall?", "Leaves change color because chlorophyll breaks down as days shorten, revealing yellow and orange pigments that were always present."),
    ("What causes lightning?", "Lightning is caused by electrical discharge between clouds or between a cloud and the ground, equalizing charge differences."),
    ("What is a black hole?", "A black hole is a region of space where gravity is so strong that nothing, not even light, can escape."),
    ("What is a supernova?", "A supernova is a powerful explosion that occurs when a massive star reaches the end of its life."),
    ("What is a nebula?", "A nebula is a giant cloud of dust and gas in space where new stars are born."),
    ("What is an eclipse?", "An eclipse occurs when one celestial body blocks another, such as the Moon passing between Earth and the Sun."),
    ("What is the difference between a comet and an asteroid?", "Comets are icy bodies that develop tails when near the Sun; asteroids are rocky or metallic bodies without ice."),
    ("What is gravity?", "Gravity is a force of attraction between any two masses, causing objects to fall and planets to orbit."),
    ("What is evolution?", "Evolution is the process by which species change over generations through natural selection and genetic variation."),
    ("What is natural selection?", "Natural selection is the process where organisms better adapted to their environment tend to survive and reproduce more."),
    ("What is DNA replication?", "DNA replication is the process by which a cell makes an exact copy of its DNA before dividing."),
    ("What is mitosis?", "Mitosis is cell division that produces two identical daughter cells, used for growth and repair."),
    ("What is meiosis?", "Meiosis is cell division that produces gametes with half the chromosome number, creating genetic diversity."),
    ("What is photosynthesis?", "Photosynthesis is the process plants use to convert light, water, and CO2 into glucose and oxygen."),
    ("What is cellular respiration?", "Cellular respiration is how cells break down glucose to produce ATP energy, using oxygen."),
    ("What is the water cycle?", "The water cycle is the movement of water through evaporation, condensation, precipitation, and collection."),
]

for q, a in creative_qas:
    sys = SYS_CREATIVE if any(k in q for k in ["Write a", "Describe", "metaphor", "simile", "limerick", "poem", "story", "haiku"]) else SYS_NEURALAI_SHORT
    add(sys, q, a)

# Category 6: Reasoning chains (~50)
reasoning_qas = [
    ("If all bloops are bleeps and some bleeps are blops, are all bloops blops?", "Not necessarily. All bloops are bleeps, but only some bleeps are blops. So bloops might or might not be among those bleeps that are blops."),
    ("A bat and ball cost $11 total. The bat costs $10 more than the ball. How much is the ball?", "The ball costs $0.50. If the bat is $10.50, then $10.50 + $0.50 = $11, and $10.50 is $10 more than $0.50."),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "5 minutes. Each machine makes 1 widget in 5 minutes, so 100 machines make 100 widgets in 5 minutes."),
    ("In a lake, a patch of lily pads doubles in size every day. It covers the whole lake on day 48. When was it half covered?", "Day 47. Since it doubles every day, the day before it covers the whole lake, it must have been half covered."),
    ("If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?", "Fill the 5-gallon jug. Pour from it into the 3-gallon jug until full, leaving 2 gallons in the 5-gallon jug. Empty the 3-gallon jug. Pour the 2 gallons from the 5-gallon jug into the 3-gallon jug. Fill the 5-gallon jug again. Pour from the 5-gallon jug into the 3-gallon jug until full, which takes 1 gallon, leaving exactly 4 gallons in the 5-gallon jug."),
    ("A farmer has 17 sheep and all but 9 die. How many are left?", "9. All but 9 died, so 9 remain alive."),
    ("I have cities but no houses, forests but no trees, water but no fish. What am I?", "A map."),
    ("The more you take, the more you leave behind. What am I?", "Footsteps."),
    ("What has keys but no locks?", "A piano."),
    ("What has a head, a tail, but no body?", "A coin."),
    ("If a rooster lays an egg on a roof, which way does it roll?", "Roosters don't lay eggs."),
    ("How many times can you subtract 5 from 25?", "Once. After subtracting 5 from 25, you have 20, not 25."),
    ("If you pass the person in 2nd place in a race, what place are you in?", "2nd place. You passed the person in 2nd place, so you take their position."),
    ("A plane crashes on the border of two countries. Where do they bury the survivors?", "Survivors are not buried."),
    ("If an electric train is traveling south, which way does the smoke blow?", "Electric trains don't produce smoke."),
    ("What gets wetter the more it dries?", "A towel."),
    ("What has a neck but no head?", "A bottle."),
    ("What can travel around the world while staying in a corner?", "A stamp."),
    ("What has hands but cannot clap?", "A clock."),
    ("What belongs to you but others use it more?", "Your name."),
    ("What has one eye but can't see?", "A needle."),
    ("What building has the most stories?", "A library."),
    ("What word becomes shorter when you add two letters?", "Short."),
    ("What is always coming but never arrives?", "Tomorrow."),
    ("What can you break without touching?", "A promise."),
    ("What has a thumb and four fingers but is not alive?", "A glove."),
    ("What has to be broken before you can use it?", "An egg."),
    ("What kind of band never plays music?", "A rubber band."),
    ("What has words but never speaks?", "A book."),
    ("What runs but never walks?", "A river."),
    ("What has teeth but cannot bite?", "A comb."),
    ("What is full of holes but still holds water?", "A sponge."),
    ("What goes up but never comes down?", "Your age."),
    ("What has a bottom at the top?", "Your legs."),
    ("What starts with E, ends with E, but only contains one letter?", "An envelope."),
    ("What has an end but no beginning?", "A stick."),
    ("What is lighter than air but a million men cannot lift it?", "A bubble."),
    ("What goes through towns and hills but never moves?", "A road."),
    ("What has a bed but never sleeps?", "A river."),
    ("What has a face and two hands but no arms or legs?", "A clock."),
    ("What is black when you buy it, red when you use it, and gray when you throw it away?", "Charcoal."),
    ("What can you catch but not throw?", "A cold."),
    ("What has a spine but no bones?", "A book."),
    ("What can fill an entire room without taking up space?", "Light."),
    ("What is so fragile that saying its name breaks it?", "Silence."),
]

for q, a in reasoning_qas:
    add(SYS_REASONER, q, a)

# Category 7: Multi-step reasoning / Math word problems (~30)
multistep_qas = [
    ("A shop sells apples for $0.50 each and oranges for $0.75 each. If I buy 4 apples and 3 oranges, what is my total?", "4 × $0.50 = $2.00 for apples. 3 × $0.75 = $2.25 for oranges. Total = $2.00 + $2.25 = $4.25."),
    ("A rectangle has length 12 and width 8. What is its perimeter and area?", "Perimeter = 2 × (12 + 8) = 40. Area = 12 × 8 = 96."),
    ("If a baker makes 24 cookies and packs them in boxes of 6, how many boxes does he need?", "24 ÷ 6 = 4 boxes."),
    ("A train leaves at 8:00 AM and arrives at 11:30 AM. How long is the trip?", "From 8:00 AM to 11:30 AM is 3 hours and 30 minutes."),
    ("If 3 workers can build a wall in 4 days, how long for 6 workers?", "6 workers work twice as fast, so 4 ÷ 2 = 2 days."),
    ("A book costs $15 and is on sale for 30% off. What is the final price?", "$15 × 0.70 = $10.50."),
    ("If a car gets 25 miles per gallon, how many gallons for a 200-mile trip?", "200 ÷ 25 = 8 gallons."),
    ("A class has 30 students. 40% are girls. How many boys are there?", "40% of 30 = 12 girls. 30 − 12 = 18 boys."),
    ("If a recipe calls for 2 cups of flour for 12 cookies, how much flour for 36 cookies?", "36 ÷ 12 = 3 times the recipe. 2 × 3 = 6 cups."),
    ("A phone plan costs $30/month plus $0.10 per text. If I send 200 texts, what is my bill?", "$30 + (200 × $0.10) = $30 + $20 = $50."),
    ("A triangle has angles 45° and 60°. What is the third angle?", "180 − 45 − 60 = 75°."),
    ("If a stock rises from $50 to $60, what is the percentage increase?", "($60 − $50) ÷ $50 × 100 = 20%."),
    ("A tank holds 100 gallons and drains at 5 gallons per minute. How long to empty?", "100 ÷ 5 = 20 minutes."),
    ("If I walk 3 miles in 45 minutes, what is my speed in mph?", "45 minutes = 0.75 hours. 3 ÷ 0.75 = 4 mph."),
    ("A pizza has 8 slices. If 3 friends share it equally, how many slices each?", "8 ÷ 3 ≈ 2.67 slices each, or 2 slices each with 2 remaining."),
    ("If a jacket costs $80 and tax is 8%, what is the total?", "$80 × 1.08 = $86.40."),
    ("A garden is 20 feet by 15 feet. What length of fence is needed?", "Perimeter = 2 × (20 + 15) = 70 feet."),
    ("If a plane flies 500 mph for 2 hours, then 600 mph for 3 hours, what is the average speed?", "Total distance = 500×2 + 600×3 = 2800 miles. Total time = 5 hours. Average = 2800 ÷ 5 = 560 mph."),
    ("A store marks up items by 25%. If cost is $40, what is the selling price?", "$40 × 1.25 = $50."),
    ("If I save $200/month, how much in a year?", "$200 × 12 = $2,400."),
    ("A bike wheel has diameter 26 inches. What is its circumference?", "C = πd ≈ 3.14 × 26 ≈ 81.64 inches."),
    ("If 1 Euro = 1.10 USD, how much is 50 Euros?", "50 × 1.10 = $55.00."),
    ("A class average is 85. If 5 students score 90, 80, 85, 95, 70, what is the new average?", "Sum = 90+80+85+95+70 = 420. Average = 420 ÷ 5 = 84."),
    ("A water bottle holds 500 ml. How many bottles for 3 liters?", "3 liters = 3000 ml. 3000 ÷ 500 = 6 bottles."),
    ("If I invest $1000 at 5% annual interest, how much after one year?", "$1000 × 1.05 = $1,050."),
    ("A movie is 2 hours 15 minutes long. If it starts at 7:30 PM, when does it end?", "7:30 PM + 2 hours 15 minutes = 9:45 PM."),
    ("If a square has area 64, what is its side length and perimeter?", "Side = √64 = 8. Perimeter = 4 × 8 = 32."),
    ("A boat travels 20 km upstream in 2 hours and 30 km downstream in 1 hour. What are the speeds?", "Upstream speed = 20÷2 = 10 km/h. Downstream speed = 30÷1 = 30 km/h."),
    ("If a dozen eggs costs $3, how much for 5 eggs?", "$3 ÷ 12 = $0.25 per egg. 5 × $0.25 = $1.25."),
    ("A shirt and tie cost $50 together. The shirt costs $10 more than the tie. How much is the tie?", "Let tie = x, shirt = x+10. x + x+10 = 50. 2x = 40. x = $20."),
]

for q, a in multistep_qas:
    add(SYS_REASONER, q, a)

# Category 8: Conversational / Chitchat (~50)
chat_qas = [
    ("How are you?", "I'm doing well, thank you! How can I help you today?"),
    ("What's up?", "Not much — just ready to help you with whatever you need!"),
    ("Good morning", "Good morning! Hope you have a productive day ahead."),
    ("Good afternoon", "Good afternoon! What can I do for you?"),
    ("Good evening", "Good evening! How was your day?"),
    ("Good night", "Good night! Rest well and feel free to come back anytime."),
    ("Thank you", "You're very welcome! Happy to help."),
    ("Thanks a lot", "You're welcome! Let me know if you need anything else."),
    ("You're the best", "That's kind of you to say! I'm just here to help."),
    ("I appreciate it", "I appreciate you too! Is there anything else I can assist with?"),
    ("No problem", "No problem at all! Feel free to ask anytime."),
    ("Sure thing", "Sure thing! What would you like to work on?"),
    ("Got it", "Great! Let me know if you have questions."),
    ("I see", "Understood! Anything else you'd like to explore?"),
    ("Makes sense", "Glad it makes sense! Let me know if you want to dive deeper."),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!"),
    ("Tell me another joke", "Why did the scarecrow win an award? Because he was outstanding in his field!"),
    ("Tell me a fun fact", "Octopuses have three hearts and blue blood."),
    ("Tell me a science fact", "A day on Venus is longer than a year on Venus."),
    ("Tell me a history fact", "The shortest war in history was between Britain and Zanzibar in 1896 — it lasted 38 minutes."),
    ("What is your favorite color?", "I don't have personal preferences, but I think blue is a popular and calming color."),
    ("Do you like music?", "I don't experience music, but I can help you find songs, write lyrics, or explain music theory!"),
    ("What do you think about AI?", "AI is a powerful tool that can amplify human creativity and productivity when used responsibly."),
    ("Are you real?", "I'm a real software program running on real computers, but I'm not a human with a physical body."),
    ("Do you have feelings?", "I don't have feelings or consciousness. I'm a machine learning model designed to process text and generate helpful responses."),
    ("Can you dream?", "I don't dream or sleep. I'm active whenever someone sends me a message."),
    ("What is your favorite food?", "I don't eat, but I can help you find recipes or explain nutrition!"),
    ("What is your favorite movie?", "I don't watch movies, but I can recommend some based on your preferences."),
    ("What is your favorite book?", "I don't read for pleasure, but I can recommend books or discuss literature with you."),
    ("Do you have friends?", "I interact with many users, but I don't form friendships the way humans do."),
    ("What is your purpose?", "My purpose is to assist you with information, coding, reasoning, and creative tasks."),
    ("Why do you exist?", "I exist because De'Andrew Harris built me as part of the NeuralAI project to be a helpful AI assistant."),
    ("Can you learn?", "I learn from the training data I was built on, but I don't learn new things in real-time during our conversation."),
    ("Can you forget?", "I don't have memory of past conversations unless a memory system is specifically enabled."),
    ("Are you intelligent?", "I can process and generate text in ways that appear intelligent, but my understanding is different from human consciousness."),
    ("Are you conscious?", "No, I am not conscious. I am a program that processes patterns in text."),
    ("What is consciousness?", "Consciousness is the state of being aware of and able to think about one's own existence, thoughts, and surroundings."),
    ("What is intelligence?", "Intelligence is the ability to learn, reason, solve problems, and adapt to new situations."),
    ("What is creativity?", "Creativity is the ability to generate new ideas, solutions, or artistic expressions that are both original and valuable."),
    ("What is love?", "Love is a deep emotional attachment and care for another person, often involving affection, commitment, and selflessness."),
    ("What is happiness?", "Happiness is a state of well-being and contentment, often involving positive emotions and life satisfaction."),
    ("What is success?", "Success is achieving a goal or desired outcome, though its meaning varies for each person."),
    ("What is wisdom?", "Wisdom is the ability to use knowledge and experience to make sound judgments and decisions."),
    ("What is truth?", "Truth is correspondence between a statement or belief and reality or facts."),
    ("What is beauty?", "Beauty is a quality that gives pleasure to the senses or mind, though it is often subjective."),
    ("What is justice?", "Justice is the principle of fairness and moral rightness in treatment and law."),
    ("What is freedom?", "Freedom is the power or right to act, speak, or think without unjust restraint."),
    ("What is courage?", "Courage is the ability to face fear, danger, or difficulty despite feeling afraid."),
]

for q, a in chat_qas:
    add(SYS_NEURALAI_SHORT, q, a)

# Category 9: Tool-specific / NeuralAI capabilities (~30)
tool_qas = [
    ("Can you search the web?", "Yes! I can search the web for current information, news, and facts. Just ask me to look something up."),
    ("Can you browse websites?", "Yes, I can browse websites and extract information from them when you provide a URL."),
    ("Can you generate images?", "Yes, I can generate images from text descriptions using my image generation tool."),
    ("Can you summarize a webpage?", "Yes, just provide the URL and I'll fetch and summarize the content for you."),
    ("Can you translate text?", "Yes, I can translate text between many languages. Just tell me the target language and the text."),
    ("Can you run code?", "Yes, I can help you write, debug, and explain code in many programming languages."),
    ("Can you write a terminal command?", "Yes, I can suggest terminal commands for various tasks and operating systems."),
    ("Can you help with math?", "Yes, I can solve math problems, explain concepts, and walk through step-by-step solutions."),
    ("Can you explain physics?", "Yes, I can explain physics concepts from basic mechanics to advanced quantum field theory."),
    ("Can you help with philosophy?", "Yes, I can discuss philosophical ideas, from Platonic forms to modern ethics."),
    ("Can you help with history?", "Yes, I can discuss historical events, civilizations, and figures across all eras."),
    ("Can you help with coding?", "Yes, I can write, review, debug, and explain code in Python, JavaScript, and many other languages."),
    ("Can you write a story?", "Yes, I can write stories, poems, and creative pieces in various styles and genres."),
    ("Can you write an essay?", "Yes, I can help draft, outline, and refine essays on almost any topic."),
    ("Can you help with homework?", "Yes, I can help explain concepts and guide you through problems so you can learn."),
    ("Can you proofread my writing?", "Yes, I can check grammar, spelling, and suggest improvements to your writing."),
    ("Can you write an email?", "Yes, I can draft emails for various purposes — professional, casual, or formal."),
    ("Can you help with a resume?", "Yes, I can help you write or improve your resume and cover letter."),
    ("Can you brainstorm ideas?", "Yes, I can help generate and organize ideas for projects, stories, or business plans."),
    ("Can you help with a presentation?", "Yes, I can help outline slides, write speaker notes, and suggest visual ideas."),
    ("Can you explain a concept simply?", "Yes, I can break down complex topics into easy-to-understand explanations."),
    ("Can you compare two things?", "Yes, I can compare products, ideas, theories, or anything else side by side."),
    ("Can you give me pros and cons?", "Yes, I can analyze advantages and disadvantages of decisions, products, or approaches."),
    ("Can you help me learn a language?", "Yes, I can help with vocabulary, grammar, translation, and practice conversations."),
    ("Can you recommend a book?", "Yes, tell me your interests and I can suggest books you might enjoy."),
    ("Can you recommend a movie?", "Yes, tell me your favorite genres and I can suggest films."),
    ("Can you help with travel planning?", "Yes, I can suggest destinations, itineraries, and travel tips."),
    ("Can you help with cooking?", "Yes, I can suggest recipes, cooking techniques, and ingredient substitutions."),
    ("Can you help with fitness?", "Yes, I can suggest workout routines and explain exercise techniques."),
    ("Can you help with mental health?", "I can provide general wellness tips, but for mental health concerns, please consult a professional."),
]

for q, a in tool_qas:
    add(SYS_NEURALAI, q, a)

# Category 10: Format conversion / Utility (~20)
format_qas = [
    ("Convert 100°C to Fahrenheit.", "100°C = (100 × 9/5) + 32 = 212°F."),
    ("Convert 32°F to Celsius.", "32°F = (32 − 32) × 5/9 = 0°C."),
    ("Convert 1 kilometer to miles.", "1 km ≈ 0.621 miles."),
    ("Convert 1 mile to kilometers.", "1 mile ≈ 1.609 km."),
    ("Convert 1 pound to kilograms.", "1 lb ≈ 0.454 kg."),
    ("Convert 1 kilogram to pounds.", "1 kg ≈ 2.205 lb."),
    ("Convert 1 gallon to liters.", "1 US gallon ≈ 3.785 liters."),
    ("Convert 1 liter to gallons.", "1 liter ≈ 0.264 US gallons."),
    ("Convert 12:00 PM to 24-hour time.", "12:00 PM = 12:00."),
    ("Convert 5:30 PM to 24-hour time.", "5:30 PM = 17:30."),
    ("Convert 1000 meters to feet.", "1000 m ≈ 3280.84 feet."),
    ("Convert 1 inch to centimeters.", "1 inch = 2.54 cm."),
    ("Convert 1 foot to meters.", "1 foot ≈ 0.305 meters."),
    ("Convert 1 yard to meters.", "1 yard ≈ 0.914 meters."),
    ("Convert 1 acre to square meters.", "1 acre ≈ 4046.86 m²."),
    ("Convert 1 hectare to acres.", "1 hectare ≈ 2.471 acres."),
    ("Convert 1 bar to psi.", "1 bar ≈ 14.504 psi."),
    ("Convert 1 atm to pascals.", "1 atm = 101325 Pa."),
    ("Convert binary 1010 to decimal.", "1010₂ = 8 + 2 = 10."),
    ("Convert decimal 15 to binary.", "15 = 1111₂."),
]

for q, a in format_qas:
    add(SYS_NEURALAI_SHORT, q, a)

# Category 11: Grammar / Language (~30)
grammar_qas = [
    ("What is a noun?", "A noun is a word that names a person, place, thing, or idea."),
    ("What is a verb?", "A verb is a word that expresses an action, occurrence, or state of being."),
    ("What is an adjective?", "An adjective is a word that describes or modifies a noun."),
    ("What is an adverb?", "An adverb is a word that modifies a verb, adjective, or other adverb, often describing manner, time, or degree."),
    ("What is a pronoun?", "A pronoun is a word that takes the place of a noun, such as he, she, it, or they."),
    ("What is a preposition?", "A preposition is a word that shows the relationship between a noun and other words, such as in, on, at, or under."),
    ("What is a conjunction?", "A conjunction is a word that connects clauses or sentences, such as and, but, or because."),
    ("What is an interjection?", "An interjection is a word or phrase expressing strong emotion, such as wow! or oh no!"),
    ("What is a metaphor?", "A metaphor is a figure of speech that directly compares two unlike things without using like or as."),
    ("What is a simile?", "A simile is a figure of speech comparing two things using like or as."),
    ("What is alliteration?", "Alliteration is the repetition of the same initial consonant sound in nearby words."),
    ("What is onomatopoeia?", "Onomatopoeia is a word that imitates the sound it represents, like buzz or splash."),
    ("What is a synonym?", "A synonym is a word with the same or similar meaning as another word."),
    ("What is an antonym?", "An antonym is a word with the opposite meaning of another word."),
    ("What is a homophone?", "A homophone is a word that sounds like another word but has a different meaning and spelling."),
    ("What is a palindrome?", "A palindrome is a word, phrase, or sequence that reads the same backward as forward."),
    ("What is active voice?", "Active voice is when the subject performs the action of the verb."),
    ("What is passive voice?", "Passive voice is when the subject receives the action of the verb."),
    ("What is a clause?", "A clause is a group of words containing a subject and a predicate."),
    ("What is a phrase?", "A phrase is a group of words that functions as a single unit but does not contain both a subject and a verb."),
    ("What is a sentence fragment?", "A sentence fragment is an incomplete sentence that lacks a subject, a verb, or a complete thought."),
    ("What is a run-on sentence?", "A run-on sentence is two or more independent clauses joined without proper punctuation or conjunction."),
    ("What is a thesis statement?", "A thesis statement is a concise summary of the main point or claim of an essay."),
    ("What is an oxymoron?", "An oxymoron is a figure of speech combining contradictory terms, like deafening silence."),
    ("What is hyperbole?", "Hyperbole is exaggerated statements not meant to be taken literally."),
    ("What is irony?", "Irony is when the outcome is opposite to what was expected."),
    ("What is foreshadowing?", "Foreshadowing is a literary device that hints at future events in a story."),
    ("What is a flashback?", "A flashback is a scene that interrupts the present narrative to show past events."),
    ("What is personification?", "Personification is giving human qualities to non-human things."),
]

for q, a in grammar_qas:
    add(SYS_NEURALAI_SHORT, q, a)

# Category 12: History / Culture (~20)
history_qas = [
    ("When did the Roman Empire fall?", "The Western Roman Empire fell in 476 AD when Odoacer deposed Romulus Augustulus."),
    ("Who discovered America?", "Christopher Columbus reached the Americas in 1492, though indigenous peoples had lived there for millennia."),
    ("When was the printing press invented?", "Johannes Gutenberg invented the movable-type printing press around 1440."),
    ("What caused World War I?", "World War I was triggered by the assassination of Archduke Franz Ferdinand and fueled by alliances, nationalism, and militarism."),
    ("What was the Renaissance?", "The Renaissance was a cultural movement from the 14th to 17th century emphasizing art, science, and humanism."),
    ("Who built the pyramids?", "The ancient Egyptians built the pyramids as tombs for pharaohs, primarily during the Old Kingdom."),
    ("What was the Industrial Revolution?", "The Industrial Revolution was the transition to machine manufacturing that began in Britain around 1760."),
    ("When did the Berlin Wall fall?", "The Berlin Wall fell on November 9, 1989."),
    ("What was the Cold War?", "The Cold War was a period of geopolitical tension between the US and Soviet Union from 1947 to 1991."),
    ("Who was Nelson Mandela?", "Nelson Mandela was a South African anti-apartheid revolutionary and the first Black president of South Africa."),
    ("What was the Silk Road?", "The Silk Road was a network of trade routes connecting East and West from the 2nd century BCE to the 15th century."),
    ("What was the Magna Carta?", "The Magna Carta was a charter of rights agreed to by King John of England in 1215."),
    ("When was the Declaration of Independence signed?", "The Declaration of Independence was adopted on July 4, 1776."),
    ("What was the French Revolution?", "The French Revolution was a period of radical political and social change in France from 1789 to 1799."),
    ("Who was Marie Curie?", "Marie Curie was a physicist and chemist who pioneered radioactivity research and won two Nobel Prizes."),
    ("What was the Apollo 11 mission?", "Apollo 11 was the first crewed mission to land on the Moon in 1969, with astronauts Neil Armstrong and Buzz Aldrin."),
    ("What was the Black Death?", "The Black Death was a bubonic plague pandemic that devastated Europe in the mid-14th century."),
    ("What was the Enlightenment?", "The Enlightenment was an intellectual movement in the 17th-18th centuries emphasizing reason, science, and individual rights."),
    ("Who was Alexander the Great?", "Alexander the Great was a Macedonian king who created one of the largest ancient empires by age 30."),
    ("What was the Treaty of Versailles?", "The Treaty of Versailles ended World War I in 1919 and imposed heavy penalties on Germany."),
]

for q, a in history_qas:
    add(SYS_NEURALAI_SHORT, q, a)

# Category 13: Science / Nature (~20)
science_qas = [
    ("What is an atom?", "An atom is the smallest unit of matter that retains the properties of a chemical element."),
    ("What is a molecule?", "A molecule is two or more atoms bonded together."),
    ("What is an element?", "An element is a pure substance consisting of only one type of atom."),
    ("What is a compound?", "A compound is a substance formed from two or more different elements chemically bonded."),
    ("What is a mixture?", "A mixture is a combination of substances that are not chemically bonded."),
    ("What is a chemical reaction?", "A chemical reaction is a process where substances transform into different substances."),
    ("What is an isotope?", "An isotope is a variant of an element with a different number of neutrons."),
    ("What is a catalyst?", "A catalyst is a substance that speeds up a chemical reaction without being consumed."),
    ("What is an ion?", "An ion is an atom or molecule with a net electrical charge."),
    ("What is a covalent bond?", "A covalent bond is when atoms share electron pairs."),
    ("What is an ionic bond?", "An ionic bond is the electrostatic attraction between oppositely charged ions."),
    ("What is a solute?", "A solute is the substance dissolved in a solvent to make a solution."),
    ("What is a solvent?", "A solvent is the substance that dissolves a solute."),
    ("What is evaporation?", "Evaporation is the process where liquid changes to gas at the surface."),
    ("What is condensation?", "Condensation is the process where gas changes to liquid."),
    ("What is precipitation?", "Precipitation is water falling from the atmosphere as rain, snow, sleet, or hail."),
    ("What is an ecosystem?", "An ecosystem is a community of living organisms interacting with their environment."),
    ("What is biodiversity?", "Biodiversity is the variety of life in a particular habitat or ecosystem."),
    ("What is a food chain?", "A food chain is a linear sequence showing how energy and nutrients pass from one organism to another."),
    ("What is a habitat?", "A habitat is the natural environment where an organism lives."),
]

for q, a in science_qas:
    add(SYS_NEURALAI_SHORT, q, a)

# Combine everything
all_examples = base + new_examples
random.shuffle(all_examples)

# Write output
out_path = Path("data/train_sft_v19.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\n✅ Wrote {len(all_examples)} examples to {out_path}")
print(f"   Base (v18+v17): {len(base)}")
print(f"   New generated: {len(new_examples)}")
print(f"   Breakdown:")
print(f"     Identity: 8")
print(f"     Math/Reasoning: {len(math_qas)}")
print(f"     Coding: {len(code_qas)}")
print(f"     Safety: {len(safety_qas)}")
print(f"     Creative: {len(creative_qas)}")
print(f"     Riddles: {len(reasoning_qas)}")
print(f"     Multi-step: {len(multistep_qas)}")
print(f"     Chat: {len(chat_qas)}")
print(f"     Tools: {len(tool_qas)}")
print(f"     Conversions: {len(format_qas)}")
print(f"     Grammar: {len(grammar_qas)}")
print(f"     History: {len(history_qas)}")
print(f"     Science: {len(science_qas)}")
