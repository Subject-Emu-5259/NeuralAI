import sys
src = open(sys.argv[1]).read()
pairs = {')': '(', ']': '[', '}': '{'}
stack = []
in_str = None
esc = False
for ch in src:
    if in_str:
        if esc:
            esc = False
        elif ch == '\\':
            esc = True
        elif ch == in_str:
            in_str = None
        continue
    if ch in '"\'`':
        in_str = ch
        continue
    if ch in '([{':
        stack.append(ch)
    elif ch in ')]}':
        if not stack or stack[-1] != pairs[ch]:
            print("UNBALANCED at", repr(ch))
            sys.exit(1)
        stack.pop()
if stack:
    print("LEFTOPEN", stack)
    sys.exit(1)
print("BALANCED, lines:", src.count('\n') + 1)
