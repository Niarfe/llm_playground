"""
Fixture script for 08. Deliberately the longest file in this directory,
so the agent's answer has an unambiguous right answer to find.
"""

total = 0
numbers = range(1, 11)

for n in numbers:
    total += n
    print(f"  running total after {n}: {total}")

print(f"The sum of 1 through 10 is {total}.")
