from src.optimizer import optimize_prompt
import csv

with open('data/test_prompts.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for row in rows:
    result = optimize_prompt(row['prompt'])
    aggressive = result['strategies']['aggressive']['candidate']
    print(f"ID {row['id']} ({row['category']}):")
    print(' ', aggressive)
    print()