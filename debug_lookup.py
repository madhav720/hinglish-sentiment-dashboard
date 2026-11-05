"""Check what the actual Hindi translation text looks like."""
import pandas as pd

trans_df = pd.read_csv('data/processed/translations.csv', encoding='utf-8')

# Print first 5 Hindi translations to a file to check
with open('hindi_check.txt', 'w', encoding='utf-8') as f:
    for i, (_, r) in enumerate(trans_df.head(10).iterrows()):
        f.write(f"ID: {r['id']}\n")
        f.write(f"Hinglish: {r['text_hinglish']}\n")
        f.write(f"Hindi:    {r['text_hindi']}\n")
        f.write(f"English:  {r['text_english']}\n")
        f.write("-" * 60 + "\n")

print("Written to hindi_check.txt")

# Also check app.py's Streamlit cache_data - is there a st.cache issue?
# Check if build_unified_index is called at module level (outside of page routing)
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all calls to build_unified_index
import re
calls = [(m.start(), content[max(0,m.start()-50):m.start()+80]) for m in re.finditer(r'build_unified_index', content)]
print(f"\nFound {len(calls)} references to build_unified_index in app.py:")
for pos, ctx in calls:
    print(f"  @pos {pos}: ...{ctx.strip()[:100]}...")

# Also check: is the @st.cache_data decorator on build_unified_index causing resolve_path to fail?
# The function uses resolve_path which depends on BASE_DIR - check if that's defined before the function
base_dir_pos = content.find('BASE_DIR')
func_pos = content.find('def build_unified_index')
print(f"\nBASE_DIR defined at char {base_dir_pos}, build_unified_index at char {func_pos}")
print("BASE_DIR defined BEFORE function:", base_dir_pos < func_pos)
