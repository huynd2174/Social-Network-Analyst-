# -*- coding: utf-8 -*-
"""Final test"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from chatbot.chatbot import KpopChatbot

chatbot = KpopChatbot(verbose=False, llm_model=None)
query = "HyunA và Yoo Jeong-yeon có cùng công ty quản lý không?"
extracted = chatbot._extract_entities_for_membership(query, expected_labels={'Artist', 'Group'})

print(f"Extracted entities ({len(extracted)}):")
for i, entity in enumerate(extracted, 1):
    base_name = chatbot._normalize_entity_name(entity).lower()
    print(f"  {i}. {entity} (base: {base_name})")

yoo_jeong_yeon = [e for e in extracted if 'yoo jeong-yeon' in chatbot._normalize_entity_name(e).lower()]
yoo = [e for e in extracted if chatbot._normalize_entity_name(e).lower() == 'yoo']
jeongyeon = [e for e in extracted if chatbot._normalize_entity_name(e).lower() == 'jeongyeon']

print(f"\nResults:")
print(f"  Yoo Jeong-yeon: {len(yoo_jeong_yeon) > 0}")
print(f"  Yoo (single): {len(yoo) > 0}")
print(f"  Jeongyeon: {len(jeongyeon) > 0}")

if len(yoo_jeong_yeon) > 0 and len(yoo) == 0 and len(jeongyeon) == 0:
    print("\n[PASS] Test passed!")
    sys.exit(0)
else:
    print("\n[FAIL] Test failed!")
    sys.exit(1)

