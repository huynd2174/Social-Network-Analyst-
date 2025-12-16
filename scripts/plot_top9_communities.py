"""Vẽ biểu đồ Top 9 cộng đồng lớn nhất trong mạng K-pop"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Data cho top 9 cộng đồng với các node đại diện
communities = [
    ('Pledis Ent.', 198, 'Company', 'After School, SF9, NU\'EST'),
    ('JYP Ent.', 188, 'Company', 'TWICE, Stray Kids, ITZY'),
    ('Cube Ent.', 162, 'Company', 'BTOB, (G)I-DLE, Pentagon'),
    ("Girls' Gen.", 119, 'Group', 'Taeyeon, Tiffany, Seohyun'),
    ('SM Ent.', 107, 'Company', 'EXO, NCT, Red Velvet'),
    ('YG Ent.', 104, 'Company', 'BIGBANG, 2NE1, WINNER'),
    ('BLACKPINK', 85, 'Group', 'Jennie, Lisa, Rose, Jisoo'),
    ('HYBE', 80, 'Company', 'BTS, TXT, ENHYPEN'),
    ('Big Bang', 78, 'Group', 'G-Dragon, Taeyang, T.O.P'),
]

names = [c[0] for c in communities]
sizes = [c[1] for c in communities]
types = [c[2] for c in communities]
representatives = [c[3] for c in communities]

# Colors based on type
colors = ['#3498db' if t == 'Company' else '#e74c3c' for t in types]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Horizontal bar chart
bars = ax.barh(range(len(names)), sizes, color=colors, edgecolor='white', linewidth=1.5, height=0.7)

# Add value labels and representative nodes
for i, (bar, size, rep) in enumerate(zip(bars, sizes, representatives)):
    # Size label
    ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, 
            f'{size}', va='center', fontsize=11, fontweight='bold')
    # Representative nodes - inside bar
    ax.text(5, bar.get_y() + bar.get_height()/2, 
            rep, va='center', fontsize=9, color='white', style='italic')

# Customize
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=12, fontweight='bold')
ax.set_xlabel('So luong Nodes', fontsize=12)
ax.set_title('TOP 9 CONG DONG LON NHAT TRONG MANG K-POP\n(voi cac node dai dien)', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.set_xlim(0, 220)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='Company-based'),
    Patch(facecolor='#e74c3c', label='Group-centric')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

# Add grid
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('outputs/top9_communities.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Da luu bieu do: outputs/top9_communities.png')

