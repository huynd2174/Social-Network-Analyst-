# Checklist: Tất cả các Pattern từ Evaluation.py

## Pattern cần chatbot xử lý:

### 1. True/False - Artist Company (2-hop)
- ✅ `"{member} thuộc công ty {company}, đúng hay sai?"`
- ✅ `"{member} có phải trực thuộc {company} không, đúng hay sai?"`
- ✅ `"{member} do {company} quản lý, đúng hay sai?"`
- ✅ `"{member} được quản lý bởi {company}, đúng hay sai?"`
- ✅ `"{member} có phải trực thuộc {company} qua nhóm {group} không, đúng hay sai?"`
- ✅ `"{member} (nhóm {group}) do {company} quản lý, đúng hay sai?"`
- ✅ `"{member} là thành viên {group} thuộc {company}, đúng hay sai?"`

### 2. Yes/No - Same Company (2-hop)
- ✅ `"{group1} và {group2} có cùng công ty quản lý không?"`
- ✅ `"{group1} và {group2} đều trực thuộc {company} phải không?"`
- ✅ `"Cả {group1} và {group2} có chung công ty {company} chứ?"`
- ✅ `"{group1} có chung công ty với {group2} chứ?"`
- ✅ `"{group1} và {group2} cùng thuộc một công ty phải không?"`

### 3. Multiple Choice - Labelmates (2-hop)
- ✅ `"Nhóm nào cùng công ty với {group1}?"`
- ⚠️ `"Nhóm nào là đồng công ty với {group1} dưới {company}?"` - cần check "đồng công ty"
- ⚠️ `"Nhóm nào khác cũng thuộc {company} giống {group1}?"` - cần check "giống"

### 4. Yes/No - Same Group (2-hop)
- ✅ `"{member1} và {member2} có cùng nhóm nhạc không?"`
- ✅ `"{member1} và {member2} đều thuộc nhóm {group} phải không?"`
- ✅ `"Cả {member1} và {member2} đều là thành viên của {group}, đúng không?"`
- ✅ `"{member1} có chung nhóm với {member2} không?"`
- ✅ `"{member1} và {member2} thuộc cùng một nhóm chứ?"`

### 5. True/False - Song Company (3-hop)
- ✅ `"{song} do {artist} (nhóm {group}) thực hiện, nhóm đó thuộc công ty {company}, đúng hay sai?"` - có pronoun "nhóm đó"
- ✅ `"{song} là bài của {artist} (nhóm {group}); nhóm này trực thuộc {company}, đúng hay sai?"` - có pronoun "nhóm này"
- ✅ `"{song} do {artist} hát trong nhóm {group}; nhóm {group} thuộc {company}, đúng hay sai?"`
- ✅ `"{artist} của nhóm {group} hát {song}; {group} được quản lý bởi {company}, đúng hay sai?"`

### 6. Multiple Choice - Song Company (3-hop)
- ✅ `"{song} do {artist} (nhóm {group}) thực hiện, nhóm đó thuộc công ty nào?"` - có pronoun "nhóm đó"
- ✅ `"{song} là bài của {artist} (nhóm {group}); nhóm này trực thuộc công ty nào?"` - có pronoun "nhóm này"
- ✅ `"{song} do {artist} hát trong nhóm {group}; nhóm {group} thuộc hãng nào?"` - "hãng nào" thay vì "công ty nào"
- ✅ `"{artist} của nhóm {group} hát {song}; {group} do công ty nào quản lý?"`

## Pattern cần thêm vào chatbot:

### Missing patterns:
1. ⚠️ **"đồng công ty"** - chưa có trong pattern matching
2. ⚠️ **"giống {X}"** - pattern comparison
3. ⚠️ **"hãng nào"** - synonym cho "công ty nào"
4. ⚠️ **"qua nhóm X"** - đã có nhưng cần test kỹ hơn
5. ⚠️ **"(nhóm X)"** trong entity - cần extract và resolve đúng

## Test cases để verify:

1. ✅ "Jungkook thuộc công ty Company_Big Hit Entertainment, đúng hay sai?"
2. ✅ "BTS và TXT có cùng công ty quản lý không?"
3. ✅ "Nhóm nào cùng công ty với BTS?"
4. ⚠️ "Nhóm nào là đồng công ty với BTS dưới Company_Big Hit Entertainment?" - test "đồng công ty"
5. ⚠️ "Nhóm nào khác cũng thuộc Company_Big Hit Entertainment giống BTS?" - test "giống"
6. ✅ "Jungkook và V có cùng nhóm nhạc không?"
7. ✅ "Jungkook và V đều thuộc nhóm BTS phải không?"
8. ✅ "Dynamite do Jungkook (nhóm BTS) thực hiện, nhóm đó thuộc công ty Company_Big Hit Entertainment, đúng hay sai?"
9. ✅ "Dynamite là bài của Jungkook (nhóm BTS); nhóm này trực thuộc công ty nào?"
10. ⚠️ "Dynamite do Jungkook hát trong nhóm BTS; nhóm BTS thuộc hãng nào?" - test "hãng nào"

