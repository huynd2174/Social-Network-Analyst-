# Kết quả shortest path cho 10 cặp node

_Chế độ: undirected, phương pháp: Cypher._

## 1. `BTS` → `BLACKPINK`
- Độ dài: **2** bước
- Đường đi:
  1. BTS [Group]
     └─[MANAGED_BY]→
  2. Universal Music [Company]
     └─[MANAGED_BY]→
  3. BLACKPINK [Group]

## 2. `BLACKPINK` → `AKMU`
- Độ dài: **4** bước
- Đường đi:
  1. BLACKPINK [Group]
     └─[RELEASED]→
  2. Born Pink [Album]
     └─[PRODUCED_ALBUM]→
  3. Bekuh Boom [Artist]
     └─[IS_GENRE]→
  4. Pop [Genre]
     └─[IS_GENRE]→
  5. AKMU [Group]

## 3. `Kill This Love (bài hát)` → `Yet to Come (The Most Beautiful Moment)`
- Độ dài: **4** bước
- Đường đi:
  1. Kill This Love (bài hát) [Song]
     └─[SINGS]→
  2. BLACKPINK [Group]
     └─[MANAGED_BY]→
  3. Universal Music [Company]
     └─[MANAGED_BY]→
  4. BTS [Group]
     └─[SINGS]→
  5. Yet to Come (The Most Beautiful Moment) [Song]

## 4. `2 Cool 4 Skool` → `Yet to Come (The Most Beautiful Moment)`
- Độ dài: **2** bước
- Đường đi:
  1. 2 Cool 4 Skool [Song]
     └─[SINGS]→
  2. BTS [Group]
     └─[SINGS]→
  3. Yet to Come (The Most Beautiful Moment) [Song]

## 5. `BTS` → `Universal Music`
- Độ dài: **1** bước
- Đường đi:
  1. BTS [Group]
     └─[MANAGED_BY]→
  2. Universal Music [Company]

## 6. `BLACKPINK` → `Pop`
- Độ dài: **3** bước
- Đường đi:
  1. BLACKPINK [Group]
     └─[RELEASED]→
  2. Born Pink [Album]
     └─[PRODUCED_ALBUM]→
  3. Bekuh Boom [Artist]
     └─[IS_GENRE]→
  4. Pop [Genre]

## 7. `Pdogg` → `Bekuh Boom`
- Độ dài: **2** bước
- Đường đi:
  1. Pdogg [Artist]
     └─[HAS_OCCUPATION]→
  2. Nhạc sĩ [Occupation]
     └─[HAS_OCCUPATION]→
  3. Bekuh Boom [Artist]

## 8. `RM (rapper)` → `Bekuh Boom`
- Độ dài: **2** bước
- Đường đi:
  1. RM (rapper) [Artist]
     └─[HAS_OCCUPATION]→
  2. Nhạc sĩ [Occupation]
     └─[HAS_OCCUPATION]→
  3. Bekuh Boom [Artist]

## 9. `BTS` → `Hip hop`
- Độ dài: **1** bước
- Đường đi:
  1. BTS [Group]
     └─[IS_GENRE]→
  2. Hip hop [Genre]

## 10. `BLACKPINK` → `YG Entertainment`
- Độ dài: **3** bước
- Đường đi:
  1. BLACKPINK [Group]
     └─[IS_GENRE]→
  2. Hip hop [Genre]
     └─[IS_GENRE]→
  3. GD X Taeyang [Group]
     └─[MANAGED_BY]→
  4. YG Entertainment [Company]
