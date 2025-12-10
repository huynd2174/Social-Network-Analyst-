"""
Quick test: entity extraction and company/group matching for reported errors.
Focuses on the 20 failed questions to inspect extracted entities and overlaps.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatbot.knowledge_graph import KpopKnowledgeGraph
from chatbot.chatbot import KpopChatbot


def print_entity_info(kg: KpopKnowledgeGraph, entity_id: str):
    etype = kg.get_entity_type(entity_id)
    print(f"    - {entity_id} ({etype})")
    if etype == "Artist":
        groups = kg.get_artist_groups(entity_id)
        comps = kg.get_artist_companies(entity_id)
        if groups:
            print(f"      groups: {list(groups)}")
        if comps:
            print(f"      companies: {list(comps)}")
    elif etype == "Group":
        comps = kg.get_group_companies(entity_id)
        if comps:
            print(f"      companies: {list(comps)}")
    elif etype == "Company":
        print(f"      (company entity)")


errors = [
    ("Q00252", "Eunseo (ca sĩ) có phải trực thuộc Company_Emi Music Japan qua nhóm WJSN không, đúng hay sai?", "artist_company"),
    ("Q00145", "Jeon So-yeon có phải trực thuộc Company_Cube không, đúng hay sai?", "artist_company"),
    ("Q00549", "Kwon Eun-bi thuộc công ty Company_Stone Music, đúng hay sai?", "artist_company"),
    ("Q01399", "Nhóm nào khác cũng thuộc Company_Woollim giống Lovelyz?", "labelmates"),
    ("Q00715", "Seventeen (nhóm nhạc) và Orange Caramel có cùng công ty quản lý không?", "same_company"),
    ("Q00597", "8Eight và Illit (nhóm nhạc) đều trực thuộc Company_HYBE phải không?", "same_company"),
    ("Q00995", "After School (nhóm nhạc) có chung công ty với April (nhóm nhạc) chứ?", "same_company"),
    ("Q01394", "Nhóm nào cùng công ty với WINNER?", "labelmates"),
    ("Q00901", "Cả MOBB và Jinusean có chung công ty Company_YG Entertainment chứ?", "same_company"),
    ("Q00273", "Kim Do-yeon thuộc công ty Company_Ymc, đúng hay sai?", "artist_company"),
    ("Q00500", "Lee Hyori có phải trực thuộc Company_DSP Media không, đúng hay sai?", "artist_company"),
    ("Q01261", "Nhóm nào cùng công ty với Apink BnN?", "labelmates"),
    ("Q01868", "Lion Heart (bài hát) là bài của Sunny (ca sĩ) (nhóm Girls' Generation-Oh!GG); nhóm này trực thuộc Company_Iriver, đúng hay sai?", "artist_company"),
    ("Q00458", "Hyoyeon có phải trực thuộc Company_Cube qua nhóm Girls' Generation-Oh!GG không, đúng hay sai?", "artist_company"),
    ("Q00656", "Cả Apink BnN và VICTON có chung công ty Company_Ist Entertainment chứ?", "same_company"),
    ("Q01346", "Nhóm nào là đồng công ty với The SeeYa dưới Company_Core Contents Media?", "labelmates"),
    ("Q00748", "Everglow và X1 (nhóm nhạc) đều trực thuộc Company_Stone Music phải không?", "same_company"),
    ("Q00958", "N.Flying và AOA Black có cùng công ty quản lý không?", "same_company"),
    ("Q01680", "Miyawaki Sakura của nhóm LE SSERAFIM hát Buenos Aires (đĩa đơn); LE SSERAFIM được quản lý bởi Company_Source Music, đúng hay sai?", "artist_company"),
    ("Q01560", "Cả MJ (ca sĩ Hàn Quốc) và Yoon San-ha đều là thành viên của Astro (nhóm nhạc), đúng không?", "same_group"),
]


def main():
    print("Loading chatbot (KG only focus)...")
    bot = KpopChatbot(verbose=False)
    kg = bot.kg

    for qid, question, category in errors:
        print("\n" + "-" * 80)
        print(f"{qid} | {category}")
        print(f"Q: {question}")

        # Use RAG to extract entities
        context = bot.rag.retrieve_context(question, max_entities=6, max_hops=3)
        ents = context.get("entities", [])
        print(f"Extracted entities: {len(ents)}")
        for e in ents:
            print_entity_info(kg, e["id"])

        # For same_company / labelmates, show company intersections
        if category in ("same_company", "labelmates"):
            if len(ents) >= 2:
                companies_lists = []
                for e in ents:
                    etype = kg.get_entity_type(e["id"])
                    comps = set()
                    if etype == "Artist":
                        comps.update(kg.get_artist_companies(e["id"]))
                        for g in kg.get_artist_groups(e["id"]):
                            comps.update(kg.get_group_companies(g))
                    elif etype == "Group":
                        comps.update(kg.get_group_companies(e["id"]))
                    companies_lists.append((e["id"], comps))
                print("Company sets:")
                for eid, comps in companies_lists:
                    print(f"  {eid}: {list(comps)}")
                # intersections
                if len(companies_lists) >= 2:
                    inter = companies_lists[0][1].copy()
                    for _, cset in companies_lists[1:]:
                        inter = inter.intersection(cset)
                    print(f"Intersection: {list(inter)}")
        # For same_group, show group intersections
        if category == "same_group":
            if len(ents) >= 2:
                groups_lists = []
                for e in ents:
                    etype = kg.get_entity_type(e["id"])
                    gset = set()
                    if etype == "Artist":
                        gset.update(kg.get_artist_groups(e["id"]))
                    elif etype == "Group":
                        gset.add(e["id"])
                    groups_lists.append((e["id"], gset))
                print("Group sets:")
                for eid, gset in groups_lists:
                    print(f"  {eid}: {list(gset)}")
                if len(groups_lists) >= 2:
                    inter = groups_lists[0][1].copy()
                    for _, gset in groups_lists[1:]:
                        inter = inter.intersection(gset)
                    print(f"Intersection: {list(inter)}")


if __name__ == "__main__":
    main()

