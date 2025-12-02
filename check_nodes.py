# -*- coding: utf-8 -*-
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://127.0.0.1:7687', auth=('neo4j', '12345678'))

with driver.session(database='network') as session:
    # Tong so nodes
    total = session.run('MATCH (n) RETURN count(n) as c').single()['c']
    print(f'Tong so nodes: {total}')
    
    # Nodes co URL Wikipedia
    wiki = session.run('MATCH (n) WHERE n.url IS NOT NULL AND n.url CONTAINS "wikipedia.org" RETURN count(n) as c').single()['c']
    print(f'Nodes co URL Wikipedia: {wiki}')
    
    # Nodes theo label
    for label in ['Artist', 'Group', 'Song', 'Album', 'Company', 'Genre']:
        c = session.run(f'MATCH (n:{label}) RETURN count(n) as c').single()['c']
        wiki_c = session.run(f'MATCH (n:{label}) WHERE n.url IS NOT NULL AND n.url CONTAINS "wikipedia.org" RETURN count(n) as c').single()['c']
        print(f'  {label}: {c} nodes, {wiki_c} co URL Wikipedia')

driver.close()








