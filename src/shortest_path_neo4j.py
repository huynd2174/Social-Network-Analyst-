"""
Thuật toán tìm đường đi ngắn nhất giữa 2 node
Dữ liệu được lưu trữ trong Neo4j

Sử dụng 2 phương pháp:
1. Neo4j Cypher query (native)
2. NetworkX sau khi load từ Neo4j
"""
import sys
import io
from typing import List, Optional, Tuple, Dict, Any
from neo4j import GraphDatabase
import networkx as nx

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class ShortestPathFinder:
    """Tìm đường đi ngắn nhất giữa 2 node trong Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = None, undirected: bool = False):
        """
        Khởi tạo kết nối với Neo4j
        
        Args:
            uri: Neo4j URI (ví dụ: bolt://localhost:7687)
            user: Username
            password: Password
            database: Tên database (None = default)
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.undirected = undirected
        print(f"✓ Đã kết nối với Neo4j: {uri}")
    
    def close(self):
        """Đóng kết nối"""
        self.driver.close()
    
    def find_node_by_name(self, name: str) -> Optional[str]:
        """
        Tìm node theo tên (name property)
        
        Args:
            name: Tên node cần tìm
            
        Returns:
            Node ID nếu tìm thấy, None nếu không
        """
        def _find_node(tx, node_name: str):
            query = """
            MATCH (n)
            WHERE n.name = $name OR n.id = $name
            RETURN n.id AS id, n.name AS name, labels(n) AS labels
            LIMIT 1
            """
            result = tx.run(query, name=node_name)
            record = result.single()
            if record:
                return {
                    'id': record['id'],
                    'name': record['name'],
                    'labels': record['labels']
                }
            return None
        
        with self.driver.session(database=self.database) as session:
            node = session.execute_read(_find_node, name)
            return node
    
    def shortest_path_cypher(self, source_name: str, target_name: str, 
                            max_depth: int = 10, undirected: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Tìm đường đi ngắn nhất sử dụng Cypher query
        
        Args:
            source_name: Tên node nguồn
            target_name: Tên node đích
            max_depth: Độ sâu tối đa để tìm
            
        Returns:
            Dictionary chứa path, length, và nodes hoặc None nếu không tìm thấy
        """
        if undirected is None:
            undirected = self.undirected

        def _find_path(tx, src: str, tgt: str, depth: int):
            relationship_pattern = f"(start)-[*..{depth}]-(end)" if undirected else f"(start)-[*..{depth}]->(end)"
            query = f"""
            MATCH path = shortestPath({relationship_pattern})
            WHERE (start.name = $source OR start.id = $source)
               AND (end.name = $target OR end.id = $target)
            RETURN path, length(path) AS pathLength
            LIMIT 1
            """
            
            result = tx.run(query, source=src, target=tgt)
            record = result.single()
            
            if record:
                path = record['path']
                path_length = record['pathLength']
                
                # Extract nodes và relationships từ path
                nodes = []
                relationships = []
                
                for i, node in enumerate(path.nodes):
                    nodes.append({
                        'id': node.get('id'),
                        'name': node.get('name'),
                        'labels': list(node.labels)
                    })
                
                for rel in path.relationships:
                    relationships.append({
                        'type': rel.type,
                        'source': rel.start_node.get('id'),
                        'target': rel.end_node.get('id')
                    })
                
                return {
                    'path_length': path_length,
                    'nodes': nodes,
                    'relationships': relationships
                }
            return None
        
        with self.driver.session(database=self.database) as session:
            return session.execute_read(_find_path, source_name, target_name, max_depth)
    
    def all_shortest_paths_cypher(self, source_name: str, target_name: str,
                                   max_depth: int = 10, undirected: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Tìm tất cả các đường đi ngắn nhất giữa 2 node
        
        Args:
            source_name: Tên node nguồn
            target_name: Tên node đích
            max_depth: Độ sâu tối đa
            
        Returns:
            List các đường đi
        """
        if undirected is None:
            undirected = self.undirected

        def _find_all_paths(tx, src: str, tgt: str, depth: int):
            relationship_pattern = f"(start)-[*..{depth}]-(end)" if undirected else f"(start)-[*..{depth}]->(end)"
            query = f"""
            MATCH path = allShortestPaths({relationship_pattern})
            WHERE (start.name = $source OR start.id = $source)
               AND (end.name = $target OR end.id = $target)
            RETURN path, length(path) AS pathLength
            """
            
            result = tx.run(query, source=src, target=tgt)
            paths = []
            
            for record in result:
                path = record['path']
                path_length = record['pathLength']
                
                nodes = []
                relationships = []
                
                for node in path.nodes:
                    nodes.append({
                        'id': node.get('id'),
                        'name': node.get('name'),
                        'labels': list(node.labels)
                    })
                
                for rel in path.relationships:
                    relationships.append({
                        'type': rel.type,
                        'source': rel.start_node.get('id'),
                        'target': rel.end_node.get('id')
                    })
                
                paths.append({
                    'path_length': path_length,
                    'nodes': nodes,
                    'relationships': relationships
                })
            
            return paths
        
        with self.driver.session(database=self.database) as session:
            return session.execute_read(_find_all_paths, source_name, target_name, max_depth)
    
    def load_graph_to_networkx(self) -> nx.DiGraph:
        """
        Load toàn bộ đồ thị từ Neo4j sang NetworkX
        
        Returns:
            NetworkX DiGraph object
        """
        def _load_graph(tx):
            # Load nodes - consume ngay trong transaction
            node_query = """
            MATCH (n)
            RETURN n.id AS id, n.name AS name, labels(n) AS labels, 
                   properties(n) AS props
            """
            nodes_result = tx.run(node_query)
            nodes_data = [record for record in nodes_result]
            
            # Load edges - consume ngay trong transaction
            edge_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id AS source, b.id AS target, type(r) AS rel_type,
                   properties(r) AS props
            """
            edges_result = tx.run(edge_query)
            edges_data = [record for record in edges_result]
            
            return nodes_data, edges_data
        
        print("Đang load đồ thị từ Neo4j sang NetworkX...")
        G = nx.DiGraph()
        
        with self.driver.session(database=self.database) as session:
            nodes_data, edges_data = session.execute_read(_load_graph)
            
            # Add nodes
            node_count = 0
            for record in nodes_data:
                node_id = record['id']
                if node_id:  # Chỉ thêm nếu có id
                    props = dict(record.get('props', {}))
                    # Thêm name và labels vào props, không override nếu đã có
                    if 'name' not in props:
                        props['name'] = record.get('name')
                    if 'labels' not in props:
                        props['labels'] = list(record.get('labels', []))
                    G.add_node(node_id, **props)
                    node_count += 1
            
            # Add edges
            edge_count = 0
            for record in edges_data:
                source = record.get('source')
                target = record.get('target')
                if source and target:
                    G.add_edge(source, target,
                              type=record.get('rel_type', 'RELATED'),
                              **dict(record.get('props', {})))
                    edge_count += 1
            
            print(f"✓ Đã load {node_count} nodes và {edge_count} edges")
        
        return G
    
    def shortest_path_networkx(self, source_name: str, target_name: str,
                               G: nx.DiGraph = None,
                               undirected: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        """
        Tìm đường đi ngắn nhất sử dụng NetworkX
        
        Args:
            source_name: Tên node nguồn
            target_name: Tên node đích
            G: NetworkX graph (nếu None sẽ load từ Neo4j)
            
        Returns:
            Dictionary chứa path và length hoặc None
        """
        if undirected is None:
            undirected = self.undirected

        if G is None:
            G = self.load_graph_to_networkx()
        
        work_graph = G.to_undirected() if undirected else G
        
        # Tìm node ID từ name
        source_id = None
        target_id = None
        
        for node_id, data in work_graph.nodes(data=True):
            if data.get('name') == source_name or node_id == source_name:
                source_id = node_id
            if data.get('name') == target_name or node_id == target_name:
                target_id = node_id
        
        if source_id is None:
            print(f"✗ Không tìm thấy node nguồn: {source_name}")
            return None
        
        if target_id is None:
            print(f"✗ Không tìm thấy node đích: {target_name}")
            return None
        
        try:
            # Tìm đường đi ngắn nhất
            path = nx.shortest_path(work_graph, source_id, target_id)
            path_length = len(path) - 1
            
            # Lấy thông tin chi tiết về path
            nodes_info = []
            relationships_info = []
            
            for i, node_id in enumerate(path):
                node_data = work_graph.nodes[node_id]
                nodes_info.append({
                    'id': node_id,
                    'name': node_data.get('name', node_id),
                    'labels': node_data.get('labels', [])
                })
                
                # Lấy relationship giữa node hiện tại và node tiếp theo
                if i < len(path) - 1:
                    next_node_id = path[i + 1]
                    edge_data = work_graph.get_edge_data(node_id, next_node_id, {})
                    relationships_info.append({
                        'type': edge_data.get('type', 'RELATED'),
                        'source': node_id,
                        'target': next_node_id
                    })
            
            return {
                'path_length': path_length,
                'nodes': nodes_info,
                'relationships': relationships_info
            }
        
        except nx.NetworkXNoPath:
            print(f"✗ Không có đường đi giữa {source_name} và {target_name}")
            return None
    
    def print_path(self, path_result: Dict[str, Any], method: str = "Cypher"):
        """
        In đường đi ra màn hình
        
        Args:
            path_result: Kết quả từ shortest_path_cypher hoặc shortest_path_networkx
            method: Phương pháp sử dụng ("Cypher" hoặc "NetworkX")
        """
        if path_result is None:
            print("Không tìm thấy đường đi")
            return
        
        print(f"\n{'=' * 70}")
        print(f"ĐƯỜNG ĐI NGẮN NHẤT ({method})")
        print(f"{'=' * 70}")
        print(f"Độ dài: {path_result['path_length']} bước\n")
        
        nodes = path_result['nodes']
        relationships = path_result.get('relationships', [])
        
        print("Đường đi:")
        for i, node in enumerate(nodes):
            node_name = node.get('name', node.get('id', 'Unknown'))
            labels = node.get('labels', [])
            label_str = ', '.join(labels) if labels else 'Entity'
            
            print(f"  {i+1}. {node_name:30s} [{label_str}]")
            
            # In relationship nếu có
            if i < len(relationships):
                rel = relationships[i]
                print(f"      --[{rel['type']}]-->")
        
        print(f"\nTổng cộng: {len(nodes)} nodes, {len(relationships)} relationships")


def main():
    """Hàm chính để chạy thử"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tìm đường đi ngắn nhất giữa 2 node trong Neo4j'
    )
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j URI (default: bolt://localhost:7687)')
    parser.add_argument('--user', type=str, default='neo4j',
                       help='Neo4j username (default: neo4j)')
    parser.add_argument('--password', type=str, required=True,
                       help='Neo4j password')
    parser.add_argument('--database', type=str, default=None,
                       help='Neo4j database name (default: neo4j)')
    parser.add_argument('--source', type=str, required=True,
                       help='Tên node nguồn')
    parser.add_argument('--target', type=str, required=True,
                       help='Tên node đích')
    parser.add_argument('--method', type=str, choices=['cypher', 'networkx', 'both'],
                       default='both', help='Phương pháp sử dụng (default: both)')
    parser.add_argument('--max-depth', type=int, default=10,
                       help='Độ sâu tối đa (default: 10)')
    parser.add_argument('--undirected', action='store_true',
                       help='Xem các cạnh là vô hướng khi tìm đường đi')
    
    args = parser.parse_args()
    
    # Khởi tạo finder
    finder = ShortestPathFinder(args.uri, args.user, args.password, args.database, undirected=args.undirected)
    
    try:
        print("\n" + "=" * 70)
        print("THUẬT TOÁN TÌM ĐƯỜNG ĐI NGẮN NHẤT")
        print("=" * 70)
        print(f"Node nguồn: {args.source}")
        print(f"Node đích: {args.target}")
        print(f"Phương pháp: {args.method}")
        print(f"Chế độ vô hướng: {'Có' if args.undirected else 'Không'}")
        print("=" * 70)
        
        # Kiểm tra node có tồn tại không
        source_node = finder.find_node_by_name(args.source)
        target_node = finder.find_node_by_name(args.target)
        
        if source_node is None:
            print(f"\n✗ Không tìm thấy node nguồn: {args.source}")
            return
        
        if target_node is None:
            print(f"\n✗ Không tìm thấy node đích: {args.target}")
            return
        
        print(f"\n✓ Node nguồn: {source_node['name']} ({', '.join(source_node['labels'])})")
        print(f"✓ Node đích: {target_node['name']} ({', '.join(target_node['labels'])})")
        
        # Tìm đường đi
        if args.method in ['cypher', 'both']:
            print("\n[1] Sử dụng Neo4j Cypher Query:")
            path_cypher = finder.shortest_path_cypher(args.source, args.target, args.max_depth, undirected=args.undirected)
            finder.print_path(path_cypher, "Cypher")
        
        if args.method in ['networkx', 'both']:
            print("\n[2] Sử dụng NetworkX:")
            path_nx = finder.shortest_path_networkx(args.source, args.target, undirected=args.undirected)
            finder.print_path(path_nx, "NetworkX")
        
        # So sánh kết quả nếu dùng cả 2
        if args.method == 'both' and path_cypher and path_nx:
            print("\n" + "=" * 70)
            print("SO SÁNH KẾT QUẢ")
            print("=" * 70)
            print(f"Cypher path length: {path_cypher['path_length']}")
            print(f"NetworkX path length: {path_nx['path_length']}")
            if path_cypher['path_length'] == path_nx['path_length']:
                print("✓ Kết quả khớp nhau!")
            else:
                print("⚠ Kết quả khác nhau (có thể do cách xử lý)")
        
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        finder.close()
        print("\n✓ Đã đóng kết nối Neo4j")


if __name__ == '__main__':
    main()

