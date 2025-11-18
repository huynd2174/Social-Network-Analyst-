"""
Thuật toán tìm đường đi ngắn nhất sử dụng Graph Data Science (GDS) Python Client
Dữ liệu được lưu trữ trong Neo4j

Yêu cầu:
- Neo4j Server với Graph Data Science Library plugin
- pip install graphdatascience
"""
import sys
import io
from typing import Optional, Dict, Any, List, Tuple
from graphdatascience import GraphDataScience

# Robust UTF-8 console output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class ShortestPathGDS:
    """Tìm đường đi ngắn nhất sử dụng GDS Python Client"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = None,
                 undirected: bool = False):
        """
        Khởi tạo kết nối với Neo4j qua GDS
        
        Args:
            uri: Neo4j URI (ví dụ: bolt://localhost:7687)
            user: Username
            password: Password
            database: Tên database (None = default)
        """
        self.gds = GraphDataScience(uri, auth=(user, password), database=database)
        self.database = database
        self.undirected = undirected
        print(f"✓ Đã kết nối với Neo4j qua GDS: {uri}")
    
    def close(self):
        """Đóng kết nối"""
        self.gds.close()
    
    def find_node_id_by_name(self, name: str) -> Optional[int]:
        """
        Tìm node ID theo tên
        
        Args:
            name: Tên node cần tìm
            
        Returns:
            Node ID (internal Neo4j ID) nếu tìm thấy, None nếu không
        """
        query = """
        MATCH (n)
        WHERE n.name = $name OR n.id = $name
        RETURN id(n) AS nodeId, n.name AS name, labels(n) AS labels
        LIMIT 1
        """
        
        result = self.gds.run_cypher(query, params={"name": name})
        
        if not result.empty:
            return int(result.iloc[0]['nodeId'])
        return None
    
    def create_graph_projection(self, graph_name: str = "myGraph",
                                node_labels: str = "*",
                                relationship_types: str = "*",
                                undirected: Optional[bool] = None) -> Any:
        """
        Tạo graph projection từ dữ liệu trong Neo4j
        
        Args:
            graph_name: Tên graph projection
            node_labels: Node labels (mặc định: "*" = tất cả)
            relationship_types: Relationship types (mặc định: "*" = tất cả)
            
        Returns:
            Graph projection object
        """
        print(f"\nĐang tạo graph projection '{graph_name}'...")
        
        # Kiểm tra xem graph đã tồn tại chưa
        try:
            existing_graph = self.gds.graph.get(graph_name)
            print(f"✓ Graph '{graph_name}' đã tồn tại, sử dụng graph hiện có")
            return existing_graph
        except:
            pass
        
        if undirected is None:
            undirected = self.undirected

        # Chuẩn bị relationship projection
        if undirected:
            rel_projection = {
                "ALL_RELATIONSHIPS": {
                    "type": relationship_types if relationship_types != "*" else "*",
                    "orientation": "UNDIRECTED"
                }
            }
        else:
            rel_projection = relationship_types

        # Tạo graph projection mới
        G, result = self.gds.graph.project(
            graph_name,
            node_labels,
            rel_projection
        )
        
        print(f"✓ Đã tạo graph projection:")
        print(f"  - Nodes: {G.node_count():,}")
        print(f"  - Relationships: {G.relationship_count():,}")
        # Thông tin thời gian tùy phiên bản GDS (createMillis hoặc projectMillis)
        if hasattr(result, "to_dict"):
            if hasattr(result, "columns"):
                project_info = result.to_dict('records')[0]
            else:
                project_info = result.to_dict()
        else:
            project_info = result
        if isinstance(project_info, dict):
            create_millis = project_info.get('createMillis') or project_info.get('projectMillis')
            if create_millis is not None:
                print(f"  - Thời gian tạo: {create_millis}ms")
        
        return G
    
    def shortest_path_dijkstra(self, source_name: str, target_name: str,
                               graph_name: str = "myGraph",
                               relationship_weight_property: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Tìm đường đi ngắn nhất sử dụng Dijkstra algorithm
        
        Args:
            source_name: Tên node nguồn
            target_name: Tên node đích
            graph_name: Tên graph projection
            relationship_weight_property: Property name cho trọng số (None = unweighted)
            
        Returns:
            Dictionary chứa path và thông tin chi tiết
        """
        # Tìm node IDs
        source_id = self.find_node_id_by_name(source_name)
        target_id = self.find_node_id_by_name(target_name)
        
        if source_id is None:
            print(f"✗ Không tìm thấy node nguồn: {source_name}")
            return None
        
        if target_id is None:
            print(f"✗ Không tìm thấy node đích: {target_name}")
            return None
        
        # Tạo hoặc lấy graph projection
        G = self.create_graph_projection(graph_name, undirected=self.undirected)
        
        # Map node names sang internal IDs trong graph projection
        # GDS sử dụng internal Neo4j IDs
        source_node_id = source_id
        target_node_id = target_id
        
        print(f"\nĐang tìm đường đi ngắn nhất (Dijkstra)...")
        print(f"  Từ: {source_name} (ID: {source_node_id})")
        print(f"  Đến: {target_name} (ID: {target_node_id})")
        
        try:
            # Chạy Dijkstra shortest path
            if relationship_weight_property:
                result = self.gds.shortestPath.dijkstra.stream(
                    G,
                    sourceNode=source_node_id,
                    targetNode=target_node_id,
                    relationshipWeightProperty=relationship_weight_property
                )
            else:
                result = self.gds.shortestPath.dijkstra.stream(
                    G,
                    sourceNode=source_node_id,
                    targetNode=target_node_id
                )
            
            if result.empty:
                print("✗ Không tìm thấy đường đi")
                return None
            
            # Lấy kết quả
            path_row = result.iloc[0]
            raw_path = path_row.get('path')
            if isinstance(raw_path, list):
                path_node_ids = raw_path
            elif hasattr(raw_path, 'nodes'):
                path_node_ids = []
                for node in raw_path.nodes:
                    internal_id = getattr(node, "element_id", None)
                    if internal_id is None:
                        internal_id = getattr(node, "id", None)
                    path_node_ids.append(internal_id)
            else:
                raise ValueError("Không thể đọc dữ liệu đường đi trả về từ GDS")
            total_cost = path_row.get('totalCost', len(path_node_ids) - 1)
            
            # Lấy thông tin chi tiết về các nodes trong path
            nodes_info = []
            for node_id in path_node_ids:
                node_info = self._get_node_info(node_id)
                if node_info:
                    nodes_info.append(node_info)
            
            # Lấy thông tin về relationships
            relationships_info = []
            for i in range(len(path_node_ids) - 1):
                rel_info = self._get_relationship_info(path_node_ids[i], path_node_ids[i + 1])
                if rel_info:
                    relationships_info.append(rel_info)
            
            return {
                'path_length': len(path_node_ids) - 1,
                'total_cost': total_cost,
                'nodes': nodes_info,
                'relationships': relationships_info,
                'path_node_ids': path_node_ids
            }
        
        except Exception as e:
            print(f"✗ Lỗi khi tìm đường đi: {e}")
            return None
    
    def shortest_path_yens(self, source_name: str, target_name: str,
                           k: int = 3,
                           graph_name: str = "myGraph") -> List[Dict[str, Any]]:
        """
        Tìm K đường đi ngắn nhất sử dụng Yen's algorithm
        
        Args:
            source_name: Tên node nguồn
            target_name: Tên node đích
            k: Số đường đi cần tìm
            graph_name: Tên graph projection
            
        Returns:
            List các đường đi
        """
        source_id = self.find_node_id_by_name(source_name)
        target_id = self.find_node_id_by_name(target_name)
        
        if source_id is None or target_id is None:
            return []
        
        G = self.create_graph_projection(graph_name, undirected=self.undirected)
        
        print(f"\nĐang tìm {k} đường đi ngắn nhất (Yen's algorithm)...")
        
        try:
            result = self.gds.shortestPath.yens.stream(
                G,
                sourceNode=source_id,
                targetNode=target_id,
                k=k
            )
            
            paths = []
            for idx, row in result.iterrows():
                path_node_ids = row['path']
                total_cost = row.get('totalCost', len(path_node_ids) - 1)
                
                nodes_info = []
                for node_id in path_node_ids:
                    node_info = self._get_node_info(node_id)
                    if node_info:
                        nodes_info.append(node_info)
                
                relationships_info = []
                for i in range(len(path_node_ids) - 1):
                    rel_info = self._get_relationship_info(path_node_ids[i], path_node_ids[i + 1])
                    if rel_info:
                        relationships_info.append(rel_info)
                
                paths.append({
                    'path_length': len(path_node_ids) - 1,
                    'total_cost': total_cost,
                    'nodes': nodes_info,
                    'relationships': relationships_info,
                    'path_index': idx
                })
            
            return paths
        
        except Exception as e:
            print(f"✗ Lỗi: {e}")
            return []
    
    def _get_node_info(self, node_id: Any) -> Optional[Dict[str, Any]]:
        """Lấy thông tin node từ Neo4j"""
        if isinstance(node_id, int):
            query = """
            MATCH (n)
            WHERE id(n) = $identifier
            RETURN n.id AS id, n.name AS name, labels(n) AS labels, properties(n) AS props
            LIMIT 1
            """
            params = {"identifier": node_id}
        else:
            query = """
            MATCH (n)
            WHERE elementId(n) = $identifier
            RETURN n.id AS id, n.name AS name, labels(n) AS labels, properties(n) AS props
            LIMIT 1
            """
            params = {"identifier": node_id}
        
        result = self.gds.run_cypher(query, params=params)
        
        if not result.empty:
            row = result.iloc[0]
            return {
                'id': row.get('id'),
                'name': row.get('name'),
                'labels': list(row.get('labels', [])),
                'nodeId': node_id
            }
        return None
    
    def _get_relationship_info(self, source_id: Any, target_id: Any) -> Optional[Dict[str, Any]]:
        """Lấy thông tin relationship giữa 2 nodes"""
        def _clause(alias: str, identifier: Any, param_name: str) -> Tuple[str, Dict[str, Any]]:
            if isinstance(identifier, int):
                return f"id({alias}) = ${param_name}", {param_name: identifier}
            else:
                return f"elementId({alias}) = ${param_name}", {param_name: identifier}
        
        def _query(src, tgt, reverse=False):
            clause_a, params_a = _clause("a", src, "sourceId" if not reverse else "revSourceId")
            clause_b, params_b = _clause("b", tgt, "targetId" if not reverse else "revTargetId")
            query = f"""
            MATCH (a)-[r]->(b)
            WHERE {clause_a} AND {clause_b}
            RETURN type(r) AS type, properties(r) AS props
            LIMIT 1
            """
            return self.gds.run_cypher(query, params={**params_a, **params_b})
        
        result = _query(source_id, target_id)
        if not result.empty:
            row = result.iloc[0]
            return {
                'type': row.get('type'),
                'source': source_id,
                'target': target_id,
                'properties': dict(row.get('props', {}))
            }
        
        if self.undirected:
            result = _query(target_id, source_id, reverse=True)
            if not result.empty:
                row = result.iloc[0]
                return {
                    'type': row.get('type'),
                    'source': source_id,
                    'target': target_id,
                    'properties': dict(row.get('props', {}))
                }
        
        return None
    
    def print_path(self, path_result: Dict[str, Any], method: str = "Dijkstra"):
        """In đường đi ra màn hình"""
        if path_result is None:
            print("Không tìm thấy đường đi")
            return
        
        print(f"\n{'=' * 70}")
        print(f"ĐƯỜNG ĐI NGẮN NHẤT ({method})")
        print(f"{'=' * 70}")
        print(f"Độ dài: {path_result['path_length']} bước")
        if 'total_cost' in path_result:
            print(f"Tổng chi phí: {path_result['total_cost']}\n")
        
        nodes = path_result['nodes']
        relationships = path_result.get('relationships', [])
        
        print("Đường đi:")
        for i, node in enumerate(nodes):
            node_name = node.get('name', node.get('id', 'Unknown'))
            labels = node.get('labels', [])
            label_str = ', '.join(labels) if labels else 'Entity'
            
            print(f"  {i+1}. {node_name:30s} [{label_str}]")
            
            if i < len(relationships):
                rel = relationships[i]
                print(f"      --[{rel['type']}]-->")
        
        print(f"\nTổng cộng: {len(nodes)} nodes, {len(relationships)} relationships")


def main():
    """Hàm chính"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Tìm đường đi ngắn nhất sử dụng GDS Python Client'
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
    parser.add_argument('--graph-name', type=str, default='myGraph',
                       help='Tên graph projection (default: myGraph)')
    parser.add_argument('--k-paths', type=int, default=1,
                       help='Số đường đi cần tìm (cho Yen algorithm, default: 1)')
    parser.add_argument('--algorithm', type=str, choices=['dijkstra', 'yens'],
                       default='dijkstra', help='Thuật toán sử dụng (default: dijkstra)')
    parser.add_argument('--undirected', action='store_true',
                       help='Chiếu graph với quan hệ vô hướng')
    
    args = parser.parse_args()
    
    finder = ShortestPathGDS(args.uri, args.user, args.password, args.database, undirected=args.undirected)
    
    try:
        print("\n" + "=" * 70)
        print("THUẬT TOÁN TÌM ĐƯỜNG ĐI NGẮN NHẤT (GDS Python Client)")
        print("=" * 70)
        print(f"Node nguồn: {args.source}")
        print(f"Node đích: {args.target}")
        print(f"Thuật toán: {args.algorithm}")
        print(f"Chế độ vô hướng: {'Có' if args.undirected else 'Không'}")
        print("=" * 70)
        
        if args.algorithm == 'dijkstra':
            path = finder.shortest_path_dijkstra(
                args.source, 
                args.target,
                graph_name=args.graph_name
            )
            finder.print_path(path, "Dijkstra")
        
        elif args.algorithm == 'yens':
            paths = finder.shortest_path_yens(
                args.source,
                args.target,
                k=args.k_paths,
                graph_name=args.graph_name
            )
            
            if paths:
                print(f"\nTìm thấy {len(paths)} đường đi:")
                for i, path in enumerate(paths, 1):
                    print(f"\n--- Đường đi {i} ---")
                    finder.print_path(path, f"Yen's (Path {i})")
            else:
                print("Không tìm thấy đường đi nào")
    
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        finder.close()
        print("\n✓ Đã đóng kết nối")


if __name__ == '__main__':
    main()

