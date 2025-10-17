# Social Network Analyst - Mạng Lưới Nghệ Sĩ Quốc tế

Dự án này tập trung vào việc xây dựng và phân tích mạng lưới xã hội của các nghệ sĩ trong ngành công nghiệp âm nhạc Hàn Quốc (K-pop). Dữ liệu được thu thập tự động từ Wikipedia tiếng Việt và được xử lý để tạo thành một đồ thị (graph) có cấu trúc, biểu diễn các mối quan hệ phức tạp giữa các nghệ sĩ, nhóm nhạc, album, và các thực thể liên quan.

## Tổng quan dự án

Mục tiêu chính là tạo ra một tập dữ liệu đồ thị chất lượng cao, nơi mỗi nút (node) và cạnh (edge) đều mang một ý nghĩa cụ thể. Dữ liệu này sau đó được sử dụng để phân tích và trực quan hóa, nhằm tìm ra những hiểu biết sâu sắc (insights) về cấu trúc của mạng lưới.

Quy trình thực hiện bao gồm:
1.  **Thu thập dữ liệu (Crawling/Scraping)**: Một script Python trong tệp Jupyter Notebook sẽ truy cập Wikipedia, bắt đầu từ một danh sách các nghệ sĩ/nhóm nhạc hạt giống.
2.  **Xây dựng đồ thị**: Script sử dụng thuật toán **BFS (Tìm kiếm theo chiều rộng)** để mở rộng mạng lưới, thu thập các trang liên quan và xác định mối quan hệ giữa chúng.
3.  **Lưu trữ dữ liệu**: Toàn bộ đồ thị gồm các nút và cạnh được lưu vào tệp `korean_artists_graph_bfs.json`.
4.  **Phân tích và Trực quan hóa**: Tệp Notebook cũng chứa các bước để đọc dữ liệu từ tệp JSON, thực hiện phân tích (ví dụ: tìm các nút trung tâm, phát hiện cộng đồng) và trực quan hóa đồ thị.

## Cấu trúc Repository

-   `Social Network Analyst.ipynb`: Tệp Jupyter Notebook chính, chứa toàn bộ mã nguồn Python để:
    -   Thu thập dữ liệu từ Wikipedia.
    -   Xử lý và làm sạch dữ liệu.
    -   Xây dựng đồ thị.
    -   Phân tích và trực quan hóa mạng lưới.
-   `korean_artists_graph_bfs.json`: Tệp dữ liệu đầu ra chứa thông tin về các nút và cạnh của đồ thị sau khi thu thập.
-   `README.md`: Tệp tài liệu hướng dẫn này.

## Mô hình dữ liệu

### Nodes (Nút)
Đại diện cho một thực thể (ví dụ: nghệ sĩ, nhóm nhạc, album).
-   **`title`**: Tên trang Wikipedia, dùng làm ID.
-   **`label`**: Phân loại thực thể.
-   **`url`**: Liên kết đến trang Wikipedia.
-   **`infobox`**: Dữ liệu có cấu trúc từ bảng thông tin của trang.

### Edges (Cạnh)
Đại diện cho mối quan hệ giữa hai nút.
-   **`source`**: Nút bắt đầu.
-   **`target`**: Nút kết thúc.
-   **`type`**: Loại quan hệ (`MEMBER_OF`, `MANAGED_BY`, `COLLABORATED_WITH`, v.v.).

## Hướng dẫn sử dụng

1.  **Cài đặt môi trường:**
    -   Đảm bảo bạn đã cài đặt Python 3 và Jupyter Notebook/JupyterLab.
    -   Cài đặt các thư viện cần thiết bằng cách chạy các cell cài đặt ở đầu tệp `Social Network Analyst.ipynb`. Các thư viện chính bao gồm:
        ```
        pip install requests beautifulsoup4 networkx matplotlib
        ```

2.  **Chạy Notebook:**
    -   Mở tệp `Social Network Analyst.ipynb` bằng Jupyter.
    -   Bạn có thể chạy tuần tự từng cell để thực hiện lại toàn bộ quy trình từ thu thập đến phân tích.
    -   **Lưu ý**: Quá trình thu thập dữ liệu có thể mất một khoảng thời gian tùy thuộc vào số lượng node bạn muốn lấy.

3.  **Sử dụng dữ liệu có sẵn:**
    -   Nếu bạn không muốn chạy lại quá trình thu thập, bạn có thể bỏ qua các cell đó và bắt đầu trực tiếp từ phần đọc và phân tích dữ liệu từ tệp `korean_artists_graph_bfs.json` đã có sẵn trong repository.

## Kết quả và Phân tích

Phần cuối của Notebook trình bày các kết quả phân tích, bao gồm:
-   Thống kê tổng quan về đồ thị (số nút, số cạnh, mật độ).
-   Xác định các nghệ sĩ có tầm ảnh hưởng lớn nhất dựa trên các độ đo trung tâm (centrality measures).
-   Phát hiện các cụm/cộng đồng (community detection) để tìm ra các nhóm nghệ sĩ thường xuyên tương tác với nhau.
-   Trực quan hóa đồ thị để thể hiện rõ các mối quan hệ và cấu trúc mạng lưới.

## Hướng phát triển

-   Tích hợp các thuật toán duyệt khác (DFS, Priority-based) để so sánh hiệu quả.
-   Làm giàu mô hình dữ liệu với nhiều loại nút và quan hệ hơn.
-   Import dữ liệu vào một hệ quản trị CSDL đồ thị chuyên dụng như **Neo4j** hoặc **ArangoDB** để truy vấn và phân tích ở quy mô lớn hơn.
-   Xây dựng một giao diện web tương tác để khám phá đồ thị.
