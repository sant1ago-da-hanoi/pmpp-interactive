# PMPP, học và thực hành trọn bộ

[English](README.md) - **Tiếng Việt**

Tài liệu học tương tác cho sách *Programming Massively Parallel Processors* (ấn bản thứ 5),
được thiết kế cho việc tự học cá nhân: các bài học có mô phỏng và animation thuyết minh
trên một web app chạy local, đi kèm các bài lab chạy trên GPU của bạn qua CUDA, HIP hoặc Metal.

Tình trạng (2026-11-21): toàn bộ các chương đã hoàn thiện. Gồm chương 1 đến 24, phụ lục A
và C (bỏ qua B: kiến thức cơ bản về deep learning), cùng ba module bổ sung ngoài sách: X1
sampling kernels, X2 quantised kernels, X3 đọc backend thực tế. Tổng cộng có 79 bài lab,
tương ứng với các cột mốc (milestone) từ M0 đến M10 của dự án (xem `PLAN.md`, mục 5).
Chương 5 được làm thí điểm trước và dùng làm mẫu cho các chương khác. `PROGRESS.md` ghi lại
trạng thái hiện tại và các đầu việc còn dang dở; kế hoạch tổng thể nằm trong `PLAN.md`.

## Chạy app

```sh
./start.sh          # sau đó mở http://localhost:5173
```

Cần Node.js 20 trở lên. Tiến độ học được lưu vào `progress/progress.json`; bạn có thể
commit hoặc đồng bộ file này để chuyển tiến độ giữa các máy.

## Chạy lab

```sh
./verify.sh                         # build + test + benchmark toàn bộ những gì máy này hỗ trợ,
                                    # lưu kết quả vào reports/verify-<time>-<os>-<backends>.md
./verify.sh --quick                 # tương tự như trên, nhưng bỏ qua benchmark
./verify.sh --backend hip           # chỉ chạy một backend (có thể truyền nhiều lần), ví dụ để bỏ qua emulator chậm
python3 labs/pmpp.py doctor         # kiểm tra các backend tìm thấy và thông tin thiết bị
python3 labs/pmpp.py run 5.2        # chạy test cho bài lab 5.2 trên mọi backend
python3 labs/pmpp.py run 5.2 --bench
python3 labs/pmpp.py run 5.2 --backend emu
python3 labs/pmpp.py run 5.2 --solution
```

Cần CMake 3.21 trở lên, trình biên dịch C++17 và Python 3. Hệ thống sẽ tự động nhận diện
các backend; backend nào thiếu sẽ được bỏ qua:

| Backend | Yêu cầu                                 | Trạng thái lời giải tham chiếu                                                                                                                                                                                                         |
|---------|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `metal` | macOS có Xcode command line tools       | verified on Metal (máy Mac của tác giả), ngoại trừ lab 11.5, 12.5 và 18.1 phần B (chỉ hỗ trợ nhóm CUDA)                                                                                                                                |
| `emu`   | không cần gì thêm                       | emulator-checked: CPU emulator cho tập con của CUDA, chỉ kiểm tra logic. Giới hạn thời gian test là 2400 giây (900 giây trên thiết bị thật; có thể ghi đè bằng `PMPP_TEST_TIMEOUT`), vì bộ test của lab X3.2 chạy lần lượt từng thread |
| `race`  | không cần gì thêm                       | cùng emulator trên nhưng chạy các thread đồng thời dưới ThreadSanitizer, áp dụng cho các lab có bật chế độ này                                                                                                                         |
| `cuda`  | NVIDIA CUDA toolkit                     | front-end-checked; **chưa từng chạy thử trên thiết bị CUDA**                                                                                                                                                                           |
| `hip`   | AMD ROCm (`ROCM_PATH` hoặc `/opt/rocm`) | **unverified**, chưa từng biên dịch                                                                                                                                                                                                    |

Nếu bạn có máy dùng CUDA hoặc ROCm: hãy chạy `./verify.sh` trên máy đó và lưu lại file
báo cáo được tạo ra. Báo cáo ghi lại lỗi build và lỗi test đủ chi tiết để có thể dựa vào đó
sửa lỗi.

## Cấu trúc thư mục

```
app/        web app; app/src/chapters/chNN/ chứa bài học của từng chương
content/    kịch bản thuyết minh, audio và manifest
labs/       hệ thống build, runtime, CPU emulator, và labs/chNN/labM_name/{starter,work,solution}
notes/      ghi chú từng chương của sách, kèm số trang
progress/   file lưu tiến độ học của bạn
reports/    các báo cáo kiểm thử (verify)
```

Repo không kèm file PDF của sách và bạn cũng không cần có file này để chạy bất kỳ phần nào.
