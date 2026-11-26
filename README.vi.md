# PMPP, làm xuyên suốt

[English](README.md) · **Tiếng Việt**

Bản học tương tác của *Programming Massively Parallel Processors* (ấn bản 5), viết cho một
người đọc: bài học có mô phỏng và hoạt cảnh thuyết minh trong một web app chạy cục bộ, cùng
các lab chạy trên GPU của bạn qua CUDA, HIP hoặc Metal.

Tình trạng (2026-11-21): mọi chương đã dựng xong. Gồm chương 1 đến 24, phụ lục A và C (bỏ B:
kiến thức cơ bản về deep learning), và ba module sách không có: X1 kernel sampling, X2 kernel
lượng tử hoá, X3 đọc backend thật. Có 79 lab, mang các milestone M0 đến M10 của dự án (xem
`PLAN.md`, mục 5). Chương 5 là bản thử và là mẫu. `PROGRESS.md` ghi trạng thái hiện tại và
việc còn mở; kế hoạch nằm trong `PLAN.md`.

## Chạy app

```sh
./start.sh          # sau đó mở http://localhost:5173
```

Cần Node.js 20 trở lên. Tiến độ được lưu vào `progress/progress.json`; commit hoặc đồng bộ
file này để mang tiến độ sang máy khác.

## Chạy lab

```sh
./verify.sh                         # build + test + benchmark mọi thứ máy này hỗ trợ,
                                    # ghi ra reports/verify-<time>-<os>-<backends>.md
./verify.sh --quick                 # như trên, bỏ benchmark
./verify.sh --backend hip           # chỉ một backend (lặp lại được), ví dụ bỏ qua emulator chậm
python3 labs/pmpp.py doctor         # backend nào được tìm thấy, và thiết bị báo cáo những gì
python3 labs/pmpp.py run 5.2        # test bài làm lab 5.2 trên mọi backend
python3 labs/pmpp.py run 5.2 --bench
python3 labs/pmpp.py run 5.2 --backend emu
python3 labs/pmpp.py run 5.2 --solution
```

Cần CMake 3.21 trở lên, trình biên dịch C++17 và Python 3. Backend được dò tự động; backend
thiếu thì bỏ qua:

| Backend | Cần | Trạng thái lời giải tham chiếu |
|---|---|---|
| `metal` | macOS có Xcode command line tools | verified on Metal (máy Mac của tác giả), trừ lab 11.5, 12.5 và 18.1 phần B, chỉ chạy được trên họ CUDA |
| `emu` | không cần gì thêm | emulator-checked: emulator CPU cho tập con của CUDA, chỉ kiểm tra logic. Giới hạn thời gian test là 2400 s (900 s trên thiết bị; `PMPP_TEST_TIMEOUT` để ghi đè), vì bộ test của lab X3.2 chạy lần lượt từng thread |
| `race` | không cần gì thêm | cùng emulator đó nhưng chạy thread đồng thời dưới ThreadSanitizer, dành cho lab tự chọn tham gia |
| `cuda` | NVIDIA CUDA toolkit | front-end-checked; **chưa từng chạy trên thiết bị CUDA** |
| `hip` | AMD ROCm (`ROCM_PATH` hoặc `/opt/rocm`) | **unverified**, chưa từng biên dịch |

Nếu có máy CUDA hoặc ROCm: chạy `./verify.sh` ở đó và giữ báo cáo nó ghi ra. Báo cáo ghi lỗi
build và lỗi test đủ chi tiết để sửa.

## Bố cục

```
app/        web app; app/src/chapters/chNN/ chứa bài học của một chương
content/    kịch bản thuyết minh, audio và manifest
labs/       hệ thống build, runtime, emulator CPU, và labs/chNN/labM_name/{starter,work,solution}
notes/      ghi chú từng chương của sách, kèm số trang
progress/   file tiến độ của bạn
reports/    báo cáo verify
```

PDF của sách không nằm trong repo và không cần có để chạy bất kỳ phần nào.
