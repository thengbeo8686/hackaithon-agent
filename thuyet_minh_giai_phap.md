# THUYẾT MINH GIẢI PHÁP: AGENT AI KIỂM CHỨNG HAI LƯỢT ĐỒNG THUẬN (DOUBLE-RUN SHUFFLED RAG AGENT WITH AUTO-TUNING CONSENSUS)

**Đơn vị dự thi:** thengbeo8686
**Bảng thi:** Bảng C - Innovator (HackAIthon 2026)

---

## 1. TỔNG QUAN GIẢI PHÁP
Đối với các mô hình ngôn ngữ lớn có kích thước nhỏ (kích thước $\le$ 5B), hai thách thức lớn nhất khi giải quyết bài toán trắc nghiệm phức tạp là:
1. **Thiên kiến vị trí (Position Bias):** Mô hình có xu hướng ưu tiên chọn các đáp án nằm ở vị trí đầu tiên (A hoặc B) bất kể nội dung câu hỏi.
2. **Ảo giác thông tin & Giới hạn ngữ cảnh (Context Noise):** Tài liệu ngữ cảnh đi kèm quá dài và chứa nhiều thông tin nhiễu, làm loãng sự chú ý của mô hình.

Giải pháp của chúng tôi xây dựng một **Agent AI tự động kiểm chứng và đồng thuận**, tích hợp công nghệ RAG tối ưu hóa để giải quyết triệt để hai vấn đề trên một cách ổn định, chính xác và hiệu quả nhất trên hạ tầng offline.

---

## 2. KIẾN TRÚC & LUỒNG XỬ LÝ (PIPELINE FLOW)

Hệ thống hoạt động theo quy trình khép kín End-to-End gồm 6 bước chính:

```
[Dữ liệu Đầu Vào] 
       │
       ▼
 1. [RAG: Pruning Context & Retrieve Top Chunks] (Sử dụng BGE-M3 cục bộ)
       │
       ▼
 2. [Option Shuffling Mapping] (Tráo đổi vị trí đáp án ngẫu nhiên)
       │
       ▼
 3. [PASS 1: Batched Parallel Inference] (GPU tính song song 2 lượt)
    ├── Lượt 1: Đề bài gốc (Original)
    └── Lượt 2: Đề bài đảo đáp án (Shuffled)
       │
       ▼
 4. [Voting Consensus] (Đối chiếu kết quả sau khi ánh xạ ngược)
    ├── Trùng khớp ──► [Ghi nhận kết quả ngay lập tức]
    └── Khác nhau  ──► 5. [PASS 2: Batched Retry & Majority Vote] (Bầu bầu chọn đa số & Giải quyết hòa phiếu)
       │
       ▼
 6. [Ghi kết quả ra file submission.csv & submission_time.csv]
```

### Chi tiết các bước kỹ thuật:
1. **RAG & Pruning Context (Trích xuất tri thức động):** 
   Hệ thống đọc các đoạn thông tin đi kèm câu hỏi, sử dụng mô hình embedding **BGE-M3** (chạy offline hoàn toàn trong container) để xếp hạng mức độ liên quan và chỉ giữ lại `top_k` ngữ cảnh có độ tương quan cao nhất. Điều này giúp tối ưu hóa token đầu vào và loại bỏ nhiễu thông tin.
2. **Option Shuffling (Khử thiên kiến vị trí):**
   Để loại bỏ Position Bias, hệ thống thực hiện tráo đổi ngẫu nhiên thứ tự các đáp án lựa chọn (A, B, C, D...) trước khi đưa vào mô hình sinh câu trả lời.
3. **Double-Run Parallel Inference (Chạy hai lượt song song):**
   GPU thực hiện sinh câu trả lời cho cả 2 phiên bản (Lượt gốc và Lượt đảo đáp án) trong cùng một tiến trình xử lý song song để tiết kiệm thời gian tối đa.
4. **Voting Consensus (Đối chiếu đồng thuận):**
   Ánh xạ kết quả của Lượt đảo đáp án về chữ cái lựa chọn ban đầu rồi đối chiếu với Lượt gốc. 
   - Nếu trùng khớp: Mô hình tự tin với đáp án $\rightarrow$ Xuất kết quả.
   - Nếu mâu thuẫn: Câu hỏi được đưa vào **Pass 2**.
5. **Pass 2 & Tie-breaking (Giải quyết xung đột):**
   Các câu hỏi mâu thuẫn được chạy lại một lần nữa (2 lượt). Hệ thống tiến hành bầu chọn đa số (Majority Voting) từ 4 kết quả thu được. Nếu xảy ra hòa phiếu, hệ thống áp dụng cơ chế ưu tiên đáp án của lượt chạy gốc ổn định nhất để phân xử (Tie-breaker).
6. **Inference Time Logging (Đo lường thời gian thực tế):**
   Hệ thống đo chính xác thời gian truy xuất embedding và thời gian chạy GPU thực tế chia đều trên từng mẫu dữ liệu để ghi nhận vào cột `time` của file báo cáo.

---

## 3. CÁC ĐIỂM TỐI ƯU CÔNG NGHỆ ĐẶC BIỆT

* **GPU VRAM-Based Auto-Tuning (Tự động thích ứng phần cứng):**
  Hệ thống tích hợp bộ tự động phát hiện dung lượng bộ nhớ (VRAM) của GPU khi khởi động. 
  - Nếu VRAM nhỏ (ví dụ T4 GPU): Hệ thống tự hạ batch size xuống `4` để chống tràn bộ nhớ (OOM).
  - Nếu VRAM lớn (như card 32GB của BTC): Hệ thống tự tăng batch size lên **`16`** để tận dụng tối đa 100% công suất tính toán song song của GPU vật lý.
* **Dynamic Model Path Resolution (Quét đường dẫn động):**
  Code tích hợp bộ nhận diện thư mục thông minh. Dù BTC mount mô hình Qwen vào bất kỳ tên thư mục con nào trong `/models/`, Agent vẫn tự tìm thấy và tải mô hình thành công, loại bỏ hoàn toàn rủi ro crash do sai tên đường dẫn.
* **Môi trường CUDA 12.8:**
  Docker Image được tối ưu hóa trên base image `nvidia/cuda:12.8.0-devel-ubuntu22.04` nhằm tương thích hoàn hảo và tối đa hiệu năng của dòng card Blackwell (RTX 5060Ti) trên máy chủ của BTC.
* **Offline 100% (Zero Internet):**
  Mô hình RAG embedding `BAAI/bge-m3` được tải trước và đóng gói trực tiếp vào Docker image lúc build. Hệ thống vận hành hoàn hảo không cần kết nối internet.

---

## 4. HƯỚNG DẪN VẬN HÀNH DÀNH CHO BTC

### Lệnh chạy Docker Container vật lý:
BTC chỉ cần chạy lệnh tiêu chuẩn sau để mount file test và mô hình vật lý vào container:

```bash
docker run --gpus all \
  -v /path/to/private_test_folder:/code \
  -v /path/to/models_folder:/models \
  thengbeo/hackaithon-agent:latest
```

*Trong đó:*
* Thư mục `/code` chứa file dữ liệu đầu vào `/code/private_test.json`.
* Thư mục `/models` chứa mô hình LLM chính (Ví dụ: `/models/Qwen3.5-4B`).
* Container sẽ tự động thực thi và ghi đè 2 file kết quả đầu ra: `/code/submission.csv` và `/code/submission_time.csv`.
