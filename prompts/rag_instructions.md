# Strict Batch Response Instructions

Nhiệm vụ của bạn là đưa ra đáp án cho các câu hỏi trắc nghiệm ở trên một cách cực kỳ ngắn gọn và tuân thủ định dạng nghiêm ngặt.

## Yêu cầu đầu ra:
1. **KHÔNG giải thích dông dài**, không suy luận Chain-of-Thought ra văn bản. Chỉ tập trung chọn đáp án.
2. BẮT BUỘC chỉ xuất ra kết quả đáp án cuối cùng dưới định dạng chính xác sau đây (mỗi câu hỏi nằm trên một dòng riêng biệt):
`<qid>: <đáp án chọn (A/B/C/D...)>`

## Ví dụ định dạng trả về:
test_0001: A
test_0002: C
test_0003: B
test_0004: D
