# KẾ HOẠCH NGHIÊN CỨU SPIKING NEURAL NETWORK (SNN)

## 1. Phạm vi đề tài

### Định hướng chính

Nghiên cứu **Spiking Neural Network (SNN)** theo hướng **tối ưu hiệu quả tính toán và năng lượng thông qua spike sparsity và temporal computation**.

Trọng tâm giai đoạn đầu:

- Tìm hiểu kiến trúc và nguyên lý hoạt động của SNN.
- Nghiên cứu cách huấn luyện SNN, đặc biệt là **surrogate gradient**.
- Nghiên cứu vấn đề **deep SNN training** và gradient error.
- Nghiên cứu **spike sparsity** và ảnh hưởng của surrogate gradient đến sparsity.
- Phân tích trade-off giữa:
  - Accuracy
  - Timestep
  - Spike rate
  - Computation
  - Latency
  - Energy
- Từ literature xác định **research gap** và đề xuất hướng cải tiến có thể thực nghiệm.

> Trong 3 tuần đầu: **chưa tập trung coding**, ưu tiên đọc paper, phân tích phương pháp và xác định research gap.

---

# 2. Mục tiêu sau 3 tuần

Sau khi hoàn thành 5 paper, nhóm cần:

1. Hiểu được nguyên lý hoạt động của SNN.
2. Giải thích được IF/LIF neuron và temporal dynamics.
3. Hiểu tại sao SNN khó train.
4. Hiểu surrogate gradient và các hạn chế của nó.
5. Hiểu các vấn đề khi xây dựng deep SNN.
6. Hiểu gradient error accumulation.
7. Hiểu spike sparsity và sparse surrogate gradient.
8. Biết cách các paper đánh giá hiệu quả của SNN.
9. So sánh được ưu/nhược điểm của các phương pháp.
10. Xác định được **2–3 research directions**.
11. Chọn được **1 research gap** có khả năng triển khai thực nghiệm.

---

# 3. Danh sách 5 paper

## Paper 1 — SNN Fundamentals

**Direct learning-based deep spiking neural networks: a review**

- Tác giả: Guo et al.
- Năm: 2023
- Mục đích: Xây dựng nền tảng kiến thức về SNN.
- Nội dung cần tập trung:
  - SNN vs ANN
  - IF/LIF neuron
  - Spike
  - Temporal dynamics
  - Surrogate gradient
  - Direct training
  - Spike sparsity
  - Computational/energy efficiency

**Link:**  
https://pmc.ncbi.nlm.nih.gov/articles/PMC10313197/

### Cần rút ra

- SNN hoạt động như thế nào?
- Tại sao SNN có spike sparsity?
- Tại sao SNN có tiềm năng tiết kiệm computation/energy?
- Tại sao SNN khó train?
- Những vấn đề nào của SNN vẫn chưa được giải quyết?

---

## Paper 2 — Surrogate Gradient

**Learnable Surrogate Gradient for Direct Training Spiking Neural Networks**

- Tác giả: Lian et al.
- Năm: 2023
- Hội nghị: IJCAI 2023
- Mục đích: Nghiên cứu vấn đề training SNN thông qua surrogate gradient.
- Nội dung cần tập trung:
  - Non-differentiable spike function
  - Surrogate gradient
  - Fixed surrogate gradient
  - Learnable Surrogate Gradient (LSG)

**Link:**  
https://www.ijcai.org/proceedings/2023/335

### Cần rút ra

- Tại sao spike function không khả vi?
- Surrogate gradient giải quyết vấn đề gì?
- Fixed surrogate gradient có hạn chế gì?
- LSG cải thiện vấn đề đó như thế nào?
- Training accuracy được cải thiện với chi phí gì?

---

## Paper 3 — Deep SNN

**Deep Residual Learning in Spiking Neural Networks**

- Tác giả: Fang et al.
- Năm: 2021
- Phương pháp: SEW-ResNet
- Mục đích: Xây dựng và train SNN sâu.
- Nội dung cần tập trung:
  - Deep SNN
  - Vanishing/exploding gradient
  - Residual connection
  - Spike-element-wise residual connection
  - Direct training

**Link:**  
https://arxiv.org/abs/2102.04159

### Cần rút ra

- Tại sao deep SNN khó train?
- Residual connection giải quyết vấn đề gì?
- SEW-ResNet hoạt động như thế nào?
- Accuracy thay đổi như thế nào khi tăng độ sâu?
- Computational cost thay đổi như thế nào?

---

## Paper 4 — Gradient Error

**Surrogate Module Learning: Reduce the Gradient Error Accumulation in Training Spiking Neural Networks**

- Tác giả: Deng et al.
- Năm: 2023
- Hội nghị: ICML 2023
- Mục đích: Giảm gradient error accumulation trong quá trình train SNN.
- Nội dung cần tập trung:
  - Gradient error
  - Gradient error accumulation
  - Surrogate gradient
  - Shortcut path
  - Surrogate Module Learning (SML)

**Link:**  
https://proceedings.mlr.press/v202/deng23d.html

### Cần rút ra

- Gradient error xuất hiện từ đâu?
- Tại sao gradient error tích lũy?
- SML giải quyết vấn đề như thế nào?
- Shortcut path có vai trò gì?
- Paper chứng minh hiệu quả bằng experiment/ablation như thế nào?
- Phương pháp có làm tăng computational cost không?

---

## Paper 5 — Sparsity & Efficiency

**Directly training temporal Spiking Neural Network with sparse surrogate gradient**

- Tác giả: Li et al.
- Năm: 2024
- Journal: Neural Networks
- Mục đích: Duy trì sparsity trong quá trình training SNN.
- Nội dung cần tập trung:
  - Spike sparsity
  - Sparse surrogate gradient
  - Masked Surrogate Gradient (MSG)
  - Temporally Weighted Output (TWO)
  - Spike rate
  - Timestep
  - Efficiency

**Link:**  
https://pubmed.ncbi.nlm.nih.gov/39013289/

### Cần rút ra

- Surrogate gradient ảnh hưởng đến spike sparsity như thế nào?
- MSG hoạt động như thế nào?
- TWO giải quyết vấn đề gì?
- Spike sparsity được đo như thế nào?
- Accuracy và sparsity có trade-off không?
- Phương pháp có thực sự giảm computation/energy không?
- Limitations của paper là gì?

---

