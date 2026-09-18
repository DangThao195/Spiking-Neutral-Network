# BÁO CÁO PHÂN TÍCH PAPER 1: DIRECT LEARNING-BASED DEEP SPIKING NEURAL NETWORKS: A REVIEW

> **Tài liệu tham chiếu:**
> - **Paper:** *Direct learning-based deep spiking neural networks: a review* (Yufei Guo, Xuhui Huang, Zhe Ma - *Frontiers in Neuroscience*, 2023). [PDF File](file:///h:/SNN/Spiking-Neutral-Network/report/fnins-17-1209795.pdf)
> - **Kế hoạch tổng quan:** [Report_01.md](file:///h:/SNN/Spiking-Neutral-Network/report/Report_01.md)

---

## 1. THÔNG TIN TỔNG QUAN BÀI BÁO (OVERVIEW)

- **Tên bài báo:** Direct learning-based deep spiking neural networks: a review
- **Tác giả:** Yufei Guo, Xuhui Huang, Zhe Ma
- **Đơn vị công tác:** Intelligent Science & Technology Academy of CASIC, Beijing, China
- **Tạp chí / Năm:** *Frontiers in Neuroscience*, Vol 17, Article 1209795 (Tháng 6/2023)
- **Mục đích bài báo:** Cung cấp một góc nhìn hệ thống và toàn diện về các phương pháp huấn luyện trực tiếp (Direct Learning) cho Mạng Nơ-ron Xung Sâu (Deep Spiking Neural Networks - Deep SNNs), phân loại theo bản chất giải quyết vấn đề và tận dụng ưu thế của SNN.

---

## 2. NỀN TẢNG SNN & MÔ HÌNH TOÁN HỌC (SNN FUNDAMENTALS & DYNAMICS)

### 2.1. So sánh SNN vs ANN

| Đặc tính | Artificial Neural Network (ANN) | Spiking Neural Network (SNN) |
| :--- | :--- | :--- |
| **Đơn vị truyền tin** | Giá trị thực liên tục (Continuous Real-valued Activation) | Chuỗi tín hiệu xung nhị phân (Binary Spikes $O \in \{0, 1\}$) |
| **Miền tính toán** | Chỉ có miền không gian (Spatial dimension) | Cả miền không gian và thời gian (Spatially-temporal dynamics) |
| **Cơ chế tính toán** | Liên tục đồng bộ (Clock-driven / Dense MAC) | Kích hoạt theo sự kiện (Event-driven / Sparse Addition) |
| **Phép toán chủ đạo** | Nhân-Tích lũy (Multiply-Accumulate - MAC) | Cộng tích lũy (Synaptic Operation - SOP / Addition) |
| **Hiệu quả năng lượng** | Tiêu tốn nhiều năng lượng trên phần cứng thông thường | Tiết kiệm năng lượng vượt trội trên phần cứng Neuromorphic |
| **Độ khả vi** | Khả vi hoàn toàn (Differentiable) | Không khả vi tại điểm phát spike (Non-differentiable) |

---

### 2.2. Mô hình Neuron Leaky Integrate-and-Fire (LIF)

Mô hình nơ-ron LIF rời rạc hóa theo thời gian là mô hình phổ biến nhất được sử dụng trong các SNN huấn luyện trực tiếp:

$$U_l^t = \tau U_l^{t-1} + W_l O_{l-1}^t \quad \text{khi } U_l^t < V_{th}$$

Trong đó:
- $U_l^t$: Điện thế màng (membrane potential) tại timestep $t$ ở lớp thứ $l$.
- $\tau \in (0, 1)$: Hằng số rò rỉ điện thế màng (membrane time leak constant). Nếu $\tau = 1$, mô hình trở thành Integrate-and-Fire (IF).
- $W_l$: Ma trận trọng số kết nối lớp $l-1$ và lớp $l$.
- $O_{l-1}^t$: Đầu ra xung nhị phân từ lớp trước tại timestep $t$.
- $V_{th}$: Ngưỡng phát xung (firing threshold).

**Cơ chế phát spike (Firing Mechanism):**
Khi điện thế màng vượt quá ngưỡng $V_{th}$, nơ-ron phát ra 1 spike nhị phân và điện thế màng được reset về điện thế nghỉ (resting potential):

$$O_l^t = \begin{cases} 1, & \text{nếu } U_l^t \ge V_{th} \\ 0, & \text{ngược lại} \end{cases}$$

---

### 2.3. Bốn tính chất cốt lõi của SNN (4 Key Characteristics)

1. **Rich Spatially-Temporal Dynamics (Động lực học không-thời gian phong phú):** SNN không chỉ xử lý thông tin theo chiều sâu không gian (các lớp mạng) mà còn tích lũy/truyền thông tin qua từng timestep thời gian $t = 1 \dots T$.
2. **High Efficiency (Hiệu quả tính toán cao):** Do đầu ra $O_{l-1}^t \in \{0, 1\}$, phép nhân trọng số $W_l \cdot O_{l-1}^t$ giảm thành phép chọn/cộng trọng số. Khi nơ-ron không phát spike ($O=0$), nơ-ron giữ trạng thái im lặng (silent), không phát sinh tính toán.
3. **Limited Representative Ability (Khả năng biểu diễn bị hạn chế):** 
   - *Lỗi lượng hóa (Quantization error):* Việc ép điện thế màng liên tục thành tín hiệu spike nhị phân 1-bit gây mất mát thông tin (information loss).
   - *Dung lượng thông tin thấp:* Một feature map spike nhị phân tại 1 timestep mang ít thông tin hơn nhiều so với activation giá trị thực của ANN.
4. **Non-differentiability (Tính không khả vi):** Hàm phát spike bước nhảy Heaviside có đạo hàm bằng $0$ tại hầu hết mọi nơi và tiến tới vô cực tại $U = V_{th}$.

---

## 3. NGUYÊN NHÂN SNN TIẾT KIỆM TÍNH TOÁN VÀ NĂNG LƯỢNG

SNN có tiềm năng tiết kiệm năng lượng và chi phí tính toán vượt trội so với ANN nhờ 3 yếu tố nền tảng:

1. **Spike Sparsity (Tính thưa của xung):** Trong một khoảng thời gian, chỉ một tỷ lệ nhỏ các nơ-ron phát spike ($O=1$). Hầu hết nơ-ron ở trạng thái $0$.
2. **Event-driven Computation (Tính toán theo sự kiện):** Trên phần cứng Neuromorphic (như Intel Loihi, Tianjic), phép tính chỉ được kích hoạt khi có sự kiện spike truyền đến. Nơ-ron yên lặng không tốn điện năng tính toán chủ động.
3. **Thay thế FLOPs bằng SOPs (Synaptic Operations):**
   - Trong ANN: Phép nhân tích lũy $\text{MAC} = a \cdot w + b$ tốn năng lượng cao (ví dụ: 32-bit Float MAC tốn $\sim 4.6\text{pJ}$).
   - Trong SNN: Do $O \in \{0,1\}$, phép nhân biến thành phép cộng chỉ khi $O=1$ ($\text{SOP} = w + b$). Phép cộng 32-bit Float chỉ tốn $\sim 0.9\text{pJ}$ (tiết kiệm hơn 5 lần năng lượng trên mỗi phép tính).

---

## 4. TẠI SAO SNN KHÓ HUẤN LUYỆN? (TRAINING DIFFICULTIES)

Huấn luyện SNN sâu theo phương pháp lan truyền ngược (Backpropagation) gặp các rào cản kỹ thuật nghiêm trọng:

### 4.1. Vấn đề đạo hàm làm gián đoạn Gradient (Non-differentiable Firing Problem)
Đạo hàm lan truyền ngược tại lớp $l$ theo thuật toán Lan truyền ngược qua thời gian (BPTT) được tính như sau:

$$\frac{\partial L}{\partial W_l} = \sum_{t} \left( \frac{\partial L}{\partial O_l^t} \frac{\partial O_l^t}{\partial U_l^t} + \frac{\partial L}{\partial U_l^{t+1}} \frac{\partial U_l^{t+1}}{\partial U_l^t} \right) \frac{\partial U_l^t}{\partial W_l}$$

Thành phần $\frac{\partial O_l^t}{\partial U_l^t}$ chính là đạo hàm của hàm bước nhảy Heaviside:
- $\frac{\partial O_l^t}{\partial U_l^t} = 0$ tại mọi $U_l^t \neq V_{th}$.
- $\frac{\partial O_l^t}{\partial U_l^t} = \infty$ tại $U_l^t = V_{th}$.

Hệ quả: Cập nhật gradient descent $W_l \leftarrow W_l - \eta \frac{\partial L}{\partial W_l}$ sẽ **bị đóng băng (freeze - không cập nhật được)** hoặc **bùng nổ tới vô cực (explode)**.

---

### 4.2. Gradient Explosion / Vanishing & Gradient Error Accumulation
- Khi thay thế hàm phát spike bằng các hàm xấp xỉ liên tục (Surrogate Gradient - SG như `tanh`, `sigmoid`), các hàm này có xu hướng thu nhỏ đạo hàm ở xa đỉnh.
- Khi lan truyền qua nhiều lớp không gian (Deep network) và nhiều timestep thời gian ($T$), gradient bị triệt tiêu dần (**Gradient Vanishing**) hoặc bùng nổ quá đà (**Gradient Explosion**).
- Thêm vào đó, khoảng cách giữa đạo hàm thực (bằng 0) và đạo hàm surrogate tạo ra **Gradient Error (Lỗi gradient)**. Qua nhiều lớp, lỗi này tích lũy dần làm mạng rơi vào điểm cực trị xấu hoặc không thể hội tụ.

---

## 5. PHÂN LOẠI CÁC PHƯƠNG PHÁP HUẤN LUYỆN TRỰC TIẾP (DIRECT LEARNING TAXONOMY)

Bài báo phân loại các công trình SNN huấn luyện trực tiếp theo 3 trục giải pháp chính:

```
Direct Learning-based Deep SNNs
 ├── 1. Accuracy Improvement Methods (Cải thiện độ chính xác)
 │    ├── Nâng cao khả năng biểu diễn (Representative Capabilities)
 │    │    ├── Cấp độ Nơ-ron (Neuron level): Adaptive/Learnable threshold, learnable leak, multi-level firing, GLIF, InfLoR (burst)
 │    │    ├── Cấp độ Kiến trúc (Structure level): SEW-ResNet, MS-ResNet (Pre-activation), NAS, Attention
 │    │    └── Cấp độ Kỹ thuật (Training technique level): IM-Loss (Max Entropy), RecDis-SNN, Knowledge Distillation
 │    └── Giảm bớt khó khăn huấn luyện (Relieving Training Difficulties)
 │         ├── Thiết kế Surrogate Gradient tốt hơn: Fixed SG, Dynamic SG, Learnable SG, DSR
 │         └── Giảm Gradient Explosion/Vanishing: ReL-PSP, PreAct ResNet, tdBN, BNTT, TET Loss, Joint A-SNN
 ├── 2. Efficiency Improvement Methods (Tăng cường hiệu quả tính toán)
 │    ├── Nén mạng (Network Compression): Spatio-temporal pruning, Temporal pruning, NAS, Distillation
 │    └── SNN thưa (Sparse SNNs): Adaptive neurons (ASNN), Correlation regularizer, Activity loss (L1/L2)
 └── 3. Temporal Dynamics Utilization Methods (Khai thác động lực học thời gian)
      ├── Học chuỗi (Sequential Learning): Speech recognition, HAR, Recurrent SNNs (RSNN)
      └── Kết hợp với Camera Neuromorphic (DVS): Optical flow, Depth estimation, Frame interpolation (5000 FPS), Pose tracking
```

### Key Highlights của các phương pháp tiêu biểu:
1. **Surrogate Gradient (SG):** Dùng hàm liên tục $\phi(x)$ (ví dụ: Sigmoid, Arctan, Piecewise linear) để tính gradient ở bước lan truyền ngược:
   - *Fixed SG:* Đạo hàm cố định trong suốt quá trình train.
   - *Dynamic SG:* Hệ số dốc $K(i)$ biến thiên theo epoch $i$ (mềm ở đầu epoch để cập nhật trọng số, dốc ở cuối epoch để tiệm cận hàm spike thực).
   - *Learnable SG:* Học hình dạng đạo hàm tối ưu bằng kỹ thuật sai số hữu hạn hoặc NAS.
2. **Kiến trúc ResNet cho SNN:**
   - *SEW-ResNet (Spike-Element-Wise):* Chuyển phép cộng sau nơ-ron thành phép cộng trước nơ-ron để tránh mất mát spike.
   - *MS-ResNet (Membrane Shortcut):* Dùng Pre-activation form để bảo toàn thông tin điện thế màng thực trước khi chuyển thành spike, duy trì cơ chế nhân-cộng chuyển thành phép cộng.
3. **Kỹ thuật Batch Normalization theo thời gian:**
   - *tdBN:* Thêm chiều thời gian vào chuẩn hóa BN.
   - *BNTT / TEBN:* Dùng bộ tham số BN riêng biệt cho từng timestep để giải quyết sự biến động phân bố điện thế màng theo thời gian.

---

## 6. VẤN ĐỀ CÒN TỒN ĐỌNG VÀ THÁCH THỨC TƯƠNG LAI (REMAINING CHALLENGES & RESEARCH GAPS)

Đáp ứng trực tiếp câu hỏi: **"Vấn đề của bài báo này còn tồn đọng là gì?"**, tác giả Guo et al. (2023) đã chỉ ra 6 hạn chế và bài toán mở lớn mà cộng đồng SNN chưa giải quyết được:

### 1. Thiếu công cụ/lý thuyết đo lường chính xác Dung lượng thông tin (Lack of measurement of information capacity)
- Hiện tại chưa có công thức lý thuyết toán học nào tính toán chính xác **dung lượng thông tin (information capacity)** bị tổn thất khi đi qua các lớp spike map nhị phân $\{0, 1\}$.
- Hầu hết kiến trúc SNN hiện nay vẫn bê nguyên xi từ ANN (như ResNet, VGG), vốn được thiết kế cho tín hiệu liên tục chứ không tối ưu cho SNN.
- *Hướng gợi mở:* Cần nghiên cứu cấu trúc nơ-ron/mạng chuyên biệt cho SNN (ví dụ: Ternary spike $\{-1, 0, 1\}$ để tăng gấp đôi dung lượng thông tin mà vẫn giữ được tính toán không cần phép nhân).

### 2. Khó khăn tối ưu hóa nội tại chưa được giải quyết triệt để (Inherent optimization difficulties)
- Mặc dù việc dùng Surrogate Gradient giúp train được SNN, việc tối ưu trên không gian rời rạc vẫn cực kỳ khó khăn.
- Khi tăng số lượng timestep $T$ hoặc tăng chiều sâu mạng, hiện tượng bùng nổ/sụt giảm gradient và sự tích tụ lỗi gradient (**gradient error accumulation**) vẫn trở nên nghiêm trọng, làm mạng khó hội tụ hoặc sụt giảm độ chính xác mạnh.

### 3. Sự đánh đổi giữa Độ chính xác (Accuracy) và Tính linh hoạt/Đơn giản (Versatility vs Efficiency Trade-off)
- Nhiều nghiên cứu cố gắng tăng Accuracy bằng cách thiết kế mô hình nơ-ron rất phức tạp (nhiều ngưỡng, gating phức tạp, truyền tín hiệu số thực trong giai đoạn forward).
- Điều này làm **mất đi ưu thế tiết kiệm năng lượng và tính toán event-driven ban đầu của SNN**, khiến mô hình không thể triển khai hiệu quả trên phần cứng Neuromorphic.
- Cần phải tìm ra sự cân bằng giữa Accuracy và tính đơn giản/linh hoạt (Versatility) để ứng dụng thực tế.

### 4. Ngộ nhận trong cách tối ưu hiệu năng tính toán (Misconception of SNN Efficiency Optimization)
- Trong ANN, giảm dung lượng mô hình (Pruning nơ-ron/trọng số) giúp giảm FLOPs. Tuy nhiên, trên phần cứng Neuromorphic, chi phí năng lượng phụ thuộc vào **Số lượng xung phát ra (Spike count/rate)** chứ không chỉ phụ thuộc vào kích thước mạng.
- Kỹ thuật tỉa nơ-ron không hoạt động (inactive pruning) kiểu ANN không thực sự làm giảm năng lượng SNN nếu số lượng spike ở các nơ-ron còn lại vẫn cao.
- *Ý tưởng mới:* Việc mở rộng kích thước mạng (enlarge network) nhưng ép tỷ lệ phát spike cực thấp (ultra-low spike rate) có thể vừa tăng độ chính xác vừa đạt hiệu quả năng lượng cao hơn.

### 5. Khai thác Temporal Dynamics chưa sâu & Thiếu lý thuyết giải thích được (Explainable SNN)
- Các nghiên cứu tận dụng động lực học thời gian hiện tại mới chỉ dừng lại ở các bài toán chuỗi đơn giản (Speech, HAR) hoặc dữ liệu từ DVS Camera.
- Chưa có các nghiên cứu lý thuyết học máy giải thích được (Explainable Machine Learning) cho SNN để giải thích tại sao và khi nào SNN vượt trội hơn ANN về mặt thời gian.

### 6. Khả năng ứng dụng thực tế và kỷ nguyên Green AI còn hạn chế (Limited Real-world Applications & Green AI)
- Dù SNN xuất sắc trên lý thuyết về tiết kiệm năng lượng, các ứng dụng thực tế trên quy mô lớn (Large Language Models, Autonomous Driving phức tạp, Robot) vẫn còn rất hạn chế so với ANN.
- Việc kết hợp SNN vào làn sóng **Green Artificial Intelligence (Green AI)** để giảm lượng phát thải carbon của AI đòi hỏi sự phát triển đồng bộ từ thuật toán đến chip Neuromorphic phần cứng thương mại.

---

## 7. ĐÁNH GIÁ TỔNG KẾT VÀ HƯỚNG ĐỀ XUẤT CHO KẾ HOẠCH NGHIÊN CỨU

Dựa trên phân tích Paper 1 và mục tiêu 3 tuần trong [Report_01.md](file:///h:/SNN/Spiking-Neutral-Network/report/Report_01.md), các hướng đi tiềm năng có thể lựa chọn cho giai đoạn thực nghiệm bao gồm:

1. **Nghiên cứu Sparse Surrogate Gradient / Loss Regularization:** Ép mạng đạt độ thưa spike cao (high spike sparsity) mà không làm suy giảm Accuracy (kế thừa ý tưởng từ TET loss, IM-Loss, RecDis-SNN).
2. **Tối ưu Gradient Error Accumulation:** Giảm lỗi tích lũy gradient khi train deep SNN với $T$ nhỏ ($T \le 4$).
3. **Cân bằng Trade-off giữa Timestep - Accuracy - Energy:** Thiết kế cơ chế điều chỉnh ngưỡng tĩnh/động linh hoạt để đạt được sự tối ưu về số lượng phép tính SOPs.

---
*Báo cáo được tổng hợp tự động và lưu tại [Report_p1.md](file:///h:/SNN/Spiking-Neutral-Network/report/Report_p1.md).*
