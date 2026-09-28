# Báo cáo đọc paper: Algorithm-hardware co-design of neuromorphic networks with dual memory pathways

> Báo cáo tiếng Việt dành cho người mới học Spiking Neural Network (SNN). Nội dung đi từ trực giác, kiến thức nền, phương pháp, thí nghiệm, phần cứng, đến đánh giá phản biện. Các số liệu dưới đây được lấy từ paper gốc và Extended Data đi kèm trong file `s42256-026-01255-3.pdf`.

## 0. Thông tin bài báo

| Mục | Thông tin |
|---|---|
| Tiêu đề | **Algorithm-hardware co-design of neuromorphic networks with dual memory pathways** |
| Tác giả | Pengfei Sun, Zhe Su, Jascha Achterberg, Giacomo Indiveri, Dan F. M. Goodman, Danyal Akarca |
| Tạp chí | *Nature Machine Intelligence*, Volume 8 |
| Trang | 901-912, chưa tính các trang Extended Data trong file PDF |
| Ngày nhận | 08/12/2025 |
| Ngày chấp nhận | 11/05/2026 |
| Xuất bản online | 16/06/2026 |
| DOI | [10.1038/s42256-026-01255-3](https://doi.org/10.1038/s42256-026-01255-3) |
| Mã nguồn | [GitHub - Dual_memory_pathways](https://github.com/sunpengfei1122/Dual_memory_pathways) |
| Bản lưu mã nguồn | [Zenodo 10.5281/zenodo.19839869](https://doi.org/10.5281/zenodo.19839869) |
| Giấy phép | Creative Commons Attribution 4.0 (CC BY 4.0) |

## 1. Tóm tắt một câu

Paper đề xuất **DMP-SNN**, một SNN có hai đường xử lý thời gian: đường spike nhanh giữ tính thưa và phản ứng tức thời, còn đường nhớ chậm dùng một vector trạng thái nhỏ để tóm tắt lịch sử; đồng thời paper thiết kế phần cứng riêng để hai đường này chạy song song, nhờ đó đạt độ chính xác cao trên chuỗi dài với ít tham số hơn và có thông lượng/năng lượng tốt hơn các thiết kế phần cứng tham chiếu.

## 2. Năm điều cần nhớ trước khi đọc chi tiết

1. **DMP không bỏ SNN truyền thống.** Nó vẫn dùng neuron LIF và đường feedforward tạo spike.
2. **“Dual memory” không có nghĩa là hai bộ nhớ dài hạn giống nhau.** Một trạng thái là điện thế màng nhanh; trạng thái còn lại là bộ nhớ chậm, liên tục và có số chiều thấp.
3. **Bộ nhớ chậm được dùng chung trong một layer.** Nó không phải một buffer dài cho từng synapse, vì vậy chi phí tăng gần tuyến tính thay vì bậc hai.
4. **Thuật toán và phần cứng được đồng thiết kế.** Cấu trúc thuật toán cố ý tạo ra một phép tính spike thưa và một phép tính memory dense nhưng nhỏ; phần cứng dành dataflow khác nhau cho hai loại phép tính này.
5. **Kết quả phần cứng là post-layout simulation, chưa phải chip DMP đã chế tạo và đo trực tiếp.** Đây là điểm quan trọng khi diễn giải các con số tốc độ và năng lượng.

---

# Phần I - Kiến thức nền cho người mới

## 3. SNN là gì?

Mạng nơ-ron thông thường thường truyền các số thực qua từng layer ở mỗi bước tính toán. SNN mô phỏng gần hơn cách nơ-ron sinh học giao tiếp: neuron chỉ phát một sự kiện rời rạc gọi là **spike** khi trạng thái nội tại vượt ngưỡng.

Ta có thể hình dung một neuron như một chiếc cốc bị rò:

- Mỗi tín hiệu đầu vào làm mực nước tăng.
- Theo thời gian, nước bị rò ra ngoài.
- Khi mực nước vượt ngưỡng, neuron phát spike.
- Nếu đầu vào ngừng đến, ký ức trong “mực nước” phai dần.

Ưu điểm quan trọng là ở nhiều bài toán cảm biến sự kiện, phần lớn neuron không spike ở hầu hết thời điểm. Phần cứng có thể bỏ qua nhiều phép tính bằng 0, giúp giảm năng lượng. Đây được gọi là **event-driven sparsity**, tức tính thưa do chỉ tính khi có sự kiện.

## 4. Neuron LIF trong paper

Paper dùng neuron **Leaky Integrate-and-Fire (LIF)**. Với neuron thứ `i` ở layer `l`, điện thế màng tại thời điểm rời rạc `k` là:

$$
u_i^l[k] = \beta u_i^l[k-1] + I_i^l[k]
$$

Trong đó:

- $u_i^l[k]$: điện thế màng hiện tại;
- $\beta \in (0,1)$: hệ số rò, quyết định ký ức cũ còn lại bao nhiêu;
- $I_i^l[k]$: dòng điện đầu vào mới.

Dòng feedforward được tạo bởi spike từ layer trước:

$$
I^l[k] = W_f s^{l-1}[k]
$$

Neuron phát spike khi vượt ngưỡng:

$$
s_i^l[k] = \Theta(u_i^l[k]-\theta_u)
$$

`Θ` là hàm bước: kết quả bằng 1 nếu vượt ngưỡng, ngược lại bằng 0.

### Vấn đề của LIF khi chuỗi rất dài

Nếu thông tin chỉ được lưu trong điện thế màng thì ảnh hưởng của một sự kiện cũ giảm theo lũy thừa $\beta^{\Delta k}$. Khi khoảng thời gian lớn, giá trị này tiến rất nhanh về 0. Vì vậy một FSNN chỉ có feedforward LIF dễ quên những gì xảy ra hàng trăm bước trước.

Ví dụ: nếu `β = 0.95`, sau 100 bước phần còn lại xấp xỉ $0.95^{100} \approx 0.0059$. Sau 784 bước, ảnh hưởng gần như biến mất.

## 5. Bốn loại kiến trúc mà paper so sánh

### 5.1 FSNN - Feedforward SNN

Mỗi layer nhận spike từ layer trước, tích lũy trong màng LIF và phát spike. Không có vòng hồi tiếp, không có delay dài, không có bộ nhớ phụ.

- Ưu: đơn giản, thưa, dễ triển khai.
- Nhược: ký ức ngắn, yếu trên chuỗi dài.

### 5.2 RSNN - Recurrent SNN

Thêm kết nối từ spike của layer quay lại chính layer đó:

$$
u[k] = \beta u[k-1] + W_f s_{in}[k] + W_r s[k-1]
$$

Nếu layer có `N` neuron, ma trận hồi tiếp $W_r$ có cỡ $N \times N$. Khi `N` tăng, số tham số và số phép tính tăng theo $O(N^2)$.

- Ưu: nhớ dài hơn FSNN.
- Nhược: tốn bộ nhớ trọng số, tăng traffic bộ nhớ, làm giảm lợi thế sparse/event-driven.

### 5.3 DSNN - Delay-based SNN

Spike được trì hoãn một số bước trước khi tới neuron đích. Delay giúp căn chỉnh thông tin với thời điểm quan trọng của bài toán.

- Ưu: tốt cho nhận dạng âm thanh và tín hiệu có cấu trúc thời gian.
- Nhược: delay dài đòi hỏi buffer sâu và metadata thời gian, đặc biệt tốn kém trên phần cứng.

### 5.4 DMP-SNN - Dual Memory Pathway SNN

DMP-SNN giữ nguyên đường spike nhanh và thêm một vector nhớ chậm $m[k]$ có số chiều `d`, trong đó `d << N`.

- Đường nhanh: spike thưa, phản ứng với input hiện tại.
- Đường chậm: vector dense nhưng nhỏ, tóm tắt lịch sử gần đây.
- Vector nhớ được đọc ngược vào điện thế màng như một dòng điện bổ sung.

Đây là đóng góp chính của paper.

## 6. So sánh chi phí lý thuyết

Ký hiệu:

- `M`: số đầu vào của layer;
- `N`: số neuron đầu ra của layer;
- `d`: số chiều bộ nhớ chậm, với `d << N`;
- $\bar d$: độ dài delay/buffer delay;
- `MN`: số trọng số feedforward cơ bản.

| Kiến trúc | Tham số gần đúng theo Hình 1 | Dynamic buffer | Ý nghĩa |
|---|---:|---:|---|
| FSNN | $MN$ | $N$ | Chỉ có feedforward và trạng thái màng |
| RSNN | $MN + N^2$ | $2N$ | Thêm ma trận hồi tiếp bậc hai |
| DSNN | $MN + N$ | $N + N\bar d$ | Thêm delay nhưng cần buffer theo neuron và độ dài delay |
| DMP-SNN | $MN + Nd + d^2 + d + M$ | $N+d$ | Thêm state nhỏ, readout và phép chuyển trạng thái |

Điểm quan trọng không phải DMP “không tốn gì”, mà là phần thêm vào phụ thuộc vào `d`. Khi `d` chỉ bằng khoảng 5-10% `N`, chi phí thấp hơn nhiều so với ma trận $N^2$ của recurrence hoặc buffer $N\bar d$ của delay dài.

---

# Phần II - Paper đang giải quyết vấn đề gì?

## 7. Câu hỏi nghiên cứu

Paper bắt đầu từ một mâu thuẫn:

- Muốn SNN hiểu chuỗi dài, mạng cần giữ ngữ cảnh trong thời gian dài.
- Những cơ chế thường dùng để giữ ngữ cảnh, như recurrence dense hoặc delay dài, lại làm phần cứng tốn bộ nhớ, tốn năng lượng và khó giữ tính event-driven.

Câu hỏi của nhóm tác giả là:

> Có thể tạo một SNN nhớ được thông tin dài hạn mà vẫn giữ đường spike thưa, dùng ít tham số và ánh xạ tốt lên phần cứng hay không?

## 8. Trực giác sinh học

Nhóm tác giả lấy cảm hứng từ sự phân tách nhanh-chậm trong vỏ não:

- Soma tạo spike nhanh.
- Nhánh dendrite và các cơ chế cục bộ có thể tích hợp thông tin trên thang thời gian chậm hơn.
- Các vùng/neuron khác nhau trong não có hằng số thời gian khác nhau.

Paper không cố mô phỏng sinh học chi tiết. Thay vào đó, họ trừu tượng hóa thành một nguyên tắc kỹ thuật:

> Mỗi layer có một đường spike nhanh và một trạng thái chậm nhỏ, cục bộ cho layer, dùng để điều biến hoạt động spike.

## 9. Ba giả thuyết chính

1. Một trạng thái chậm có số chiều thấp có thể giữ đủ ngữ cảnh để cải thiện SNN trên chuỗi dài.
2. Trạng thái chậm giúp gradient đi ngược qua nhiều time step ổn định hơn, nên giải quyết một phần bài toán long-term credit assignment.
3. Nếu thuật toán tách rõ sparse spike path và dense memory path, phần cứng có thể tối ưu riêng hai loại dataflow và chạy chúng song song.

---

# Phần III - DMP-SNN hoạt động như thế nào?

## 10. Luồng xử lý từng bước

Tại mỗi time step `k` của một layer:

1. Layer nhận vector spike $s^{l-1}[k]$ từ layer trước.
2. Spike đi thẳng vào đường feedforward để tạo dòng điện nhanh $I^l[k]$.
3. Cùng vector spike đó được nén thành một scalar $x^l[k]$.
4. Scalar này cập nhật vector nhớ chậm $m^l[k]$.
5. Vector nhớ được chiếu trở lại không gian `N` neuron để tạo dòng nhớ $I_m^l[k]$.
6. Dòng nhanh và dòng nhớ cùng cập nhật điện thế màng.
7. Neuron vượt ngưỡng sẽ phát spike.

Điểm tinh tế là lịch sử dài không được lưu thành toàn bộ chuỗi spike. Nó được nén vào một vài “mode” chậm trong vector $m$.

## 11. Đường nhớ chậm

### 11.1 Dạng liên tục

Paper mô tả bộ nhớ chậm như một hệ state-space tuyến tính:

$$
m'(t) = A m(t) + Bx(t)
$$

- $m(t) \in \mathbb{R}^d$: trạng thái nhớ;
- $x(t)$: input scalar cho memory;
- $A$: ma trận chuyển trạng thái;
- $B$: ma trận đưa input vào state.

Trong thí nghiệm, paper dùng một hiện thực gần với **Legendre Memory Unit (LMU)**. LMU biểu diễn lịch sử input bằng các cơ sở trực giao liên quan đến đa thức Legendre. Trực giác: thay vì ghi lại từng mẫu trong cửa sổ thời gian, LMU lưu các hệ số tóm tắt hình dạng của tín hiệu trong cửa sổ đó.

### 11.2 Rời rạc hóa

Sau khi rời rạc hóa bằng zero-order hold:

$$
\bar A=e^{A\Delta t}, \qquad
\bar B=A^{-1}(e^{A\Delta t}-I)B
$$

Trạng thái được cập nhật:

$$
m^l[k] = \bar A m^l[k-1] + \bar B x^l[k]
$$

Các ma trận trạng thái này thường được cố định trong quá trình training theo mô tả của paper.

## 12. Nén spike thành input cho memory

Paper nén toàn bộ vector spike presynaptic thành một scalar:

$$
x^l[k] = f_x(W_x s^{l-1}[k] + b)
$$

Trong cấu hình chính, $f_x$ là ReLU.

Điều này rất tiết kiệm: thay vì đưa một vector lớn vào memory, mỗi time step chỉ có một giá trị kích thích vector state `d` chiều. Đổi lại, đây cũng có thể là một nút thắt thông tin; phần đánh giá hạn chế sẽ quay lại điểm này.

## 13. Memory tác động lên neuron như thế nào?

Dòng điện nhanh:

$$
I^l[k] = W_f s^{l-1}[k]
$$

Dòng điện từ memory:

$$
I_m^l[k] = W_m m^l[k]
$$

Cập nhật điện thế màng:

$$
u^l[k] = \beta u^l[k-1] + I^l[k] + I_m^l[k]
$$

Như vậy memory không tự tạo toàn bộ biểu diễn. Nó là một **contextual modulator**: cung cấp ngữ cảnh chậm để điều biến phản ứng của đường spike nhanh.

Đây là kết luận được hỗ trợ bởi ablation: khi bỏ $W_f$ và chỉ còn state chậm, độ chính xác rơi về mức đoán ngẫu nhiên trên cả bốn dataset.

## 14. Ba khái niệm rất dễ nhầm

| Ký hiệu | Tên | Nó điều khiển gì? | Không phải là gì? |
|---|---|---|---|
| `d` | Memory dimension / số memory state | Số phần tử trong vector $m[k]$ | Không phải độ dài chuỗi |
| `θ` | State buffer length | Độ dài cửa sổ lịch sử mà LMU tóm tắt | Không phải số neuron |
| `d_s` | Dilation/skip length | Bao nhiêu bước mới cập nhật hoặc bơm lại memory-driven input | Không phải axonal delay của từng kết nối |

Ví dụ: `d = 10`, `θ = 40`, `d_s = 5` nghĩa là layer dùng 10 số để tóm tắt một cửa sổ khoảng 40 bước, và phần memory-driven input chỉ cần làm mới mỗi 5 bước trong thử nghiệm dilation.

## 15. Dilated slow memory

Để giảm số lần tính memory trên phần cứng, paper thử cập nhật thưa:

$$
I_m^l[k] = W_m m^l[k_d], \qquad
k_d = \left\lfloor \frac{k}{d_s} \right\rfloor d_s
$$

Tức là giữa hai mốc cập nhật, neuron dùng trạng thái memory gần nhất. Kết quả:

- S-MNIST và PS-MNIST gần như không bị ảnh hưởng ngay cả khi skip length khoảng 10.
- SHD và SSC giảm sớm hơn khi skip tăng.

Giải thích: pixel stream MNIST là chuỗi đều và dày, nên memory có thể cập nhật thưa. Âm thanh spike bất quy tắc cần phản ứng theo thời gian mịn hơn.

## 16. Vì sao gradient đi xa hơn?

### 16.1 FSNN thuần

Nếu bỏ qua input mới, trạng thái màng tiến hóa gần như:

$$
u[k+1] = \beta u[k]
$$

Gradient từ thời điểm cuối `T` về thời điểm `k` chứa hệ số:

$$
\frac{\partial u[T]}{\partial u[k]} = \beta^{T-k} I_N
$$

Vì $|\beta|<1$, gradient biến mất theo lũy thừa khi đi về quá khứ.

### 16.2 DMP-SNN

Gộp điện thế màng và memory thành trạng thái chung:

$$
z^l[k] =
\begin{bmatrix}
u^l[k] \\
m^l[k]
\end{bmatrix}
$$

Động lực học tuyến tính cục bộ có thể viết với ma trận block trên tam giác:

$$
F =
\begin{bmatrix}
\beta I_N & W_m\bar A \\
0 & \bar A
\end{bmatrix}
$$

và:

$$
\frac{\partial z^l[T]}{\partial z^l[k]} = F^{T-k}
$$

Các mode của màng vẫn suy giảm theo `β`, nhưng memory đóng góp các trị riêng của $\bar A$. LMU được thiết kế để bán kính phổ của $\bar A$ gần 1, nên các mode này suy giảm chậm hơn. Vì memory liên tục bơm ngữ cảnh lại vào màng qua $W_m m[k]$, gradient có thêm một “đường chậm”.

### 16.3 Bằng chứng thực nghiệm

Trong Extended Data Fig. 4, tác giả đặt loss chỉ ở time step cuối và đo gradient quay về hidden layer đầu:

- DMP-SNN giữ được đuôi gradient trên toàn bộ 784 bước của S-MNIST.
- Trên PS-MNIST, gradient hữu ích kéo dài hơn 200 bước.
- FSNN có gradient chủ yếu tập trung gần time step cuối rồi suy giảm nhanh.
- SSC vẫn khó hơn, phù hợp với cấu trúc thời gian đa thang và nhiều nhiễu.

Điều này không chứng minh mọi bài toán gradient dài đều đã được giải quyết, nhưng nó hỗ trợ đúng cơ chế mà paper tuyên bố.

---

# Phần IV - Thiết kế thí nghiệm

## 17. Các dataset

| Dataset | Loại dữ liệu | Bài toán | Chuỗi/thời gian | Vì sao phù hợp để kiểm tra memory? |
|---|---|---|---:|---|
| S-MNIST | Ảnh MNIST đọc tuần tự | Phân loại chữ số | 784 pixel/time step | Cần tích lũy thông tin từ đầu đến cuối ảnh |
| PS-MNIST | MNIST với thứ tự pixel bị hoán vị cố định | Phân loại chữ số | 784 bước | Phá cấu trúc cục bộ, phụ thuộc mạnh vào memory dài hạn |
| SHD | Spike train âm thanh Heidelberg Digits | Nhận dạng chữ số nói | Cấu hình paper dùng 100 bước | Input sự kiện bất quy tắc, cần căn chỉnh thời gian |
| SSC | Spiking Speech Commands | Nhận dạng 35 từ | Cấu hình paper dùng 250 bước | Khó hơn SHD, cấu trúc đa thang và nhiễu |
| DVS Gesture | Camera sự kiện | Nhận dạng cử chỉ | 1.000 bước trong Extended Data | Kiểm tra khả năng gắn DMP vào pipeline convolutional SNN |

Paper mô tả SHD gốc là spike train 700 kênh từ mô hình cochlea; Extended Data Table 4 lại ghi cấu hình đầu vào triển khai là 140 units. Paper không giải thích chi tiết bước ánh xạ 700 xuống 140 ngay trong bảng, nên báo cáo này không suy đoán thêm.

## 18. Hyperparameter chính

| Tham số | SHD | SSC | S-MNIST / PS-MNIST |
|---|---:|---:|---:|
| Input units | 140 | 140 | 1 |
| Simulation time steps | 100 | 250 | 784 |
| Output neurons/classes | 20 | 35 | 10 |
| Hidden spiking neurons | 128 | 128 | 200 |
| Memory states `d` | 10 | 10 | 40 |
| State buffer length `θ` | 40 | 40 | 300 |
| Hidden layers | 2 | 1 và 2 | 2 |
| Spiking threshold `θ_u` | 10 | 10 | 10 |
| Memory input activation `f_x` | ReLU | ReLU | ReLU |
| Axonal delay | Learnable trong thí nghiệm delay | Learnable | Learnable |

Mạng được huấn luyện bằng SpikingJelly và SLAYER. Output là điện thế màng trung bình của layer cuối.

## 19. Các baseline

### Baseline thuật toán

- FSNN: feedforward LIF.
- RSNN: có recurrent connections.
- DSNN: có learnable axonal delays.
- Nhiều mô hình SNN gần đây trong Hình 2c, gồm LSNN, GLIF, PLIF, ASRNN, SRNN, DH, TC-LIF, Rhythm và DelRec.
- State-space model không spike và Spiking SSM trong Extended Data Table 2.

### Baseline phần cứng

- **Loihi2**: nền tảng neuromorphic số; paper dùng kết quả triển khai DSNN có synaptic delay đã công bố.
- **DenRAM**: feedforward SNN analog với dendritic compartments và RRAM, công nghệ 130 nm.
- **ReckOn**: processor RSNN số, hướng tới online learning ở thang nhiều giây.
- **ElfCore**: processor SNN số có cả sparse spike path và dense eligibility-trace path; paper còn xây một biến thể ElfCore làm ablation phần cứng.

---

# Phần V - Kết quả thuật toán

## 20. Bảng kết quả chính

### 20.1 PS-MNIST

| Model | Tham số | Accuracy |
|---|---:|---:|
| FSNN | 42K | 11.30% |
| RSNN | 122K | 71.00% |
| DSNN | 43K | 72.06% |
| DMP-SNN (I) | 61K | **95.50%** |
| DMP-SNN (II) | 102K | **96.65%** |
| DMP-SNN (peak) | 202K | **97.32%** |

Mức đoán ngẫu nhiên là 10%. FSNN gần như không học được PS-MNIST. DMP-SNN tăng rất mạnh, cho thấy bộ nhớ chậm xử lý được phụ thuộc dài sau khi cấu trúc không gian cục bộ đã bị phá bởi phép hoán vị.

### 20.2 S-MNIST

| Model | Tham số | Accuracy |
|---|---:|---:|
| FSNN | 42K | 59.00% |
| RSNN | 122K | 74.00% |
| DSNN | 43K | 88.79% |
| DMP-SNN (I) | 51K | **98.08%** |
| DMP-SNN (II) | 73K | **99.20%** |
| DMP-SNN (peak) | 202K | **99.28%** |

Solution II chỉ kém peak 0.08 điểm phần trăm nhưng dùng khoảng 36% số tham số của cấu hình peak. Đây là ví dụ rõ về điểm vận hành accuracy-efficiency tốt.

### 20.3 SHD

| Model | Tham số | Accuracy |
|---|---:|---:|
| FSNN | 37K | 48.60% |
| RSNN | 70K | 71.40% |
| DSNN | 37K | 90.98% |
| DMP-SNN (I) | 38K | 89.00% |
| DMP-SNN (II) | 40K | **91.23%** |
| DMP-SNN (peak) | 46K | **91.69%** |

Mức ngẫu nhiên là 5%. Trên SHD, delay đã là baseline mạnh. DMP-SNN II vượt DSNN khoảng 0.25 điểm phần trăm với số tham số tăng từ 37K lên 40K.

### 20.4 SSC

Kết quả được báo cáo cho mạng 1 hidden layer / 2 hidden layers.

| Model | Tham số 1L / 2L | Accuracy 1L / 2L |
|---|---:|---:|
| FSNN | 22K / 39K | 26.08% / 38.50% |
| RSNN | 39K / 72K | 50.90% / 60.00% |
| DSNN | 23K / 39K | 60.01% / 69.40% |
| DMP-SNN (I) | 23K / 40K | **65.09% / 69.50%** |
| DMP-SNN (II) | 24K / 42K | **65.37% / 72.90%** |
| DMP-SNN (peak) | 24K / 42K | **65.37% / 72.90%** |

Mức ngẫu nhiên là 2.90%. DMP đặc biệt cải thiện mạnh cấu hình một layer so với DSNN, còn ở hai layer mức tăng so với DSNN là 3.5 điểm phần trăm.

## 21. Solution I, Solution II và peak có nghĩa gì?

- **Solution I:** cấu hình tiết kiệm tham số nhất nhưng vẫn cạnh tranh với baseline mạnh.
- **Solution II:** cấu hình cân bằng accuracy và số tham số.
- **Peak accuracy:** cấu hình ưu tiên độ chính xác cao nhất, có thể dùng nhiều tham số hơn đáng kể.

Vì vậy không nên lấy câu “ít tham số hơn” áp dụng cho mọi cấu hình. Chẳng hạn cấu hình peak trên MNIST dùng 202K tham số. Tuyên bố giảm 40-60% tham số trong abstract nói về so sánh với các SNN SOTA tương đương ở điểm vận hành phù hợp, không phải mọi hàng trong bảng.

## 22. Tốc độ hội tụ và độ ổn định

Hình 2b cho thấy, dưới cùng điều kiện training và số neuron:

- DMP-SNN bắt đầu ở độ chính xác cao hơn.
- Hội tụ nhanh hơn baseline.
- Accuracy cuối cao hơn trên cả bốn dataset.
- Đường trung bình lấy từ `n = 5` lần chạy độc lập; vùng tô là standard error of the mean.

Đây là bằng chứng DMP không chỉ tăng capacity mà còn làm tối ưu dễ hơn.

## 23. Bộ nhớ nhỏ có đủ không?

Hình 2a và Extended Data Fig. 1 sweep hai đại lượng: số memory state `d` và state buffer length `θ`.

Kết quả chung:

- Chỉ một số lượng memory state nhỏ đã tạo mức tăng đáng kể.
- Khi cửa sổ memory vượt khoảng 25% độ dài chuỗi, hiệu năng đã cạnh tranh trên nhiều cấu hình.
- Tăng `d` và `θ` mãi không đảm bảo accuracy tăng tiếp; lợi ích bão hòa khi đã bao phủ đúng thang thời gian của task.

Thông điệp thực hành: `d`, `θ` và số neuron nên được **co-tune**, không nên mặc định càng lớn càng tốt.

## 24. DMP có thể thay delay hoàn toàn không?

Không hẳn. Paper cho thấy hai cơ chế bổ sung cho nhau:

- Memory chậm giữ thành phần dài hạn.
- Delay ngắn căn chỉnh thời gian chính xác.

Khi memory window dài hơn, các delay học được dịch về giá trị ngắn hơn và đuôi delay dài co lại. Ở cấu hình paper nêu với `d = 5` và window `θ = 5`, kết hợp memory với delay vẫn tăng accuracy khoảng 2%. Cần chú ý paper dùng `θ` trong phần này để mô tả độ dài cửa sổ liên quan đến thí nghiệm state/delay, nên không nên nhầm nó với số memory state `d`.

Kết luận của tác giả: delay dài không còn là thành phần bắt buộc; một thiết kế lai gồm memory dài hạn dùng chung và delay ngắn có thể thân thiện phần cứng hơn.

## 25. Ablation quan trọng: bỏ fast path

Extended Data Table 2 so sánh DMP đầy đủ với DMP khi $W_f=0$:

| Dataset | DMP không fast path | DMP đầy đủ |
|---|---:|---:|
| SHD | 5.00%, 37K | 91.23%, 40K |
| SSC | 2.90%, 39K | 72.90%, 42K |
| S-MNIST | 10.00%, 42K | 99.20%, 73K |
| PS-MNIST | 10.00%, 42K | 96.65%, 102K |

Các giá trị 5%, 2.9% và 10% chính là chance level. Điều này chứng minh memory chậm không phải một mạng độc lập đủ để giải bài toán. Nó cần đường kích thích trực tiếp từ input.

## 26. So với state-space model khác

| Dataset | SSM | Spiking SSM | DMP-SNN |
|---|---:|---:|---:|
| SHD | 90.06%, 73K | 85.09%, 73K | **91.23%, 40K** |
| SSC | 67.44%, 75K | 69.13%, 75K | **72.90%, 42K** |
| S-MNIST | 99.03%, 202K | 99.02%, 202K | **99.20%, 73K** |
| PS-MNIST | 97.21%, 202K | 95.49%, 202K | 96.65%, 102K |

DMP tốt hơn về cân bằng accuracy/tham số trên phần lớn trường hợp. Riêng PS-MNIST, SSM không spike đạt 97.21%, cao hơn DMP II 96.65%, nhưng DMP dùng gần một nửa tham số và giữ kiến trúc spike-driven.

## 27. DVS Gesture

Extended Data Table 1 gắn DMP vào layer fully connected spiking đầu tiên của một pipeline gồm hai convolutional spiking layer và hai fully connected spiking layer.

| Model | Accuracy | Tham số |
|---|---:|---:|
| FSNN baseline | 87.12% | 1.07M |
| DMP | **91.32%** | 1.09M |

Tăng 4.20 điểm phần trăm với khoảng 20K tham số thêm, cho thấy DMP có thể đóng vai trò module gắn vào kiến trúc lớn hơn chứ không chỉ hoạt động trên MLP nhỏ.

---

# Phần VI - Đồng thiết kế phần cứng

## 28. “Algorithm-hardware co-design” nghĩa là gì trong paper này?

Không chỉ là viết thuật toán trước rồi đem chạy trên accelerator có sẵn. Hai phía được thiết kế để khớp nhau:

| Đặc tính thuật toán | Hệ quả phần cứng |
|---|---|
| Spike path thưa | Dùng input-stationary để chỉ đọc trọng số ứng với spike khác 0 |
| Memory path dense nhưng nhỏ | Dùng output-stationary để giữ partial sum/neuron state gần compute |
| Memory update tuyến tính | Phân rã phép tính để phá dependency và chạy song song |
| State dùng chung, số chiều thấp | Giữ state on-chip, tránh buffer dài theo từng synapse |
| Các phép LIF, spike và memory nối tiếp | Fuse operator để giảm số lần đọc/ghi SRAM |

Đây là phần mạnh nhất về mặt tư duy hệ thống của bài báo.

## 29. Kiến trúc phần cứng tổng quan

Thiết kế là digital near-memory-compute cho inference cảm biến sự kiện. Hình 4a có bốn đường tính song song:

1. Spike integration path.
2. Memory integration path thứ nhất.
3. Memory integration path thứ hai.
4. Memory update path.

Hai register slot giữ $m[k]$ và $m[k-1]$. Bốn register slot khác tạm giữ điện thế màng của các neuron liên tiếp để fuse phép LIF và vector-matrix multiplication trước khi ghi về neuron SRAM.

## 30. Tối ưu 1 - Phá dependency của dataflow

Nếu tính đúng theo công thức tuần tự, ta phải:

1. cập nhật $m[k]$;
2. tính $W_m m[k]$;
3. mới cập nhật neuron.

Đường memory sẽ trở thành bottleneck. Paper dùng tính tuyến tính để viết lại:

$$
W_m m[k]
= W_m\bar A m[k-1] + W_m\bar B x[k]
= Pm[k-1] + \nu x[k]
$$

với $P=W_m\bar A$ và $\nu=W_m\bar B$ được tiền tính. Hai thành phần có thể chạy song song. Nhờ vậy memory integration không cần chờ toàn bộ memory update hoàn tất.

## 31. Tối ưu 2 - Operator fusion

Thiết kế SNN thông thường có thể đọc/ghi neuron state nhiều lần:

- đọc state để leak;
- ghi tạm;
- đọc lại để cộng spike input;
- đọc lại để cộng memory input;
- ghi kết quả.

DMP hardware fuse các phép leak, spike integration và memory integration thành một tiến trình. Intermediate state được giữ trong register, nên paper cho biết chỉ cần một lần truy cập memory cho mỗi neuron trong toàn chuỗi phép tính tương ứng.

Kết quả là tăng arithmetic intensity: nhiều phép tính hơn cho mỗi byte dữ liệu đọc từ SRAM.

## 32. Tối ưu 3 - Heterogeneous operand stationarity

Hai phép nhân ma trận có tính chất khác nhau:

- Spike integration: input vector rất thưa.
- Memory integration: input vector dense nhưng kích thước nhỏ.

Một dataflow duy nhất không tối ưu cho cả hai. Paper dùng:

- **Input-stationary** cho spike path: giữ/khai thác spike non-zero và đọc các cột trọng số cần thiết.
- **Output-stationary** cho memory path: giữ output/partial sum gần neuron để giảm truy cập neuron memory.

“Heterogeneous” ở đây nghĩa là phần cứng cho phép mỗi path dùng chiến lược tái sử dụng dữ liệu khác nhau.

## 33. Cách đánh giá phần cứng

| Mục | Thiết lập |
|---|---|
| Process | Advanced 22FDX |
| Kiểu đánh giá | Post-layout simulation |
| Công cụ | QuestaSim và Innovus |
| Dataset | Toàn bộ SHD test set |
| Training | Offline |
| Accuracy | Hardware-based inference measurement |
| Baseline khác process | Lấy từ publication và chuẩn hóa theo process/supply voltage |

Ba metric chính:

- **Maximum acceleration factor:** tốc độ xử lý mỗi time step so với real-time 1 ms/time step của SHD.
- **Energy per time step:** năng lượng trung bình cho một bước inference.
- **Area cost:** diện tích core sau place-and-route/post-layout, hướng tới tape-out.

## 34. Kết quả phần cứng

### 34.1 Accuracy dùng để so sánh

| Thiết kế | Accuracy SHD được paper nêu |
|---|---:|
| Loihi2 delay implementation | 88.0% |
| DenRAM | khoảng 87.0% |
| ReckOn | 86.2% |
| ElfCore | 90.3% |
| DMP-SNN hardware | 90.3% |

ElfCore và DMP dùng cùng trained network trong ablation phần cứng, giúp tách ảnh hưởng của operator fusion và heterogeneous stationarity.

### 34.2 Throughput

- DMP-SNN đạt thông lượng cao hơn **trên 4 lần** so với triển khai delay-based số trên Loihi2.
- Cao hơn **trên 1.9 lần** so với kiến trúc recurrent ReckOn.
- Hình 4b cho thấy DMP có acceleration factor cao nhất trong nhóm so sánh.

### 34.3 Energy efficiency

- DMP-SNN có hiệu quả năng lượng cao hơn **trên 5 lần** so với Loihi2 delay implementation và DenRAM theo cách chuẩn hóa của paper.
- Operator fusion và heterogeneous stationarity tạo thêm khoảng **2.5 lần** cải thiện energy efficiency so với biến thể ElfCore dùng cho ablation.

### 34.4 Area

- DMP có area overhead nhỏ so với ElfCore do cần intermediate accumulation buffers.
- Tuy nhiên DMP đạt area efficiency cao hơn **trên 2 lần** so với ReckOn vì không cần lưu ma trận recurrent $N\times N$.

## 35. Scaling khi tăng gấp đôi số neuron LIF

Extended Data Fig. 5 khảo sát cấu hình `DMP_Double`, tăng số neuron lên 256 và tăng MAC array cùng LIF logic 4 lần.

| Metric | Kết quả khi scale |
|---|---|
| Throughput | Gần như giữ nguyên, giảm nhẹ do spike sparsity thấp hơn |
| Tổng năng lượng | Tăng 2.4 lần, thấp hơn tăng trưởng bậc hai khoảng 4 lần |
| Spike-path energy | Tăng 3.4 lần |
| Memory-path energy | Tăng 1.8 lần |
| Memory-update energy | Khoảng 1 lần |
| Output-path energy | Tăng 1.9 lần |
| Tổng area | Tăng 3.1 lần |
| SRAM share | Khoảng 87% area ở DMP_Double |
| Spike path | Khoảng 59% tổng năng lượng ở DMP_Double |

Ý nghĩa: memory vẫn nhỏ khi scale neuron, nên memory path không tăng mạnh bằng spike path. Tuy vậy SRAM trọng số trở thành giới hạn area và spike path trở thành giới hạn năng lượng. Tác giả đề xuất sparse weight representation là hướng tiếp theo.

---

# Phần VII - Đọc từng hình và bảng

## 36. Hình 1 - Ý tưởng DMP

- Hình 1a so sánh FSNN, RSNN, DSNN và DMP-SNN về phương trình, số tham số và buffer.
- Hình 1b nối abstraction sinh học nhanh-chậm với phần cứng hai dataflow.
- Thông điệp: chuyển ký ức dài hạn từ per-connection state sang một shared low-dimensional state.

## 37. Hình 2 - Accuracy và efficiency

- Hình 2a: contour accuracy theo `d` và `θ`; đánh dấu solution I và II.
- Hình 2b: learning curve DMP so với baseline, `n=5`.
- Hình 2c: accuracy theo model size so với nhiều SNN gần đây.
- Thông điệp: DMP đạt điểm Pareto tốt, không chỉ tối đa accuracy bằng cách tăng model.

## 38. Hình 3 - Nhu cầu thời gian phụ thuộc task

- 3a: vision sequence chịu được dilation lớn hơn auditory spike stream.
- 3b: accuracy bão hòa theo số memory state và parameter budget.
- 3c: buffer dài hơn giúp hội tụ nhanh hơn trong kiến trúc kết hợp delay.
- 3d: buffer dài làm delay học được ngắn lại.
- Thông điệp: memory horizon phải phù hợp task; không có một cấu hình tối ưu chung.

## 39. Hình 4 - Phần cứng

- 4a: microarchitecture với bốn computation path.
- 4b: acceleration factor.
- 4c: energy per time step.
- 4d: normalized area cost.
- 4e: biến đổi đại số để phá dependency.
- 4f: operator fusion.
- 4g: input-stationary cho sparse path và output-stationary cho dense path.

## 40. Extended Data

| Mục | Vai trò |
|---|---|
| Extended Data Table 1 | Chứng minh DMP cải thiện DVS Gesture với overhead nhỏ |
| Extended Data Table 2 | So sánh SSM, Spiking SSM, ablation bỏ fast path và DMP |
| Extended Data Table 3 | Liệt kê kích thước mọi tensor trong DMP-SNN |
| Extended Data Table 4 | Hyperparameter theo dataset |
| Extended Data Fig. 1 | Cho thấy cấu hình tốt thường dùng memory nhỏ và buffer tương đối ngắn |
| Extended Data Fig. 2 | Sơ đồ ablation và so sánh Spiking SSM |
| Extended Data Fig. 3 | Buffer dài hơn làm distribution của delay ngắn và hẹp hơn |
| Extended Data Fig. 4 | DMP giữ gradient dài hạn tốt hơn FSNN |
| Extended Data Fig. 5 | Phân tích throughput, energy và area khi scale neuron |

---

# Phần VIII - Đánh giá phản biện

## 41. Điểm mạnh

### 41.1 Một abstraction rõ và có giá trị

Paper không chỉ đề xuất một neuron phức tạp hơn. Nó chuyển câu hỏi từ “làm neuron nhớ lâu bằng cách nào?” sang “làm sao tách state nhanh và state chậm để cả thuật toán lẫn phần cứng đều hưởng lợi?”.

### 41.2 Ablation đúng vào cơ chế

- Bỏ fast path làm accuracy rơi về chance, chứng minh memory chỉ là modulator.
- Đo gradient dưới last-step supervision, kiểm tra đúng long-term credit assignment.
- Sweep memory size, buffer length và dilation, cho thấy trade-off thay vì chỉ báo một điểm tốt nhất.
- So ElfCore và DMP với cùng trained model để cô lập tối ưu phần cứng.

### 41.3 Bao phủ nhiều kiểu dữ liệu thời gian

Paper có dense clocked sequence (MNIST), auditory event stream (SHD/SSC) và vision event stream (DVS Gesture). Điều này mạnh hơn chỉ kiểm tra một benchmark.

### 41.4 Có code và dữ liệu công khai

Code thuật toán và phần cứng được công bố trên GitHub/Zenodo; dataset đều công khai. Điều này làm tăng khả năng tái lập.

### 41.5 Phần cứng phản ánh đúng cấu trúc thuật toán

Ba tối ưu dependency breaking, operator fusion và heterogeneous stationarity đều xuất phát trực tiếp từ phương trình DMP, không phải các mẹo không liên quan gắn thêm sau cùng.

## 42. Hạn chế và điều cần thận trọng

### 42.1 Chưa có chip DMP được chế tạo và đo trực tiếp

Kết quả DMP là post-layout simulation trên 22FDX. Nó đáng tin hơn synthesis sơ bộ, nhưng vẫn khác silicon measurement về variation, clock/power delivery, I/O và các hiệu ứng vật lý thực tế.

### 42.2 So sánh phần cứng không hoàn toàn apples-to-apples

Loihi2, DenRAM và ReckOn lấy số từ các công bố khác, khác process node và điện áp rồi được chuẩn hóa. DenRAM dùng hardware-aware simulation thay vì silicon end-to-end. Vì vậy các hệ số “4x”, “5x” nên được hiểu là kết quả theo phương pháp chuẩn hóa của paper, không phải tất cả chip cùng chạy trong một phòng lab với một bộ đo.

### 42.3 Chỉ đánh giá inference và training offline

Paper huấn luyện mạng offline. Kiến trúc phần cứng được đánh giá cho inference, chưa chứng minh on-chip learning hoặc hiệu quả của backward pass trên hardware.

### 42.4 Benchmark còn nhỏ và tương đối chuẩn hóa

S-MNIST/PS-MNIST là benchmark chuỗi cổ điển, SHD/SSC là speech dataset quy mô hạn chế. DVS Gesture mở rộng phạm vi nhưng vẫn chưa trả lời hiệu năng trên streaming perception lớn, nhiều layer, nhiều sensor hoặc task liên tục hàng giờ.

### 42.5 Hardware prototype chủ yếu ở single-core

Paper nói rõ mở rộng multi-core cần partitioning, mapping và routing, nằm ngoài phạm vi. Scaling study mới tăng layer lên 256 neuron và tăng compute logic; chưa chứng minh hệ thống lớn nhiều core.

### 42.6 Lợi ích dilation phụ thuộc task

Vision sequence chịu cập nhật thưa tốt, nhưng SHD/SSC suy giảm sớm. Vì vậy không phải ứng dụng nào cũng nhận được toàn bộ tiết kiệm switching/memory traffic từ dilation.

### 42.7 Nén về một scalar có thể giới hạn biểu diễn

Input vào memory là $x[k]\in\mathbb{R}^1$. Điều này tiết kiệm nhưng có thể làm mất nhiều kiểu thông tin xảy ra đồng thời. Paper chưa khảo sát sâu memory có nhiều input channel hoặc nhiều nhóm state trên các task phức tạp.

### 42.8 Cần tuning theo task

`d`, `θ`, số neuron và `d_s` phải đồng điều chỉnh. Cấu hình quá nhỏ bỏ mất thời gian liên quan; cấu hình quá lớn tăng cost mà accuracy đã bão hòa.

### 42.9 Tuyên bố ít tham số không đúng cho mọi cấu hình

Solution I/II có lợi rõ, nhưng peak MNIST dùng 202K tham số. Khi đọc abstract, cần gắn tuyên bố 40-60% ít tham số với điểm vận hành SOTA tương đương, không dùng như một mệnh đề tuyệt đối.

### 42.10 Uncertainty chưa được trình bày đồng đều

Hình learning curve báo `n=5` và standard error, nhưng bảng accuracy chính không kèm độ lệch chuẩn cho từng hàng. Các hệ số hardware cũng không có khoảng tin cậy theo cùng cách.

## 43. Những điều paper không tuyên bố

- Không chứng minh DMP giống đầy đủ cơ chế dendrite sinh học.
- Không chứng minh DMP luôn tốt hơn mọi SSM không spike.
- Không chứng minh training trên chip.
- Không chứng minh scaling multi-core lớn.
- Không chứng minh delay không còn hữu ích; paper thực tế cho thấy delay ngắn và memory có thể bổ sung nhau.
- Không cho thấy slow memory có thể thay fast spike path.

---

# Phần IX - Ý nghĩa đối với người học và người triển khai SNN

## 44. Bài học thuật toán

### Bài học 1: State không cần gắn vào mọi kết nối

RSNN phân bố memory qua nhiều recurrent weight; DSNN phân bố memory qua delay buffer. DMP gom phần dài hạn thành state layer-level nhỏ. Đây là một ví dụ của thiết kế low-rank/shared state.

### Bài học 2: Hai thang thời gian tốt hơn kéo dài một thang duy nhất

Chỉ tăng `β` để màng rò chậm hơn có thể giúp nhớ lâu nhưng làm neuron phản ứng ì và gradient vẫn bị ràng buộc bởi một mode. DMP giữ mode nhanh cho phản ứng và mode chậm cho context.

### Bài học 3: State-space model và SNN có thể bổ trợ nhau

SSM không nhất thiết thay SNN. Ở đây SSM nhỏ đóng vai trò memory module bên cạnh spike dynamics.

### Bài học 4: Thiết kế theo Pareto frontier

Solution I/II cho thấy ta nên tìm điểm cân bằng accuracy, parameter, buffer, energy và throughput, thay vì chỉ chọn accuracy cao nhất.

## 45. Bài học phần cứng

### Bài học 1: Sparsity không tự động tạo hiệu quả

Nếu dataflow vẫn đọc nhiều SRAM hoặc ép sparse và dense vào cùng một pipeline, năng lượng có thể vẫn cao. Cần stationarity và memory hierarchy phù hợp.

### Bài học 2: Biến đổi đại số có thể mở khóa song song

Viết lại $W_m(\bar A m + \bar Bx)$ thành $(W_m\bar A)m + (W_m\bar B)x$ cho phép tiền tính và phá dependency. Đây là ví dụ điển hình co-design từ phương trình tới datapath.

### Bài học 3: Register nhỏ có thể tiết kiệm nhiều SRAM access

Operator fusion dùng register để giữ intermediate state. Area register tăng nhẹ nhưng energy do truy cập SRAM giảm đáng kể.

### Bài học 4: Bottleneck thay đổi khi scale

Ở cấu hình lớn hơn, SRAM chiếm 87% area và spike path chiếm 59% energy. Tối ưu memory state thêm nữa có thể không phải bước tốt nhất; sparse weight storage mới là hướng hứa hẹn.

---

# Phần X - Cách tự đọc lại paper theo lộ trình

## 46. Vòng đọc thứ nhất: chỉ hiểu câu chuyện

Đọc theo thứ tự:

1. Abstract ở trang 901.
2. Hình 1 ở trang 903.
3. Bảng 1 ở trang 904.
4. Discussion ở trang 907-908.

Mục tiêu: trả lời được “vấn đề là gì, DMP làm gì, kết quả chính là gì?”. Chưa cần học phương trình LMU.

## 47. Vòng đọc thứ hai: hiểu thuật toán

1. Đọc Methods - Spiking neurons.
2. Đọc Slow memory pathway.
3. Viết lại ba phương trình cốt lõi: nén spike, cập nhật memory, cập nhật màng.
4. Đối chiếu Extended Data Table 3 để kiểm tra shape tensor.
5. Đọc ablation bỏ $W_f$.

Mục tiêu: tự mô tả được một forward pass của DMP-SNN.

## 48. Vòng đọc thứ ba: hiểu gradient

1. Ôn chain rule và vanishing gradient.
2. Đọc phương trình joint state `z=[u;m]`.
3. Hiểu tại sao ma trận block triangular có spectrum gồm `β` và eigenvalue của $\bar A$.
4. Xem Extended Data Fig. 4.

Mục tiêu: giải thích được vì sao đường memory giữ gradient lâu hơn.

## 49. Vòng đọc thứ tư: hiểu phần cứng

1. Xem Hình 4a để nhận diện SRAM, MAC, LIF logic và register.
2. Đọc dependency breaking và tự khai triển phép nhân.
3. So sánh flow có/không operator fusion ở Hình 4f.
4. Học khái niệm input-stationary và output-stationary.
5. Cuối cùng mới đọc biểu đồ throughput/energy/area.

Mục tiêu: nối được từng đặc tính thuật toán với một quyết định dataflow.

---

# Phần XI - Bảng thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Spike | Sự kiện nhị phân neuron phát ra khi vượt ngưỡng |
| SNN | Mạng nơ-ron truyền thông tin bằng spike theo thời gian |
| LIF | Neuron tích lũy input, bị rò và phát spike khi vượt ngưỡng |
| Membrane potential | Trạng thái tích lũy nhanh của neuron |
| Event-driven | Chỉ kích hoạt tính toán khi có sự kiện/spike |
| Sparsity | Phần lớn phần tử hoặc sự kiện bằng 0/vắng mặt |
| Temporal dependency | Đầu ra hiện tại phụ thuộc sự kiện trong quá khứ |
| Credit assignment | Xác định sự kiện/tham số quá khứ chịu trách nhiệm cho loss hiện tại |
| Vanishing gradient | Gradient giảm gần 0 khi truyền ngược qua chuỗi dài |
| Recurrence | Kết nối hồi tiếp từ trạng thái/output cũ vào hiện tại |
| Axonal delay | Độ trễ truyền spike trên kết nối |
| State-space model | Mô hình duy trì một state và cập nhật nó theo động lực học |
| LMU | Memory unit dùng cơ sở Legendre để nén lịch sử trong cửa sổ thời gian |
| Low-dimensional state | Vector state có số chiều nhỏ hơn nhiều số neuron |
| Low-rank/shared state | Thông tin được chia sẻ thay vì có state riêng cho mọi connection |
| Near-memory compute | Đặt compute gần bộ nhớ để giảm di chuyển dữ liệu |
| Dataflow | Cách dữ liệu di chuyển và được giữ lại trong accelerator |
| Operand stationarity | Chọn loại toán hạng được giữ tại chỗ để tái sử dụng |
| Input-stationary | Giữ/ưu tiên tái sử dụng input, phù hợp spike thưa |
| Output-stationary | Giữ partial output gần compute cho đến khi hoàn tất |
| Operator fusion | Gộp nhiều phép toán để tránh đọc/ghi trung gian |
| Post-layout simulation | Mô phỏng sau place-and-route, gần silicon hơn RTL/synthesis nhưng chưa phải chip đo thật |
| Tape-out | Chốt thiết kế để gửi đi chế tạo chip |
| Pareto frontier | Tập cấu hình không thể cải thiện một mục tiêu mà không làm xấu mục tiêu khác |

---

# Phần XII - Kết luận cuối cùng

## 50. Đóng góp thực sự của paper

Đóng góp quan trọng nhất không phải chỉ là thêm một memory vector vào SNN. Paper đưa ra một nguyên tắc thiết kế thống nhất:

> Giữ đường spike nhanh, thưa và trực tiếp; tách ngữ cảnh dài hạn thành một state chậm, nhỏ và dùng chung; sau đó cho hai đường chạy bằng dataflow phần cứng phù hợp với bản chất sparse/dense của từng đường.

Ở phía thuật toán, DMP-SNN đạt kết quả rất mạnh trên S-MNIST, PS-MNIST, SHD và SSC, giữ gradient qua chuỗi dài và thường đạt điểm cân bằng accuracy/tham số tốt hơn FSNN, RSNN, DSNN và các biến thể SSM được so sánh.

Ở phía phần cứng, thiết kế 22FDX post-layout cho thấy lợi thế lớn về throughput và energy theo phương pháp đánh giá của paper. Tuy nhiên, các tuyên bố phần cứng cần được đọc cùng ba điều kiện: DMP chưa được đo trên silicon thật, nhiều baseline lấy từ paper khác rồi chuẩn hóa, và đánh giá tập trung vào inference single-core trên SHD.

## 51. Đánh giá ngắn gọn

| Tiêu chí | Đánh giá |
|---|---|
| Tính mới | Cao: kết hợp shared slow state với co-designed heterogeneous dataflow |
| Độ rõ cơ chế | Tốt: có phương trình, ablation và gradient analysis |
| Kết quả thuật toán | Mạnh trên benchmark chuỗi và event-based được chọn |
| Kết quả phần cứng | Hứa hẹn, nhưng mới ở post-layout simulation |
| Khả năng tái lập | Khá tốt nhờ code, Zenodo và dataset công khai |
| Khả năng mở rộng | Có bằng chứng bước đầu; multi-core và task lớn vẫn là câu hỏi mở |
| Giá trị cho người mới học SNN | Cao: minh họa rõ trade-off giữa memory, sparsity và hardware |

## 52. Nếu chỉ mang theo một mô hình tư duy

Hãy hình dung DMP-SNN như một người vừa phản xạ nhanh vừa có sổ ghi chú ngắn gọn:

- Đường spike là phản xạ: nhanh, thưa, chỉ hoạt động khi có sự kiện.
- Vector memory là sổ ghi chú: nhỏ, cập nhật chậm, tóm tắt điều vừa xảy ra.
- Neuron dùng cả cảm giác hiện tại và sổ ghi chú để quyết định có spike hay không.
- Phần cứng bố trí một lối đi riêng cho phản xạ và một lối đi riêng cho sổ ghi chú, thay vì ép cả hai qua cùng một đường.

Đó là toàn bộ ý tưởng của paper ở mức trực giác, và các phương trình, thí nghiệm cùng thiết kế phần cứng đều được xây quanh ý tưởng này.

---

## Phụ lục A - Checklist tự kiểm tra mức hiểu

Sau khi đọc báo cáo, bạn nên trả lời được:

- [ ] Vì sao điện thế màng LIF không đủ cho chuỗi 784 bước?
- [ ] RSNN và DSNN gây chi phí phần cứng ở đâu?
- [ ] `d`, `θ` và `d_s` khác nhau thế nào?
- [ ] Vì sao DMP vẫn cần $W_f$?
- [ ] Tại sao eigenvalue của $\bar A$ gần 1 giúp gradient?
- [ ] Vì sao vision chịu dilation tốt hơn auditory stream trong thí nghiệm này?
- [ ] Dependency breaking ở phần cứng dựa trên biến đổi đại số nào?
- [ ] Vì sao spike path dùng input-stationary còn memory path dùng output-stationary?
- [ ] Con số 5x energy efficiency có những caveat nào?
- [ ] Vì sao không nên kết luận peak model luôn tiết kiệm tham số?

## Phụ lục B - Nguồn để thực hành tiếp

1. Đọc code chính thức: [sunpengfei1122/Dual_memory_pathways](https://github.com/sunpengfei1122/Dual_memory_pathways).
2. Ôn LIF bằng cách tự mô phỏng một neuron với các giá trị `β` khác nhau.
3. Cài SpikingJelly và chạy một FSNN đơn giản trước khi thêm memory.
4. Vẽ norm của gradient theo time step cho FSNN và DMP trên một chuỗi giả.
5. Sweep `d` và `θ`, không chỉ nhìn accuracy mà ghi cả parameter count và thời gian chạy.
6. Khi đọc kết quả phần cứng, luôn tách bốn mức bằng chứng: analytical estimate, RTL simulation, post-layout simulation và silicon measurement.
