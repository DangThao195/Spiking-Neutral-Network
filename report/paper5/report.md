# Báo cáo Paper 5: Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient

**Tác giả:** Yang Li, Feifei Zhao, Dongcheng Zhao, Yi Zeng  
**Năm:** 2024  
**Nguồn:** Neural Networks; bản preprint arXiv:2406.19645v1, ngày 28/06/2024  
**Mục tiêu:** huấn luyện trực tiếp SNN bằng surrogate gradient nhưng vẫn duy trì tính thưa của gradient và khai thác tốt thông tin theo thời gian.

## 1. Tóm tắt nội dung

Spiking Neural Network (SNN) truyền tín hiệu bằng các spike nhị phân theo nhiều timestep. Tính event-driven và số lượng spike thấp đem lại tiềm năng tiết kiệm năng lượng, đặc biệt trên phần cứng neuromorphic. Tuy nhiên, hàm phát spike dạng Heaviside không khả vi, khiến việc huấn luyện trực tiếp bằng backpropagation gặp khó khăn.

Surrogate gradient (SG) giải quyết vấn đề này bằng cách dùng một hàm trơn thay cho đạo hàm của hàm spike trong bước backward. SG giúp SNN học được, nhưng làm gradient trở nên dày hơn gradient thật của quá trình phát spike. SG quá rộng có thể cập nhật nhiều tham số bằng hướng gradient không phù hợp, gây **gradient mismatch**; SG quá hẹp lại làm gradient gần như bằng 0, gây **gradient vanishing**.

Paper đề xuất hai kỹ thuật:

- **Masked Surrogate Gradient (MSG):** tạo mặt nạ Bernoulli để chỉ giữ lại một phần gradient surrogate, nhờ đó kết hợp khả năng tối ưu của SG với tính thưa của gradient gốc.
- **Temporally Weighted Output (TWO):** gán trọng số thích nghi cho output ở từng timestep dựa trên độ chính xác lịch sử của timestep đó, giúp decoder nhấn mạnh các thời điểm có ích hơn.

Kết quả cho thấy MSG và TWO cải thiện độ chính xác trên cả dữ liệu ảnh tĩnh và dữ liệu neuromorphic, trong khi firing rate và thời gian huấn luyện gần với baseline.

## 2. Kiến thức nền

### 2.1. Nơ-ron LIF

LIF (Leaky Integrate-and-Fire) tích lũy dòng điện đầu vào vào điện thế màng, làm điện thế rò dần theo thời gian và phát spike khi vượt ngưỡng. Một dạng rời rạc của mô hình trong paper là:

$$
I^{(l)}[t+1] = W^{(l)}S^{(l-1)}[t+1]
$$

$$
V^{(l)}[t+1] = V^{(l)}[t] - \frac{1}{\tau}\left(I^{(l)}[t+1]-V^{(l)}[t]\right)
$$

$$
S^{(l)}[t+1] = \Theta\left(V^{(l)}[t+1]-V_{th}\right),
\qquad V^{(l)}[t+1] \leftarrow V^{(l)}[t+1](1-S^{(l)}[t+1]).
$$

Paper đặt $V_{th}=0.5$ và $\tau=2$. $S[t]$ là spike nhị phân, còn $V[t]$ mang trạng thái lịch sử nên đầu ra phụ thuộc cả không gian lẫn thời gian.

### 2.2. Surrogate gradient

Đạo hàm của $S[t]=\Theta(V[t]-V_{th})$ bằng 0 ở hầu hết vị trí và không thuận tiện cho gradient descent. Paper dùng các hàm thay thế như Arctan gradient:

$$
\frac{\partial S[t]}{\partial V[t]} = \frac{\alpha}{1+\left(\frac{\pi}{2}\alpha(V[t]-V_{th})\right)^2}
$$

và Piecewise Linear gradient. Forward vẫn dùng spike thật; chỉ backward dùng hàm thay thế.

## 3. Phương pháp đề xuất

### 3.1. Masked Surrogate Gradient (MSG)

Gọi $G_{SG}=\partial L/\partial W$ là gradient tính bằng surrogate gradient. Paper sinh một mặt nạ nhị phân $M_b$ theo:

$$
M_b \sim \mathrm{Bernoulli}(1-p),
$$

trong đó $p$ là xác suất che gradient. Gradient cập nhật là:

$$
G = G_{SG}\odot M_b,
\qquad W\leftarrow W-\eta G.
$$

Với $p=0$, MSG trở thành SG thông thường. Khi $p$ tăng, nhiều phần tử gradient bị đặt thành 0 và gradient thưa hơn. Mặt nạ được tạo theo từng layer, minibatch và epoch; vì vậy các neuron có cơ hội được cập nhật tương tự nhau.

Trực giác của MSG là SG không bị loại bỏ hoàn toàn mà chỉ được dùng cho một tập con tham số ở mỗi lần cập nhật. Các phần còn lại tạm thời không cập nhật, làm giảm các cập nhật do surrogate gradient gây nhiễu. Theo phân tích trong paper, masking làm tăng phương sai gradient:

$$
\mathrm{Var}(G)=\sigma^2+\frac{p}{1-p}(\mu^2+\sigma^2).
$$

Tác giả diễn giải điều này như một dạng regularization mạnh, có thể giúp mô hình thoát khỏi saddle point hoặc local minimum không phù hợp. Lưu ý: trong mô tả công thức, paper không luôn trình bày nhất quán việc có chuẩn hóa mask bởi $(1-p)$ hay không; vì vậy khi tái lập cần kiểm tra mã nguồn hoặc cấu hình thực tế.

### 3.2. Temporally Weighted Output (TWO)

Ở output layer, paper không cho neuron phát spike mà tích lũy synaptic current. Với $T$ timestep, output được tính từ các dòng điện $I^{(L)}[t]$:

$$
y[t]=f_t I^{(L)}[t].
$$

Ban đầu $f_t=1/T$. Sau mỗi giai đoạn huấn luyện, paper đo độ chính xác của từng timestep. Gọi $\Delta f_t$ là tỷ lệ mẫu mà output riêng tại timestep $t$ dự đoán đúng:

$$
\Delta f_t=\frac{\sum_{i=1}^{N}[y_i[t]=\hat y_i]}{N|y|}.
$$

Sau đó cập nhật bằng low-pass filtering:

$$
f_t\leftarrow \beta f_t+(1-\beta)\Delta f_t.
$$

Những timestep thường phân loại đúng sẽ có trọng số lớn hơn. TWO không thêm tham số trainable vào mạng và trọng số được cố định khi inference, nên không làm tăng đáng kể gánh nặng backpropagation.

## 4. Quy trình huấn luyện

1. Nhận input và nhãn.
2. Chạy SNN qua $T$ timestep.
3. Tính output tổng hợp bằng các trọng số $f_t$.
4. Đánh giá độ chính xác riêng của từng timestep và cập nhật TWO.
5. Tính loss phân loại.
6. Backpropagation bằng surrogate gradient.
7. Sinh mask Bernoulli và nhân mask với gradient.
8. Cập nhật tham số bằng AdamW.

Input ảnh tĩnh được đưa trực tiếp vào mạng rồi mã hóa thành spike ở layer đầu. Với dữ liệu neuromorphic, dữ liệu được chuyển thành chuỗi frame. Paper dùng learning rate ban đầu 0.05, cosine decay và thường huấn luyện 300 epoch.

## 5. Thiết kế và kết quả thực nghiệm

Paper thử nghiệm trên CIFAR10, CIFAR100, ImageNet, DVS-CIFAR10 và N-Caltech101, với CIFARNet, VGGSNN, ResNet18 và Sew-ResNet34. Kết quả tiêu biểu:

| Dataset và mô hình | Timestep | Accuracy |
| --- | ---: | ---: |
| CIFAR10, CIFARNet, không AutoAugment | 4 | 94.30% ± 0.22 |
| CIFAR10, CIFARNet, có AutoAugment | 4 | 95.40% ± 0.12 |
| CIFAR100, CIFARNet, có AutoAugment | 4 | 77.48% ± 0.14 |
| DVS-CIFAR10, ResNet18 | 10 | 79.35% ± 0.34 |
| DVS-CIFAR10, VGGSNN | 10 | 83.97% ± 0.12 |
| N-Caltech101, ResNet18 | 10 | 70.81% ± 0.31 |
| N-Caltech101, VGGSNN | 10 | 77.70% ± 0.15 |

Trong bảng so sánh, VGGSNN trên DVS-CIFAR10 đạt 83.97%, cao hơn TET 83.17% trong cấu hình được báo cáo. Trên ImageNet, Sew-ResNet34 đạt 67.46% với 4 timestep, thấp hơn TET 68.00%, cho thấy MSG không phải lúc nào cũng tốt nhất.

### 5.1. Ablation MSG và TWO

| Dataset / mô hình | Baseline | TWO | MSG | MSG + TWO |
| --- | ---: | ---: | ---: | ---: |
| CIFAR100 / ResNet18 | 68.79% | 68.98% | 69.08% | 69.21% |
| DVS-CIFAR10 / ResNet18 | 72.90% | 74.40% | 78.30% | 79.10% |
| DVS-CIFAR10 / VGGSNN | 82.70% | 83.20% | 83.30% | 83.90% |

MSG thường tạo cải thiện lớn hơn TWO trên DVS-CIFAR10; TWO đặc biệt hữu ích khi dữ liệu có temporal information phong phú và có nhiều timestep.

### 5.2. Ảnh hưởng của xác suất mask

Kết quả quét $p$ cho thấy hiệu năng thường tốt nhất trong khoảng 0.4 đến 0.6. Khi $p$ gần 0, phương pháp gần với SG thường và vẫn chịu gradient mismatch. Khi $p$ gần 1, phần lớn gradient bị loại bỏ và mô hình gặp gradient vanishing. Vì vậy $p$ tạo ra trade-off giữa khả năng học và độ thưa gradient, không phải càng mask nhiều càng tốt.

### 5.3. Spike rate và thời gian huấn luyện

Theo Figure 6, firing rate của MSG gần với vanilla ở các layer; MSG thay đổi backward chứ không thêm spike vào forward. Thời gian mỗi epoch cũng gần baseline, dù một số cấu hình MSG lâu hơn một chút. Đây là bằng chứng rằng MSG không làm tăng rõ rệt hoạt động spike trong thí nghiệm.

Tuy nhiên, paper đo training time trên NVIDIA A100 và không đo điện năng, số phép toán synapse thực tế hoặc latency trên chip neuromorphic. Vì vậy không thể kết luận trực tiếp rằng MSG đã giảm energy consumption; chỉ có thể nói phương pháp bảo toàn sparsity và có tiềm năng phù hợp với phần cứng event-driven.

## 6. Trả lời các câu hỏi trong kế hoạch nghiên cứu

### 6.1. Surrogate gradient ảnh hưởng đến spike sparsity như thế nào?

Surrogate gradient chủ yếu được dùng trong backward nên không trực tiếp làm neuron phát thêm spike ở forward. Tuy nhiên, SG có support rộng và giá trị khác 0 ở nhiều membrane potential, khiến nhiều tham số được cập nhật hơn gradient thật của hàm spike. Các cập nhật không phù hợp có thể thay đổi trọng số, membrane potential và firing pattern trong các epoch sau; do đó gián tiếp làm thay đổi spike rate và làm mất lợi thế sparse trong quá trình tối ưu.

MSG xử lý sparsity ở gradient bằng mask, còn không thay đổi luật forward của LIF. Vì vậy nó bảo toàn firing rate tốt hơn so với việc thay đổi neuron để ép sparsity trực tiếp.

### 6.2. MSG hoạt động như thế nào?

MSG sinh mặt nạ nhị phân cùng kích thước gradient, với xác suất giữ gradient là $1-p$. Gradient surrogate sau đó được nhân với mask trước khi optimizer cập nhật trọng số. Một phần tham số cập nhật bình thường, phần còn lại nhận gradient bằng 0 ở lần đó. Việc này bổ sung sparsity cho SG, giảm gradient mismatch và hoạt động như regularization ngẫu nhiên.

### 6.3. TWO giải quyết vấn đề gì?

Output ở từng timestep không có mức độ hữu ích như nhau. Trung bình đều các output có thể làm loãng các thời điểm đã hình thành dự đoán đúng, nhất là trong dữ liệu event-based. TWO thống kê accuracy lịch sử của từng timestep rồi tăng trọng số cho timestep đáng tin cậy hơn. Nó cải thiện decoding, không phải một cơ chế sửa gradient.

### 6.4. Spike sparsity được đo như thế nào?

Paper dùng **firing rate** theo layer để đánh giá mức hoạt động spike. Về bản chất, firing rate có thể viết là:

$$
r_l=\frac{\text{số spike của layer }l}{\text{tổng số neuron và timestep được quan sát}}.
$$

Paper so sánh firing rate giữa vanilla và MSG trong VGGSNN, đồng thời báo cáo training time mỗi epoch. Paper không báo cáo đầy đủ các chỉ số như số synaptic operation thực tế, energy trên chip, memory traffic hay latency inference trên phần cứng neuromorphic.

### 6.5. Có trade-off giữa accuracy và sparsity không?

Có. Mask probability thấp giữ lại nhiều gradient, giúp học mạnh hơn nhưng gần SG thường và có thể chịu gradient mismatch. Mask probability cao tạo gradient thưa hơn nhưng dễ làm mất tín hiệu học. Kết quả tốt nhất thường nằm ở vùng trung gian 0.4 đến 0.6. Ngoài ra, tăng timestep thường cung cấp thêm temporal information nhưng làm tăng chi phí mô phỏng và latency.

### 6.6. Phương pháp có thực sự giảm computation/energy không?

Kết luận thận trọng là **chưa được chứng minh đầy đủ**. Paper cho thấy MSG không làm firing rate tăng đáng kể và training time gần vanilla; tác giả lập luận mask có thể giảm chi phí cập nhật gradient, nhất là trong online learning. Nhưng thí nghiệm vẫn chạy trên GPU, không có phép đo năng lượng hoặc benchmark phần cứng neuromorphic. Mask gradient trên GPU cũng chưa chắc chuyển thành speedup thực tế vì phép nhân mask và sparse operation có thể không được phần cứng khai thác hiệu quả.

## 7. Ưu điểm

1. **Ý tưởng đơn giản và dễ tích hợp:** MSG chỉ thêm mask vào gradient, không yêu cầu thay đổi kiến trúc SNN hay forward dynamics.
2. **Giải quyết đúng vấn đề gradient mismatch:** phương pháp tận dụng khả năng học của SG nhưng giảm số cập nhật không cần thiết.
3. **Có ablation tương đối rõ:** paper tách baseline, TWO, MSG và MSG + TWO, giúp nhận diện đóng góp của từng thành phần.
4. **Đánh giá đa dạng:** thử cả ảnh tĩnh và neuromorphic data, nhiều kiến trúc và nhiều loại surrogate gradient.
5. **TWO không làm tăng số tham số inference:** trọng số temporal được cập nhật từ thống kê lịch sử rồi cố định khi suy luận.
6. **Bảo toàn firing rate:** kết quả Figure 6 cho thấy cải thiện accuracy không đến từ việc làm mạng phát nhiều spike hơn.

## 8. Nhược điểm và hạn chế

1. **Mask là ngẫu nhiên và cần chọn $p$:** hiệu năng nhạy với mask probability; paper chưa đưa ra quy tắc tự động chọn $p$ cho từng layer hoặc từng dataset.
2. **Cơ chế chưa thật sự mô phỏng gradient gốc:** đặt một số phần tử SG bằng 0 tạo sparsity, nhưng không khôi phục chính xác Dirac-delta gradient và không đảm bảo hướng gradient đúng hơn.
3. **Thiếu đo lường phần cứng:** không có energy, synaptic operation, memory access hoặc latency trên Loihi, TrueNorth hay accelerator tương tự.
4. **Chi phí huấn luyện chưa chắc giảm:** trên GPU dense, nhân mask có thể không tạo speedup; bảng training time cho thấy MSG chỉ gần baseline, không phải luôn nhanh hơn.
5. **TWO phụ thuộc thống kê lịch sử:** phân phối accuracy theo timestep có thể thay đổi theo epoch, batch size, augmentation và dữ liệu; trọng số cập nhật chậm có thể phản ứng kém khi phân phối thay đổi.
6. **Phân tích lý thuyết còn giới hạn:** lập luận phương sai gradient giả định phân phối Gaussian và độc lập giữa mask với gradient, trong khi gradient BPTT thực tế có tương quan mạnh theo layer và timestep.
7. **Chưa khảo sát rộng tác động theo layer:** mask cùng xác suất cho các layer có thể không tối ưu, vì layer đầu, layer sâu và output có vai trò khác nhau.
8. **Một số kết quả không vượt mọi baseline:** trên ImageNet, kết quả của paper thấp hơn TET trong bảng so sánh. MSG vì vậy là một regularizer hữu ích nhưng không phải lời giải phổ quát.

## 9. Ý nghĩa và liên hệ với các paper trước

Paper 2 nghiên cứu cách học hình dạng surrogate gradient; Paper 4 giảm gradient error accumulation bằng shortcut và surrogate module. Paper 5 đi theo hướng khác: giữ surrogate gradient tương đối cố định nhưng kiểm soát **mật độ cập nhật** bằng mask. Vì vậy, ba hướng có thể kết hợp: học hình dạng SG, tạo đường gradient bổ sung và làm sparse gradient.

Paper 3 dùng SEW-ResNet để giúp gradient đi qua SNN sâu. MSG có thể được áp dụng trên backbone SEW để nghiên cứu trade-off giữa độ sâu, gradient sparsity và firing rate. Đây là một hướng nối tự nhiên giữa deep SNN và sparse training.

## 10. Hướng nghiên cứu đề xuất

1. **Layer-wise MSG:** học hoặc điều chỉnh $p_l$ riêng cho từng layer dựa trên firing rate, độ lớn gradient và độ sâu.
2. **Mask có cấu trúc:** so sánh random element-wise mask với channel-wise, block-wise hoặc timestep-wise mask để kiểm tra khả năng tăng tốc thật trên phần cứng.
3. **Kết hợp MSG với SML hoặc SEW-ResNet:** đánh giá liệu gradient phụ hoặc residual shortcut có cho phép dùng mask mạnh hơn mà không mất accuracy hay không.
4. **Đánh giá end-to-end trên neuromorphic hardware:** đo energy per inference, synaptic operations, memory access và latency thay vì suy luận từ firing rate.
5. **Tối ưu TWO online:** thay vì cập nhật chỉ theo accuracy lịch sử, thử confidence, margin hoặc entropy nhưng phải kiểm soát chi phí thống kê.

Một research gap có tính khả thi là: **thiết kế layer-wise structured MSG và đo trực tiếp accuracy, firing rate, số synaptic operations và năng lượng trên một simulator hoặc accelerator neuromorphic**. Gap này mở rộng paper gốc theo hướng hardware-aware, đồng thời kiểm tra liệu gradient sparsity có thực sự biến thành hiệu quả tính toán hay chỉ là sparsity trên lý thuyết.

## 11. Kết luận

Paper đề xuất MSG để cân bằng hai yêu cầu trái chiều trong huấn luyện trực tiếp SNN: SG phải đủ dày để truyền tín hiệu học, nhưng gradient cũng cần đủ thưa để tránh cập nhật nhiễu và duy trì tinh thần event-driven. TWO bổ sung một cơ chế decode thích nghi theo thời gian, nhấn mạnh những timestep có khả năng phân loại tốt.

Đóng góp thực tế của paper là một kỹ thuật đơn giản, tương thích với nhiều kiến trúc và cho cải thiện rõ trên một số benchmark, đặc biệt DVS-CIFAR10. Tuy nhiên, bằng chứng về tiết kiệm computation và energy vẫn chưa đầy đủ vì chưa có đánh giá phần cứng trực tiếp. Do đó, giá trị lớn nhất của paper hiện tại là mở ra hướng **sparse gradient training cho SNN**, còn câu hỏi về energy efficiency cần được kiểm chứng bằng các phép đo hardware-aware.

## Tài liệu tham khảo

1. Li, Y., Zhao, F., Zhao, D., & Zeng, Y. (2024). *Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient*. Neural Networks. Preprint: [arXiv:2406.19645](https://arxiv.org/abs/2406.19645).
2. Trang bài báo được dùng trong kế hoạch nghiên cứu: [PubMed 39013289](https://pubmed.ncbi.nlm.nih.gov/39013289/).
