# Báo cáo đọc paper: Temporal Interaction in Spiking Transformers with Multi-Delay Mixer

> Paper: **Temporal Interaction in Spiking Transformers with Multi-Delay Mixer**  
> Tác giả: Kexin Shi, Hanwen Liu, Zeyang Song, Yang Liu, Jieyuan Zhang, Shuai Wang, Jibin Wu, Malu Zhang, Yang Yang  
> Phạm vi: toàn bộ nội dung chính của PDF 11 trang, gồm phương pháp, công thức, hình, bảng thí nghiệm, ablation và đánh giá phản biện.  
> Đối tượng đọc: người mới học Spiking Neural Network (SNN).

---

## 0. Kết luận ngắn trước khi đi vào chi tiết

| Câu hỏi | Trả lời ngắn |
|---|---|
| Paper đang sửa vấn đề gì? | Spiking Transformer hiện có chủ yếu attention theo không gian trong từng timestep; ký ức thời gian chỉ được xử lý gián tiếp bởi neuron spiking nên thường quá yếu. |
| Ý tưởng chính là gì? | Cho mỗi kênh đặc trưng nhìn lại nhiều thời điểm trong quá khứ bằng các độ trễ học được, rồi trộn các tín hiệu trễ trước khi đưa chúng vào nhánh Key và Value của attention. |
| Tên mô-đun mới | Multi-Delay Mixer, viết tắt là MD-Mixer. |
| Đóng góp phân tích | Temporal Interaction Coefficient (TIC), một chỉ số entropy dùng để đo đầu ra tại thời điểm hiện tại phụ thuộc rộng đến mức nào vào các thời điểm trước. |
| Cách học độ trễ rời rạc | Trong lúc train, mỗi độ trễ được biểu diễn bằng một phân phối tam giác mềm; nhiệt độ giảm dần làm phân phối trở thành gần one-hot. Đây là chiến lược soft-to-hard. |
| MD-Mixer đặt ở đâu? | Dùng MD-Mixer để tạo K và V; Q vẫn giữ căn chỉnh theo thời điểm hiện tại. Sau đó Q, K, V đi vào một spiking self-attention có sẵn. |
| Kết quả nổi bật | 66.3% trên UCF101-DVS, 62.8% trên HMDB51-DVS, 86.65% trên s-CIFAR10 và 64.33% trên s-CIFAR100 ở các cấu hình tốt nhất được báo cáo. |
| Thông điệp quan trọng | Trong SNN, chỉ có trạng thái màng theo thời gian chưa chắc đã đủ. Việc tạo đường truyền trễ rõ ràng và học được có thể giúp attention khai thác lịch sử tốt hơn. |
| Điểm cần thận trọng | Paper không báo cáo năng lượng/độ trễ thực tế trên phần cứng neuromorphic, không có độ lệch chuẩn, và nhiều chi tiết tái lập nằm ở Supplementary Material không có trong PDF này. |

**Một câu để nhớ:** MD-Mixer biến câu hỏi “token hiện tại liên quan đến token nào?” thành “token hiện tại liên quan đến token nào, dựa trên nhiều lát cắt lịch sử đã được học?”.

---

## 1. Bản đồ học paper

Nên đọc báo cáo theo thứ tự sau:

1. Hiểu spike, timestep và neuron LIF.
2. Hiểu Q, K, V trong self-attention.
3. Hiểu vì sao attention theo từng timestep chưa thật sự mô hình hóa thời gian.
4. Hiểu TIC đo sự phụ thuộc thời gian như thế nào.
5. Hiểu MD-Mixer trộn các tín hiệu trễ ra sao.
6. Hiểu cách soft-to-hard giúp học độ trễ rời rạc.
7. Đọc kết quả thí nghiệm và ablation.
8. Cuối cùng mới đánh giá ưu, nhược điểm và khả năng tái lập.

Sơ đồ logic của paper:

~~~text
Attention SNN hiện có
        |
        v
Chủ yếu xử lý không gian trong từng timestep
        |
        v
Dùng TIC để kiểm tra mức phụ thuộc vào lịch sử
        |
        v
Phát hiện reliance tập trung mạnh ở timestep hiện tại
        |
        v
Đề xuất nhiều đường trễ học được theo kênh
        |
        v
Trộn lịch sử vào K và V, giữ Q ở hiện tại
        |
        v
Spiking self-attention thực hiện space-time mixing
        |
        v
TIC tăng và accuracy tăng trên nhiều benchmark
~~~

---

## 2. Từ điển ký hiệu và thuật ngữ

| Ký hiệu/thuật ngữ | Nghĩa | Cách hiểu trực giác |
|---|---|---|
| SNN | Spiking Neural Network | Mạng truyền thông tin bằng các sự kiện spike rời rạc. |
| ANN | Artificial Neural Network thông thường | Mạng thường dùng số thực liên tục làm activation. |
| Spike | Thường là giá trị nhị phân 0/1 | 1 nghĩa là neuron phát xung tại timestep đó. |
| Timestep | Một bước thời gian rời rạc | Một “khung” trong chuỗi xử lý của SNN. |
| Event-driven | Chỉ tính khi có sự kiện cần xử lý | Nền tảng cho tiềm năng tiết kiệm năng lượng. |
| Neuromorphic data | Dữ liệu cảm biến sự kiện, ví dụ DVS | Ghi thay đổi độ sáng theo thời gian thay vì ảnh đầy đủ đều đặn. |
| LIF | Leaky Integrate-and-Fire | Neuron tích lũy điện thế, bị rò, phát spike khi vượt ngưỡng. |
| $U_t$ | Điện thế màng sau tích phân, trước khi phát spike | “Mức điện” hiện tại của neuron. |
| $H_t$ | Trạng thái màng sau bước phát/reset | Trạng thái được mang sang timestep sau. |
| $I_t$ | Dòng vào tại thời điểm $t$ | Thông tin mới đi vào neuron. |
| $\lambda$ | Hệ số rò | Quyết định neuron nhớ trạng thái cũ nhiều hay ít. |
| $V_{th}$ | Ngưỡng phát spike | $U_t$ vượt ngưỡng thì neuron phát xung. |
| $Q,K,V$ | Query, Key, Value | Q đặt câu hỏi; K mô tả thứ có thể được khớp; V là nội dung được lấy. |
| Token $N$ | Vị trí không gian hoặc patch | Một đơn vị mà attention so sánh. |
| Channel $C$ hoặc dimension $D$ | Chiều đặc trưng | Mỗi kênh có thể học kiểu độ trễ riêng. |
| Delay $d$ | Số timestep nhìn lùi về quá khứ | $d=0$ là hiện tại, $d=3$ là lấy tín hiệu cách đây 3 bước. |
| Delay branch $K$ | Số đường trễ song song | Nhiều “ống dẫn” mang về các lát cắt lịch sử khác nhau. |
| $\alpha_i^{(k)}$ | Trọng số nhánh trễ thứ $k$ ở kênh $i$ | Quyết định nhánh nào quan trọng hơn. |
| TIC | Temporal Interaction Coefficient | Entropy của phân bố mức phụ thuộc theo thời gian. |
| Soft-to-hard | Học dạng mềm rồi dần rời rạc hóa | Giúp gradient descent học một lựa chọn delay vốn là số nguyên. |
| Annealing | Giảm nhiệt độ trong quá trình train | Ban đầu khám phá rộng, cuối cùng chốt một delay rõ ràng. |
| SOTA | State of the art | Kết quả tốt nhất hoặc thuộc nhóm tốt nhất tại thời điểm paper. |

**Lưu ý ký hiệu:** paper dùng cả $C$ và $D$ để chỉ số chiều đặc trưng ở các đoạn khác nhau. Trong báo cáo này, $C$ nhấn mạnh số kênh, còn $D$ thường xuất hiện trong phân tích độ phức tạp.

---

## 3. Bước 1 - SNN và neuron LIF

### 3.1 SNN khác ANN ở đâu?

Trong ANN thông thường, một neuron có thể xuất ra các số thực như 0.13, 0.72 hoặc -1.4. Trong SNN, đầu ra thường là một chuỗi spike:

$$
[0,0,1,0,1,0,\ldots]
$$

Thông tin không chỉ nằm ở “có spike hay không”, mà còn nằm ở **spike xuất hiện lúc nào**. Vì vậy SNN có sẵn một trục thời gian tự nhiên.

Hai lợi ích thường được kỳ vọng:

- Tín hiệu thưa: nhiều vị trí là 0.
- Tính toán theo sự kiện: trên phần cứng phù hợp, không có spike thì có thể không cần thực hiện phép tính tương ứng.

Tuy nhiên, “có trục thời gian” không tự động đồng nghĩa với “mô hình hóa thời gian tốt”. Đây chính là điểm xuất phát của paper.

### 3.2 Ba phương trình LIF

Paper dùng neuron Leaky Integrate-and-Fire:

$$
U_t = \lambda H_{t-1} + I_t
\tag{1}
$$

$$
S_t = \Theta(U_t - V_{th})
\tag{2}
$$

$$
H_t = U_t(1-S_t) + V_{reset}S_t
\tag{3}
$$

Giải thích từng bước:

| Bước | Phương trình | Ý nghĩa |
|---|---|---|
| Tích lũy | $U_t=\lambda H_{t-1}+I_t$ | Lấy một phần trạng thái cũ cộng với dòng vào mới. |
| Phát spike | $S_t=\Theta(U_t-V_{th})$ | Nếu điện thế vượt ngưỡng thì $S_t=1$, nếu không thì bằng 0. |
| Reset | $H_t=U_t(1-S_t)+V_{reset}S_t$ | Không phát thì giữ $U_t$; phát rồi thì đưa trạng thái về mức reset. |

Ví dụ đơn giản với $\lambda=0.8$, $V_{th}=1$, $V_{reset}=0$:

| $t$ | $H_{t-1}$ | $I_t$ | $U_t=0.8H_{t-1}+I_t$ | Spike $S_t$ | $H_t$ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.4 | 0.40 | 0 | 0.40 |
| 2 | 0.40 | 0.5 | 0.82 | 0 | 0.82 |
| 3 | 0.82 | 0.5 | 1.156 | 1 | 0 |
| 4 | 0 | 0.2 | 0.20 | 0 | 0.20 |

LIF đã có ký ức vì $H_{t-1}$ đi vào bước tiếp theo. Nhưng ký ức này là một trạng thái nén duy nhất, suy giảm theo $\lambda$. Nó không cho attention một cơ chế rõ ràng để chọn: “hãy lấy thông tin cách đây 1, 3 và 7 bước với trọng số khác nhau”.

### 3.3 Vì sao paper cho rằng động lực LIF là chưa đủ?

LIF truyền quá khứ qua một chuỗi hồi quy:

$$
H_{t-1}\rightarrow H_t\rightarrow H_{t+1}.
$$

Thông tin xa có thể bị suy giảm hoặc trộn lẫn. MD-Mixer bổ sung các đường tắt:

$$
X_{t-1}\rightarrow t,\quad X_{t-3}\rightarrow t,\quad X_{t-7}\rightarrow t.
$$

Do đó, LIF là **ký ức ẩn mang tính tích lũy**, còn delay branch là **truy cập lịch sử có địa chỉ tương đối rõ ràng**.

---

## 4. Bước 2 - Spiking self-attention cơ bản

### 4.1 Trực giác Q, K, V

- Q: thứ vị trí hiện tại đang tìm.
- K: nhãn mô tả của các vị trí có thể được tìm thấy.
- V: nội dung thực sự sẽ được tổng hợp.

Trong Transformer thị giác, token thường tương ứng với patch/vị trí không gian. Attention giúp một vị trí tương tác với các vị trí khác.

### 4.2 Spiking Self-Attention trong paper

Với đầu vào:

$$
X\in\mathbb{R}^{T\times N\times D},
$$

trong đó $T$ là số timestep, $N$ là số token và $D$ là số chiều đặc trưng.

SSA của Spikformer được tóm tắt như sau:

$$
\begin{aligned}
Q &= SN(BN(W_QX)),\\
K &= SN(BN(W_KX)),\\
V &= SN(BN(W_VX)),\\
Z &= SN(\alpha\cdot QK^\top V).
\end{aligned}
\tag{4}
$$

Trong đó $W_Q,W_K,W_V$ là phép chiếu học được; $BN$ là batch normalization; $SN$ là lớp neuron spiking; $\alpha$ là hệ số scale; $Z$ là đầu ra attention.

### 4.3 Nút thắt thời gian

Trong baseline, attention chủ yếu được tính riêng trong từng timestep:

$$
(Q_t,K_t,V_t)\longrightarrow Z_t.
$$

Nó làm tốt việc trộn các token không gian ở cùng $t$, nhưng không trực tiếp tạo các cặp như:

$$
Q_t\leftrightarrow K_{t-3}.
$$

Lịch sử chỉ đi vào gián tiếp qua trạng thái neuron spiking trong các lớp tạo Q, K, V.

**Điểm tinh tế:** “spatial-only” không có nghĩa hệ thống hoàn toàn không biết quá khứ. LIF vẫn có trạng thái. Ý paper là attention không có cơ chế **tường minh, đa thang và có thể học** để lấy thông tin ở nhiều độ trễ.

---

## 5. Bước 3 - TIC đo tương tác thời gian

### 5.1 Reliance coefficient

Tại thời điểm đầu ra $q$, paper đo mức $Z_q$ nhạy với $X_p$ ở thời điểm $p$:

$$
R_{q\rightarrow p}
=
\left\|
\frac{\partial Z_q}{\partial X_p}
\right\|_1,
\qquad p\le q.
\tag{6}
$$

Đọc công thức:

1. Thay đổi rất nhỏ $X_p$.
2. Xem $Z_q$ thay đổi bao nhiêu.
3. Lấy chuẩn $L_1$ để thu thành một số không âm.
4. Số lớn nghĩa là $Z_q$ phụ thuộc mạnh hơn vào timestep $p$.

Với $q$ cố định:

$$
R_q=[R_{q\rightarrow1},\ldots,R_{q\rightarrow q}].
$$

Sau đó chuẩn hóa $L_1$ để có phân bố $\widetilde{R}_q$ với tổng bằng 1.

### 5.2 Temporal Interaction Coefficient

TIC là entropy:

$$
TIC_q=\mathcal{H}(\widetilde{R}_q).
\tag{5}
$$

| Hình dạng $\widetilde{R}_q$ | Diễn giải | TIC |
|---|---|---|
| Gần như toàn bộ khối lượng ở $p=q$ | Chỉ dựa vào hiện tại | Thấp |
| Tập trung vào 2-3 timestep | Có dùng lịch sử nhưng hẹp | Trung bình |
| Trải trên nhiều timestep | Dùng lịch sử đa dạng | Cao |

Ví dụ tại $q=4$:

- $[0,0,0,1]$ có entropy bằng 0.
- $[0.25,0.25,0.25,0.25]$ có entropy cực đại.

Paper không ghi rõ cơ số log trong phần chính. Các giá trị phù hợp với cơ số 2: khi $q=16$, entropy tối đa là $\log_2(16)=4$, còn paper báo các giá trị tới 3.39.

### 5.3 Paper phát hiện gì?

Figure 1 phân tích SSA, QKTA và SDSA trên CIFAR10-DVS. Với $q=4,8,12,16$, reliance của baseline đều nhọn mạnh ở timestep hiện tại; ảnh hưởng của quá khứ xa gần như biến mất.

| Attention | $TIC_{16}$ trước | $TIC_{16}$ sau MD-Mixer | Mức tăng |
|---|---:|---:|---:|
| QKTA | 1.74 | 2.86 | +1.12 |
| SSA | 2.10 | 3.39 | +1.29 |
| SDSA | 2.35 | 3.15 | +0.80 |

Reliance sau MD-Mixer trải rộng hơn. Đây là bằng chứng cơ chế: module không chỉ làm accuracy tăng mà còn thay đổi cách mạng sử dụng lịch sử theo hướng tác giả mong muốn.

### 5.4 TIC đo được gì và không đo được gì?

TIC đo **độ phân tán của sensitivity theo thời gian**. Nó không trực tiếp chứng minh:

- Mọi timestep quá khứ đều chứa thông tin hữu ích.
- Phụ thuộc đó là quan hệ nhân quả theo nghĩa thống kê.
- TIC càng cao thì accuracy luôn càng cao.
- Mô hình hiểu đúng thứ tự thời gian.

Một mô hình có thể phân tán gradient rộng nhưng vẫn dùng lịch sử nhiễu. Vì vậy TIC nên đi cùng accuracy và ablation.

---

## 6. Bước 4 - Multi-Delay Mixer

### 6.1 Cảm hứng sinh học

Tín hiệu thần kinh truyền qua các sợi trục với độ trễ khác nhau. Các tín hiệu phát ở những thời điểm khác nhau có thể đến soma gần nhau và được tích hợp. Paper lấy ý tưởng này để tạo nhiều delay branch.

Đây là **cảm hứng sinh học**, không phải mô phỏng đầy đủ sinh học. MD-Mixer vẫn là module học máy tối ưu bằng backpropagation.

### 6.2 Công thức trộn theo kênh

Với kênh $i$, tại thời điểm $t$:

$$
\widetilde{X}_{t,i}
=
\sum_{k=1}^{K}
\alpha_i^{(k)}
X_{t-d_i^{(k)},i}.
\tag{7}
$$

Trong đó:

- $K$: số nhánh delay.
- $d_i^{(k)}$: delay của nhánh $k$ tại kênh $i$.
- $\alpha_i^{(k)}$: trọng số nhánh.
- $X_{t-d_i^{(k)},i}$: đặc trưng lịch sử.
- $\widetilde{X}_{t,i}$: đặc trưng đã trộn thời gian.

Hai thứ được học đồng thời:

1. **Lấy lúc nào**: $d_i^{(k)}$.
2. **Tin nhánh đó bao nhiêu**: $\alpha_i^{(k)}$.

### 6.3 Ví dụ số

Giả sử một kênh có ba nhánh:

| Nhánh | Delay | Trọng số |
|---:|---:|---:|
| 1 | 0 | 0.5 |
| 2 | 1 | 0.3 |
| 3 | 3 | 0.2 |

Tại $t=5$:

$$
\widetilde{X}_{5,i}
=0.5X_{5,i}+0.3X_{4,i}+0.2X_{2,i}.
$$

Nếu $X_{5,i}=1$, $X_{4,i}=0$, $X_{2,i}=1$, thì $\widetilde{X}_{5,i}=0.7$.

Như vậy đầu ra ở $t=5$ chứa cả hiện tại, lịch sử gần và lịch sử xa.

### 6.4 Vì sao phải có nhiều nhánh?

Một delay chỉ nhìn một thang thời gian. Nhiều nhánh có thể học:

- Nhánh nhanh: chuyển động vừa xảy ra.
- Nhánh trung bình: một pha hành động ngắn.
- Nhánh chậm: bối cảnh dài hơn.

Nhiều hơn không luôn tốt hơn. Ablation cho thấy quá nhiều nhánh làm accuracy giảm, có thể vì mô hình thời gian dư thừa hoặc khó tối ưu.

### 6.5 Channel-wise có ý nghĩa gì?

Mỗi kênh có thể học delay khác nhau. Một kênh chuyên cho thay đổi nhanh, kênh khác cho mẫu chậm. Điều này giàu biểu diễn hơn một delay chung cho toàn tensor.

Paper nói dùng **synaptic-wise delays** trên ImageNet, trong khi phương pháp chính mô tả channel-wise delays. PDF chính không giải thích đầy đủ khác biệt, cách triển khai hoặc chi phí của biến thể synaptic-wise. Cần Supplementary Material hoặc mã nguồn để tái lập chính xác.

---

## 7. Bước 5 - Multi-Delay Self-Attention Framework

### 7.1 Cách tạo Q, K, V

$$
Q=SN(BN(Linear(X))),
\tag{8}
$$

$$
K=SN(BN(MD\text{-}Mixer(X))),
\tag{9}
$$

$$
V=SN(BN(MD\text{-}Mixer(X))).
\tag{10}
$$

Sau đó:

$$
Z=Atten(Q,K,V).
\tag{11}
$$

~~~text
X hiện tại ------------> Linear -> BN -> SN ----> Q

X hiện tại + lịch sử --> MD-Mixer -> BN -> SN -> K

X hiện tại + lịch sử --> MD-Mixer -> BN -> SN -> V

                         Q, K, V
                            |
                            v
                  Spiking Self-Attention
                            |
                            v
                            Z
~~~

### 7.2 Vì sao chỉ trộn K và V?

- Q đại diện cho câu hỏi ở đúng thời điểm hiện tại $t$.
- K chứa ngữ cảnh để Q đối sánh.
- V chứa nội dung lịch sử được tổng hợp.

Giữ Q không trễ giúp câu hỏi vẫn căn chỉnh với hiện tại, trong khi K/V mang thông tin quá khứ đến để được truy vấn.

Paper chưa đưa ablation trong phần chính cho Q-only, K-only, V-only, K+V hoặc Q+K+V. Vì vậy lựa chọn K/V hợp lý về trực giác, nhưng chưa được chứng minh là vị trí tối ưu duy nhất.

### 7.3 “Space-time mixing” nghĩa là gì?

MD-Mixer trộn theo thời gian trong từng kênh. Attention sau đó trộn các token không gian:

$$
\text{Temporal mixing}\rightarrow\text{Spatial attention}.
$$

Khi K/V tại $t$ đã chứa $t-1,t-3,\ldots$, attention ở $t$ gián tiếp kết hợp cả không gian và lịch sử.

### 7.4 Tính nhân quả

Công thức chỉ dùng $p\le q$ và $X_{t-d}$ với $d\ge0$. Không cần nhìn tương lai, nên về nguyên tắc phù hợp với streaming.

Triển khai streaming cần buffer lưu các timestep gần đây. Dung lượng phụ thuộc delay lớn nhất. Paper không báo cáo chi phí buffer hoặc latency phần cứng thực tế.

---

## 8. Bước 6 - Học delay rời rạc bằng soft-to-hard

### 8.1 Tại sao delay khó học?

Delay là số nguyên:

$$
d\in\{0,1,\ldots,T-1\}.
$$

Phép chọn chỉ số nguyên không khả vi theo cách thông thường, trong khi backpropagation cần gradient. Paper dùng một phân phối mềm trong lúc train.

### 8.2 Phân phối tam giác

$$
\phi(d;d^*)
=
\mathcal{N}
\left(
\sigma
\left(
1-\frac{|d-d^*|}{\tau}
\right)
\right).
\tag{12}
$$

- $d$: delay ứng viên nguyên.
- $d^*$: tâm delay học được.
- $\tau$: nhiệt độ/độ rộng.
- $\sigma(x)=\max(0,x)$: ReLU.
- $\mathcal{N}$: chuẩn hóa trên mọi delay ứng viên.

Nếu $d^*=3.2$ và $\tau$ lớn, các delay lân cận cùng có trọng số. Khi $\tau$ giảm, khối lượng tập trung quanh delay gần $3.2$ nhất.

### 8.3 Đưa delay vào neuron

$$
U_{t,j}=\lambda H_{t-1,j}+I_{t,j},
\tag{13}
$$

$$
I_{t,j}
=
\sum_{k=1}^{K}
\alpha_j^{(k)}
\sum_{d=0}^{T-1}
\phi(d;d_j^{(k),*})
X_{t-d,j}.
\tag{14}
$$

Đọc theo hai vòng tổng:

1. Vòng trong thử mọi delay ứng viên bằng phân phối mềm $\phi$.
2. Vòng ngoài gộp $K$ nhánh bằng trọng số $\alpha$.
3. Kết quả là dòng vào $I_{t,j}$.

### 8.4 Lịch giảm nhiệt

Paper dùng squared cosine schedule:

- Đầu train: $\tau$ lớn, phân phối rộng, khám phá nhiều delay.
- Giữa train: phân phối hẹp dần.
- Cuối train: gần one-hot, tức gần một delay rời rạc.

Đây là cầu nối từ “dễ tối ưu bằng gradient” sang “delay rời rạc, gọn khi inference”.

Phần chính không cho công thức schedule đầy đủ, $\tau$ đầu/cuối, cách khởi tạo $d^*$ hoặc quy tắc làm tròn; các chi tiết nằm ở Supplementary Material.

### 8.5 Chi tiết biên

Khi $t-d<0$, công thức truy cập thời điểm trước khi chuỗi bắt đầu. Phần chính không nói rõ dùng zero padding, cắt delay không hợp lệ hay trạng thái khởi tạo khác. Đây là chi tiết cần thiết để tái lập.

---

## 9. Bước 7 - Độ phức tạp tính toán

| Thành phần | Độ phức tạp paper nêu | Số tham số paper nêu |
|---|---:|---:|
| Linear chuẩn | $O(TD^2)$ | $D^2$ |
| MD-Mixer | $O(KTD)$ | $KD$ |

Với $K\ll D$, hệ số giảm lý thuyết xấp xỉ $D/K$. Ví dụ $K=5$, $D=256$ thì $D/K=51.2$.

### Cách hiểu đúng tuyên bố hiệu quả

- Đây là so sánh cho phần Linear được thay, không phải toàn bộ Transformer nhanh hơn 51 lần.
- Kích thước token $N$ bị lược khỏi cả hai biểu thức; nếu tính đầy đủ, hai phía thường cùng có thêm $N$.
- Equation 14 trong lúc train cộng trên mọi delay ứng viên, có thể tốn hơn dạng rời rạc ở Equation 7.
- Vì delay center và $\alpha$ đều học được, cách đếm $KD$ cần giải thích thêm. Nếu lưu hai mảng độc lập, cách đếm trực tiếp có thể gần $2KD$, dù vẫn nhỏ hơn $D^2$ khi $K\ll D$.
- Không có benchmark latency, memory hoặc năng lượng thực đo trên chip neuromorphic.

Kết luận an toàn: **thiết kế có độ phức tạp tiệm cận hấp dẫn, nhưng lợi ích hệ thống thực tế chưa được đo đầy đủ**.

---

## 10. Bước 8 - Thiết lập thí nghiệm

MD-Mixer được gắn vào:

| Kiến trúc | Vai trò |
|---|---|
| Spikformer | Baseline Spiking Transformer kinh điển với SSA. |
| QKFormer | Baseline attention kiểu QK. |
| Spike-driven Transformer V1 | Baseline chính cho ablation. |
| Spike-driven Transformer V2 | Baseline mạnh cho ImageNet và action recognition. |

Nhóm dữ liệu:

| Nhóm | Dataset | Mục đích |
|---|---|---|
| Ảnh tĩnh | ImageNet | Kiểm tra ích lợi khi đầu vào gốc không phải event stream. |
| Neuromorphic object | CIFAR10-DVS, N-Caltech101 | Nhận dạng vật thể từ camera sự kiện. |
| Neuromorphic action | UCF101-DVS, HMDB51-DVS | Nhận dạng mẫu chuyển động theo thời gian. |
| Chuỗi dài nhân tạo | s-CIFAR10, s-CIFAR100 | Mỗi timestep là một cột ảnh; buộc tích hợp qua 32 bước. |

Optimizer, learning rate và cấu hình chi tiết được dẫn sang Supplementary Material.

---

## 11. Bước 9 - Kết quả ImageNet

ImageNet có khoảng 1.3 triệu ảnh train, 50,000 ảnh validation, 1,000 lớp. Table 1 dùng $T=4$.

| Backbone | Cấu hình | Baseline 224 | MD-Mixer 224 | Tăng | Baseline 288 | MD-Mixer 288 | Tăng |
|---|---|---:|---:|---:|---:|---:|---:|
| SDT-V1 | 8-512 | 74.57 | 76.27 | +1.70 | Không báo | 76.87 | Không tính được |
| SDT-V1 | 8-768 | 76.32 | 78.23 | +1.91 | 77.07 | 78.53 | +1.46 |
| SDT-V2 | 8-512 | 79.49 | 80.02 | +0.53 | 79.98 | 80.73 | +0.75 |

So với STAtten cùng backbone:

| Cấu hình | STAtten 224 | MD-Mixer 224 | Chênh | STAtten 288 | MD-Mixer 288 | Chênh |
|---|---:|---:|---:|---:|---:|---:|
| SDT-V1 8-512 | 76.18 | 76.27 | +0.09 | 76.56 | 76.87 | +0.31 |
| SDT-V1 8-768 | 78.11 | 78.23 | +0.12 | 78.39 | 78.53 | +0.14 |
| SDT-V2 8-512 | 79.85 | 80.02 | +0.17 | 80.67 | 80.73 | +0.06 |

Diễn giải:

- MD-Mixer cải thiện nhất quán so với đúng baseline SDT.
- Nó chỉ nhỉnh hơn STAtten một khoảng nhỏ trên ImageNet.
- QKFormer trong bảng đạt 84.22%, cao hơn các cấu hình MD-Mixer liệt kê. Không nên nói MD-Mixer có accuracy cao nhất tuyệt đối của toàn Table 1.
- Ảnh tĩnh được chạy qua nhiều timestep. Cải thiện cho thấy delay giúp động lực nội bộ SNN, nhưng không chứng minh riêng khả năng hiểu chuyển động tự nhiên.

---

## 12. Bước 10 - Neuromorphic object classification

- CIFAR10-DVS: 10,000 mẫu, 9,000 train và 1,000 test, độ phân giải gốc $128\times128$.
- N-Caltech101: 8,831 mẫu, 101 lớp, độ phân giải gốc $180\times240$.
- Cả hai resize về $64\times64$, dùng 16 timestep.

| Backbone | Dataset | Baseline | STAtten | MD-Mixer | Tăng so baseline | Tăng so STAtten |
|---|---|---:|---:|---:|---:|---:|
| Spikformer | CIFAR10-DVS | 80.90 | 82.40 | 83.37 | +2.47 | +0.97 |
| Spikformer | N-Caltech101 | 80.23 | 83.12 | 84.57 | +4.34 | +1.45 |
| QKFormer | CIFAR10-DVS | 84.10 | 84.48 | 85.20 | +1.10 | +0.72 |
| QKFormer | N-Caltech101 | 79.69 | 81.22 | 82.06 | +2.37 | +0.84 |
| SDT-V1 | CIFAR10-DVS | 80.00 | 81.10 | 82.40 | +2.40 | +1.30 |
| SDT-V1 | N-Caltech101 | 81.80 | 83.15 | 84.59 | +2.79 | +1.44 |

**Sai khác nội bộ:** Table 2 ghi 82.40% cho MD-Mixer + SDT-V1 trên CIFAR10-DVS, Table 5 ghi 82.44%. Mức tăng tương ứng là +2.40 và +2.44.

Kết luận:

- Cải thiện trên cả ba backbone, nên khó quy về một kiến trúc duy nhất.
- Mức tăng thường lớn hơn ImageNet, phù hợp với giả thuyết event data cần temporal modeling rõ ràng.
- TIC tăng đồng thời với accuracy, hỗ trợ cơ chế đề xuất.

---

## 13. Bước 11 - Nhận dạng hành động DVS

| Dataset | Phương pháp tốt trước trong bảng | Accuracy trước | MD-Mixer + SDT-V2 | Tăng |
|---|---|---:|---:|---:|
| UCF101-DVS | TIM | 63.8 | 66.3 | +2.5 |
| HMDB51-DVS | TIM | 58.6 | 62.8 | +4.2 |

Đây là nhóm có ý nghĩa thời gian tự nhiên rõ: hành động là chuỗi chuyển động. Kết quả hỗ trợ giá trị của delay đa thang.

Tuy nhiên, Table 3 so sánh các mô hình có thể khác backbone và recipe. Phần chính không cung cấp baseline SDT-V2 không có MD-Mixer trên hai dataset này, nên attribution riêng cho MD-Mixer chưa hoàn toàn sạch.

---

## 14. Bước 12 - Chuỗi dài s-CIFAR10/100

Mỗi ảnh được biến thành chuỗi 32 timestep; mỗi bước là một cột ảnh. Mô hình phải tích hợp nhiều bước mới thấy đủ ảnh.

| Backbone | Dataset | Baseline | STAtten | MD-Mixer | Tăng so baseline | Tăng so STAtten |
|---|---|---:|---:|---:|---:|---:|
| Spikformer | s-CIFAR10 | 76.94 | 77.96 | 78.74 | +1.80 | +0.78 |
| Spikformer | s-CIFAR100 | 55.65 | 56.60 | 59.41 | +3.76 | +2.81 |
| QKFormer | s-CIFAR10 | 79.48 | 79.54 | 85.09 | +5.61 | +5.55 |
| QKFormer | s-CIFAR100 | 55.99 | 56.23 | 64.33 | +8.34 | +8.10 |
| SDT-V1 | s-CIFAR10 | 83.65 | 83.90 | 86.65 | +3.00 | +2.75 |
| SDT-V1 | s-CIFAR100 | 61.54 | 61.07 | 62.44 | +0.90 | +1.37 |

Điểm đáng chú ý:

- Gain lớn nhất là +8.34 điểm trên QKFormer/s-CIFAR100.
- MD-Mixer cải thiện cả ba backbone.
- STAtten thấp hơn baseline SDT-V1 trên s-CIFAR100, trong khi MD-Mixer vẫn tăng.
- Kết quả mạnh trên 32 bước cho thấy lợi ích khi bài toán cần tích hợp lịch sử dài.

---

## 15. Bước 13 - Ablation study

### 15.1 Delay ngẫu nhiên so với delay học được

| Dataset | Baseline | Random delay | Learned MD-Mixer | Gain random | Gain learned |
|---|---:|---:|---:|---:|---:|
| s-CIFAR10 | 83.65 | 84.39 | 86.65 | +0.74 | +3.00 |
| CIFAR10-DVS | 80.00 | 80.52 | 82.44 | +0.52 | +2.44 |

- Đa dạng delay ngẫu nhiên đã có chút ích lợi.
- Phần lớn lợi ích đến từ delay được tối ưu theo dữ liệu.
- Đây là ablation quan trọng nhất cho đóng góp “learnable delay”.

### 15.2 Số delay branch

| Số nhánh $K$ | s-CIFAR10 | CIFAR10-DVS |
|---:|---:|---:|
| 1 | 84.3 | 81.1 |
| 2 | 85.8 | 81.7 |
| 3 | **86.7** | 82.0 |
| 4 | 85.5 | **82.4** |
| 5 | 84.3 | 82.2 |
| 6 | 84.2 | 80.9 |

- s-CIFAR10 tốt nhất ở 3 nhánh.
- CIFAR10-DVS tốt nhất ở 4 nhánh.
- Quá nhiều nhánh làm giảm kết quả.
- $K$ là hyperparameter theo dataset, chưa có giá trị tối ưu chung.

### 15.3 Số timestep

| Timestep | CIFAR10-DVS | N-Caltech101 |
|---:|---:|---:|
| 10 | 81.7 | 84.1 |
| 16 | 82.4 | 84.6 |
| 20 | 82.5 | 85.1 |
| 32 | 83.5 | 85.8 |

Accuracy tăng theo timestep, nhưng chi phí và latency cũng có thể tăng. Paper không đưa đường cong accuracy so với năng lượng hoặc thời gian chạy.

---

## 16. Tổng hợp đóng góp

| Đóng góp | Nội dung | Đánh giá |
|---|---|---|
| TIC | Entropy của sự phụ thuộc gradient qua timestep | Hữu ích để nhìn cơ chế; đây là sensitivity, không phải bằng chứng nhân quả. |
| MD-Mixer | Nhiều delay branch học được theo kênh | Ý tưởng gọn, trực giác rõ, tương thích nhiều backbone. |
| Soft-to-hard | Phân phối tam giác + annealing | Hợp lý; thiếu chi tiết tái lập trong main paper. |
| Multi-Delay Self-Attention | Temporal mixing ở K/V rồi spatial attention | Trực quan; thiếu ablation vị trí Q/K/V. |
| Thí nghiệm | Static, DVS object, DVS action, sequential CIFAR | Là điểm mạnh; cải thiện khá nhất quán. |
| Hiệu quả | Projection từ $O(TD^2)$ xuống $O(KTD)$ | Hấp dẫn nhưng thiếu latency/energy thực tế và chi phí train. |

---

## 17. Điểm mạnh

1. Nút thắt temporal interaction được mô tả rõ và có TIC minh họa.
2. Module có tính mô-đun, thử trên nhiều Spiking Transformer.
3. Thiết kế nhân quả, phù hợp hướng streaming.
4. Learned delay vượt random delay rõ ràng.
5. Benchmark đa dạng.
6. Mức tăng nhất quán ở hầu hết cấu hình báo cáo.
7. TIC và accuracy cùng cải thiện, liên kết cơ chế với hiệu quả nhiệm vụ.

---

## 18. Hạn chế và câu hỏi phản biện

| Hạn chế/câu hỏi | Vì sao quan trọng |
|---|---|
| Không có mean ± std qua nhiều seed | Không biết gain nhỏ 0.06-0.17 điểm trên ImageNet có ổn định thống kê hay không. |
| Không báo năng lượng thực đo | Paper nhấn mạnh neuromorphic efficiency nhưng mới đưa độ phức tạp lý thuyết. |
| Không báo latency và memory | Delay cần buffer lịch sử, quan trọng cho streaming. |
| Chi phí train soft delay chưa rõ | Equation 14 quét mọi delay ứng viên, có thể đắt hơn inference. |
| Cách đếm $KD$ chưa đủ rõ | Có cả delay center và aggregation weight. |
| Thiếu ablation vị trí Q/K/V | Chưa biết K+V có thật sự tối ưu. |
| Thiếu cách xử lý $t-d<0$ | Ảnh hưởng tái lập ở đầu chuỗi. |
| Thiếu hyperparameter | Optimizer, learning rate, $\tau$ schedule, khởi tạo delay nằm ở supplement. |
| Channel-wise và synaptic-wise chưa nối rõ | ImageNet dùng biến thể không được mô tả đủ trong phần phương pháp. |
| ImageNet là ảnh tĩnh | Gain không tự động chứng minh hiểu thời gian tự nhiên. |
| So sánh action khác backbone/recipe | Khó tách chính xác phần gain chỉ do MD-Mixer. |
| TIC có thể cao nhưng không hữu ích | Entropy cao không đảm bảo lịch sử đúng hoặc có ích. |
| Không có dense prediction | Chưa biết kết quả trên detection, segmentation, tracking. |
| Chưa thử chuỗi rất dài | 32 bước chưa đủ cho audio dài hoặc event stream hàng nghìn bước. |

---

## 19. Paper chứng minh và chưa chứng minh

### Được hỗ trợ khá tốt

- Attention SNN baseline được khảo sát có reliance tập trung vào hiện tại.
- MD-Mixer làm reliance rộng hơn theo TIC.
- Delay học được tốt hơn delay ngẫu nhiên trong hai ablation.
- MD-Mixer cải thiện accuracy trên nhiều backbone/dataset.
- Số branch tối ưu hữu hạn và phụ thuộc dataset.

### Chưa nên kết luận quá mạnh

- MD-Mixer luôn là temporal module tốt nhất cho mọi SNN.
- MD-Mixer chắc chắn tiết kiệm năng lượng hơn trên mọi phần cứng.
- TIC cao luôn dẫn đến accuracy cao.
- Delay của MD-Mixer là mô hình sinh học chính xác.
- Module cắm vào mọi kiến trúc mà không cần tuning.

---

## 20. Pseudocode khái niệm

~~~text
Input:
    X shape [T, N, C]
    K_delay nhánh cho mỗi channel
    delay center d_star[channel, branch]
    branch weight alpha[channel, branch]
    temperature tau

Trong training:
    Với mỗi channel i và branch k:
        phi[d] = relu(1 - abs(d - d_star[i,k]) / tau)
        Chuẩn hóa phi để tổng bằng 1

    Với mỗi timestep t và channel i:
        mixed[t,i] = 0
        Với mỗi branch k:
            history = tổng_d phi[d] * X[t-d,i]
            mixed[t,i] += alpha[i,k] * history

    K = Spike(BN(MD_Mixer_K(X)))
    V = Spike(BN(MD_Mixer_V(X)))
    Q = Spike(BN(Linear_Q(X)))
    Z = SpikingAttention(Q, K, V)

    Giảm tau theo squared cosine schedule

Khi hội tụ/inference:
    Mỗi phi gần one-hot
    Thay tổng mềm bằng phép lấy X[t-delay]
~~~

Điểm triển khai cần bổ sung từ supplement/code:

- Padding đầu chuỗi.
- Delay tối đa.
- Ràng buộc $d^*$ vào $[0,T-1]$.
- $\alpha$ có được normalize hay không.
- Hai MD-Mixer cho K/V có chia sẻ delay hay độc lập.
- Cách chuyển phân phối cuối thành delay nguyên.

---

## 21. Ví dụ trực giác hoàn chỉnh

Giả sử camera sự kiện quan sát quả bóng đi từ trái sang phải:

- $t=1$: bóng ở trái.
- $t=2$: bóng gần giữa.
- $t=3$: bóng ở giữa.
- $t=4$: bóng ở phải.

Attention chỉ dùng $K_4,V_4$ chủ yếu thấy hiện tại. LIF có thể mang dấu vết cũ, nhưng dấu vết đã bị nén.

Với MD-Mixer:

- Delay 0 mang vị trí hiện tại.
- Delay 1 mang vị trí ngay trước.
- Delay 3 mang vị trí ban đầu.

K/V tại $t=4$ chứa “hiện tại ở phải + trước đó ở giữa + ban đầu ở trái”. Attention có thêm bằng chứng về hướng di chuyển, không chỉ vị trí hiện tại.

---

## 22. Cách đọc các figure

| Figure | Nội dung | Điều cần nhìn |
|---|---|---|
| Figure 1 | Reliance của SSA, QKTA, SDSA trước MD-Mixer | Các đường nhọn ở hiện tại; TIC thấp. |
| Figure 2 | Neuron sinh học và MD-Mixer | Nhiều đường delay; mỗi kênh có delay/trọng số riêng. |
| Figure 3 | Multi-Delay Self-Attention | Q qua Linear; K/V qua MD-Mixer. |
| Figure 4 | Soft-to-hard | Khi $\tau$ giảm, phân phối hẹp quanh $d^*$. |
| Figure 5 | Reliance sau MD-Mixer | Đường rộng hơn và TIC tăng. |
| Figure 6a | Accuracy theo số branch | Tăng tới tối ưu rồi giảm. |
| Figure 6b | Accuracy theo timestep | Tăng từ 10 tới 32 timestep. |

---

## 23. Bộ câu hỏi tự kiểm tra

### Mức cơ bản

1. Spike khác activation số thực ở điểm nào?
2. $\lambda$ trong LIF ảnh hưởng ký ức ra sao?
3. Q, K, V có vai trò gì?
4. Delay 0 và delay 3 có nghĩa gì?

### Mức hiểu paper

5. Vì sao LIF có trạng thái nhưng temporal modeling vẫn yếu?
6. TIC cao biểu thị điều gì?
7. Vì sao MD-Mixer dùng nhiều branch?
8. Vì sao soft-to-hard cần annealing?
9. Vì sao Q giữ căn chỉnh với hiện tại?
10. Random delay và learned delay khác kết quả thế nào?

### Mức phản biện

11. TIC cao có luôn tốt không?
12. $O(KTD)$ có phản ánh đủ chi phí training không?
13. Cần thí nghiệm gì để chứng minh tiết kiệm năng lượng?
14. Cần ablation gì để xác nhận K/V là vị trí tốt nhất?
15. ImageNet tĩnh nói được gì về temporal modeling?

### Đáp án ngắn

1. Spike thường rời rạc 0/1 và có ý nghĩa thời điểm.  
2. $\lambda$ lớn giữ trạng thái cũ lâu hơn.  
3. Q truy vấn, K để đối sánh, V là nội dung.  
4. 0 lấy hiện tại; 3 lấy cách đây ba bước.  
5. LIF mang ký ức gián tiếp và bị nén; attention thiếu truy cập lịch sử tường minh.  
6. Sensitivity trải trên nhiều timestep hơn.  
7. Để mô hình hóa nhiều thang thời gian.  
8. Dạng mềm cho gradient, annealing giúp chốt delay.  
9. Để câu truy vấn đại diện hiện tại, K/V mang lịch sử.  
10. Learned delay cho gain lớn hơn rõ rệt.  
11. Không; reliance rộng có thể chứa nhiễu.  
12. Chưa; training mềm còn tổng qua delay ứng viên.  
13. Đo energy, latency, spike rate và memory trên cùng hardware/recipe.  
14. So Q-only, K-only, V-only, K+V và Q+K+V.  
15. Cho thấy ích lợi cho động lực nội bộ SNN, không trực tiếp chứng minh hiểu chuyển động.

---

## 24. Gợi ý thí nghiệm tái lập

### Giai đoạn 1 - Tái lập nhỏ

1. Dùng backbone nhỏ như Spikformer-2-256.
2. Bắt đầu với s-CIFAR10 vì chuỗi 32 bước dễ quan sát.
3. Chạy baseline cùng seed và recipe.
4. Thêm random delay cố định.
5. Thêm learned delay.
6. So accuracy, loss, TIC, số tham số và thời gian train.

### Giai đoạn 2 - Kiểm tra cơ chế

- Vẽ histogram delay theo layer/channel.
- Vẽ TIC từng layer.
- So K-only, V-only, K+V.
- Thử shared delay và delay độc lập giữa K/V.
- Thử 1-6 branch.
- Kiểm tra channel học delay trùng hay phân hóa.

### Giai đoạn 3 - Kiểm tra hiệu quả hệ thống

- FLOPs train/inference.
- Peak memory.
- Throughput.
- Latency online.
- Kích thước temporal buffer.
- Spike rate.
- Năng lượng trên phần cứng neuromorphic nếu có.

### Giai đoạn 4 - Kiểm tra độ bền

- Nhiều seed và mean ± std.
- Event noise.
- Missing timestep.
- Thay đổi tốc độ chuyển động.
- Chuỗi dài hơn độ dài train.
- Delay tối đa khác nhau.

---

## 25. Bảng chấm điểm paper

Thang 1-5, dựa trên bản PDF chính:

| Tiêu chí | Điểm | Lý do |
|---|---:|---|
| Độ rõ vấn đề | 5/5 | Nút thắt được mô tả rõ và có TIC. |
| Tính mới | 4/5 | Delay không mới, nhưng cách tích hợp vào K/V gọn. |
| Trực giác phương pháp | 5/5 | Nhiều delay branch dễ hiểu. |
| Bằng chứng thực nghiệm | 4/5 | Nhiều dataset/backbone; thiếu nhiều seed. |
| Bằng chứng năng lượng | 2/5 | Chỉ chủ yếu phân tích tiệm cận. |
| Tái lập từ main PDF | 3/5 | Thiếu schedule, initialization, xử lý biên. |
| Áp dụng rộng | 4/5 | Nhiều Transformer; chưa thử dense task/audio/chuỗi rất dài. |
| Giá trị học SNN | 5/5 | Làm rõ neuron memory và explicit temporal interaction. |

---

## 26. Kết luận cuối

Paper có lập luận ba tầng:

1. **Đo vấn đề:** TIC cho thấy attention SNN dựa quá nhiều vào hiện tại.
2. **Sửa vấn đề:** MD-Mixer đưa nhiều tín hiệu quá khứ học được vào K và V.
3. **Kiểm chứng:** TIC tăng, accuracy tăng, learned delay tốt hơn random delay.

Đóng góp quan trọng là phân biệt:

- **Neuron có trạng thái theo thời gian.**
- **Mạng chủ động truy cập và kết hợp nhiều thời điểm.**

Hai điều này không giống nhau. MD-Mixer biến điều thứ hai thành phép toán rõ ràng, có thể học và tương đối nhẹ.

> Kết luận cân bằng: MD-Mixer là cách đơn giản và có bằng chứng thực nghiệm tốt để tăng temporal interaction trong Spiking Transformer. Tuy nhiên, tuyên bố về hiệu quả phần cứng và khả năng “drop-in” phổ quát vẫn cần benchmark hệ thống, thống kê nhiều seed và mô tả triển khai đầy đủ hơn.

---

## 27. Tham chiếu nên đọc tiếp

| Chủ đề | Công trình paper trích |
|---|---|
| Spikformer/SSA | Zhou et al., Spikformer [63] |
| Spike-driven Transformer | Yao et al. [53] |
| Spike-driven Transformer V2 | Yao et al. [56] |
| QKFormer | Zhou et al. [62] |
| Temporal Interaction Module | Shen et al., TIM [34] |
| Spatial-temporal attention | Lee et al., STAtten [18] |
| Spatiotemporal self-attention | Wang et al., STSA [48] |
| Temporal receptive field | Zhang et al. [61] |
| Biological axonal delay | Stoelzel et al. [43] |

---

## 28. Checklist ghi nhớ

- [ ] Tôi giải thích được LIF.
- [ ] Tôi phân biệt temporal state và explicit temporal interaction.
- [ ] Tôi hiểu reliance gradient trong Equation 6.
- [ ] Tôi hiểu vì sao entropy được dùng làm TIC.
- [ ] Tôi viết lại được Equation 7.
- [ ] Tôi giải thích được soft-to-hard.
- [ ] Tôi nhớ MD-Mixer ở K/V, không phải Q.
- [ ] Tôi biết learned delay tốt hơn random delay.
- [ ] Tôi biết kết quả mạnh nhất không đồng nghĩa mọi so sánh cùng backbone.
- [ ] Tôi biết paper chưa đo energy/latency thực tế.

Đánh dấu được ít nhất 8/10 mục nghĩa là bạn đã nắm phần lớn nội dung cốt lõi.
