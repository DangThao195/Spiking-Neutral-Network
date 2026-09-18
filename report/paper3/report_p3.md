# BÁO CÁO PAPER 3: DEEP RESIDUAL LEARNING IN SPIKING NEURAL NETWORKS

> Paper: Deep Residual Learning in Spiking Neural Networks  
> Tác giả: Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothée Masquelier, Yonghong Tian  
> Năm: 2021  
> Công bố: NeurIPS 2021  
> Mục tiêu: giải quyết vấn đề huấn luyện SNN sâu bằng cách áp dụng residual learning và đề xuất kiến trúc SEW-ResNet.

---

## 1. Tóm tắt ngắn của paper

Paper này tập trung vào một vấn đề trọng tâm của Spiking Neural Networks (SNN): SNN sâu rất khó huấn luyện trực tiếp bằng backpropagation vì tín hiệu đầu ra là spike nhị phân, rời rạc và có tính động học theo thời gian. Điều này làm cho gradient dễ bị suy yếu hoặc không ổn định khi mạng trở nên sâu. Trong khi đó, ở mạng ANN, ResNet đã chứng minh thành công lớn trong việc cho phép mạng học hiệu quả ở độ sâu lớn nhờ residual connection.

Vì vậy, bài báo đặt câu hỏi: liệu có thể áp dụng nguyên lý residual learning cho SNN để xây dựng deep SNN một cách hiệu quả? Kết quả của paper cho thấy: có thể, nhưng không thể đơn giản “đổi ReLU bằng LIF neuron” trong một ResNet thông thường, vì khi hoạt động với spike, phép tính residual không còn giữ được dạng tương thích tốt với identity mapping và gặp tình trạng degradation. Để giải quyết, paper đề xuất kiến trúc mới gọi là Spike-Element-Wise Residual Network (SEW-ResNet).

SEW-ResNet cho phép residual connection hoạt động trên các tín hiệu spike theo cách “element-wise”, giúp network dễ triển khai identity mapping, giảm vấn đề vanishing/exploding gradient và cho phép huấn luyện trực tiếp SNN sâu hơn 100 lớp. Bài báo chứng minh SEW-ResNet đạt kết quả tốt hơn các SNN trực tiếp huấn luyện trước đó trên các bộ dữ liệu như ImageNet, DVS Gesture và CIFAR10-DVS.

---

## 2. Vấn đề nghiên cứu và lý do cần thiết

### 2.1 SNN có tiềm năng nhưng khó huấn luyện sâu

SNN khác ANN ở chỗ:

- truyền tín hiệu dưới dạng spike nhị phân, không phải số thực liên tục;
- có thêm chiều thời gian (timestep);
- hoạt động theo động lực điện thế màng và ngưỡng phát spike;
- không có đạo hàm chuẩn cho hàm spike dạng bước.

Điều này khiến backpropagation trực tiếp trên SNN gặp khó khăn, đặc biệt ở mạng sâu. Khi số lớp tăng, gradient có xu hướng:

- suy yếu quá mức (vanishing gradient), hoặc
- tăng quá mạnh (exploding gradient).

### 2.2 Vì sao deep SNN không thể chỉ đơn giản dùng ResNet truyền thống?

Một cách tư duy đơn giản: ResNet thành công vì residual branch có thể học phần sai lệch và shortcut cho phép identity mapping. Nhưng trong SNN, nếu chỉ thay ReLU bằng LIF neuron trong block chuẩn của ResNet, thì tín hiệu ở các layer là spike, tức là các giá trị rời rạc 0/1 hoặc gần như vậy. Khi đó: 

- shortcut và residual branch không cùng loại tín hiệu;
- không dễ duy trì dạng identity mapping;
- cấu trúc residual không hoạt động như trong ANN.

Do đó, paper cho rằng cần một biến thể residual phù hợp với tập tín hiệu spike, không chỉ là “dùng lại ResNet bình thường nhưng thay activation”.

---

## 3. Kiến thức nền cần hiểu trước khi đọc paper

### 3.1 Nơ-ron LIF

Nơ-ron trong SNN thường sử dụng mô hình LIF (Leaky Integrate-and-Fire):

- nhận tín hiệu đầu vào;
- tích lũy điện thế màng theo thời gian;
- nếu điện thế vượt ngưỡng thì phát spike;
- sau đó reset điện thế và tiếp tục vòng lặp ở timestep mới.

Đây là động lực học tập trung của SNN: tín hiệu được truyền theo thời gian, không chỉ theo không gian.

### 3.2 Spike là tín hiệu nhị phân

Spike là sự kiện nhị phân: có hoặc không. Tức là:

- không phát spike = 0
- phát spike = 1

Nó tạo ra tính “sparsity” và giúp tiết kiệm tính toán trên phần cứng hướng sự kiện, nhưng đồng thời làm cho gradient truyền ngược rất khó vì hàm spike không khả vi ở hầu hết các điểm.

### 3.3 Residual learning trong ANN

ResNet giới thiệu cách viết:

- output = input + residual

Điều này giúp học phần “sai lệch” thay vì phải học trực tiếp toàn bộ ánh xạ. Khi residual bằng 0, block trở thành identity mapping, giúp cho mạng sâu hơn vẫn học tốt.

Paper muốn mang tính chất này sang SNN, nhưng phải sửa lại vì tín hiệu spike không hoạt động như activation liên tục của ANN.

---

## 4. Nền tảng ý tưởng của bài báo

### 4.1 Mục tiêu chính

Paper muốn xây dựng deep SNN có thể huấn luyện trực tiếp theo kiểu gradient-based, nhưng vẫn giữ được các ưu điểm của SNN như tính sparse và độ tin cậy về temporal processing. 

### 4.2 Vấn đề chính của Spiking ResNet cũ

Các kiến trúc “Spiking ResNet” trước đó thường làm theo cách rất đơn giản: giữ nguyên cấu trúc ResNet, chỉ thay ReLU bằng spiking neuron. Tuy nhiên, cách này gặp ba khó khăn lớn:

1. Spike không giống activation liên tục của ANN.
2. Shortcut và residual path không cùng phạm vi tín hiệu.
3. Không thể dễ dàng thực hiện identity mapping. 

Kết quả là mạng sâu có thể bị degradation: tăng số lớp không dẫn đến độ chính xác tốt hơn, thậm chí tệ hơn.

### 4.3 Ý tưởng giải quyết

Paper đề xuất loại residual có tính chất “spike-element-wise” (SEW): thay vì chỉ đơn giản cộng hoặc thay thế activation như ANN, residual block được thiết kế để hoạt động trực tiếp trên dữ liệu spike và giữ các thành phần ở cùng một dạng tín hiệu. Nghĩa là:

- shortcut và residual path đều làm việc trên spike domain;
- cấu trúc identity mapping dễ thực hiện hơn;
- gradient có thể được truyền ổn định hơn trong mạng sâu.

Đây là điểm cốt lõi của paper.

---

## 5. Phương pháp: SEW-ResNet

### 5.1 Khái niệm cơ bản

SEW-ResNet (Spike-Element-Wise Residual Network) là kiến trúc residual được thiết kế riêng cho SNN. Nó mang những đặc điểm sau:

- dùng spiking neuron trong khối convolution và activation;
- residual branch và shortcut branch được kết nối theo cách element-wise phù hợp với tín hiệu spike;
- hình thành một dạng “cộng / nhân / biến đổi theo phần tử” trên không gian spike;
- cho phép mô hình duy trì sự tương thích giữa residual learning và đặc tính nhị phân của spike.

### 5.2 Tại sao SEW-ResNet hiệu quả hơn?

Vì nó giải quyết được hai điểm yếu của plain Spiking ResNet:

1. Identity mapping dễ hơn: khi residual branch gần 0 hoặc không thay đổi nhiều, shortcut vẫn đưa thông tin ngắn gọn xuống lớp sâu hơn.
2. Gradient dễ truyền: phép toán residual giúp giảm nguy cơ vanishing gradient, đồng thời tạo đường đi ngắn hơn cho gradient trong mạng cực sâu.

### 5.3 Ý nghĩa của “element-wise” trong SNN

Trong ANN, phép cộng đơn giản giữa input và residual là hợp lý vì cả hai đều là số thực. Nhưng trong SNN, input và residual có thể là các chuỗi spike, nên cần một phép toán phù hợp để không làm phá vỡ tính chất spike và temporal dynamics. SEW định nghĩa các phép biến đổi theo phần tử để làm việc với spike mà không làm mất phân phối tín hiệu quan trọng.

### 5.4 Kết quả đạt được

Paper báo cáo rằng SEW-ResNet:

- cho phép huấn luyện trực tiếp deep SNN trên nhiều lớp;
- có khả năng vượt quá 100 lớp;
- đạt độ chính xác cao hơn các SNN trực tiếp huấn luyện trước đó;
- có thể cải thiện hiệu suất bằng cách tăng độ sâu mạng;
- hoạt động tốt trên các bộ dữ liệu như ImageNet, DVS Gesture và CIFAR10-DVS.

---

## 6. Thiết kế thí nghiệm và đánh giá

### 6.1 Bộ dữ liệu thử nghiệm

Paper đánh giá trên các bộ dữ liệu tiêu biểu:

- ImageNet: benchmark ảnh tĩnh lớn, đánh giá độ chính xác và khả năng mở rộng;
- DVS Gesture: dữ liệu event-based, rất phù hợp với tính chất thời gian của SNN;
- CIFAR10-DVS: dữ liệu chuyển đổi từ ảnh thành sự kiện, cho thấy sức mạnh của SNN trong xử lý temporal data.

### 6.2 Chỉ số đánh giá

Các tiêu chí quan trọng gồm:

- accuracy;
- số timestep cần thiết cho mỗi mẫu;
- khả năng huấn luyện mạng sâu;
- hiệu suất so với SNN trực tiếp huấn luyện trước đó.

### 6.3 Kết luận từ thí nghiệm

Kết quả cho thấy:

- deep SNN không cần phải giới hạn ở mạng hẹp để học tốt;
- việc thêm layer có thể cải thiện hiệu suất nếu residual learning được thiết kế đúng;
- SEW-ResNet cho phép hậu quả của phép học sâu đáng kể hơn mạng truyền thống sử dụng spike đơn giản;
- với cùng cách huấn luyện trực tiếp, SEW-ResNet có thể đạt độ chính xác cao hơn bản chất của SNN trực tiếp trước đó.

---

## 7. Ưu điểm của paper

### 7.1 Giải quyết trực tiếp nhược điểm của SNN sâu

Đây là điểm mạnh lớn nhất. Bài báo không chỉ xây dựng SNN tốt hơn, mà còn giải quyết trực tiếp “vì sao SNN sâu lại khó học” bằng cách sử dụng residual learning.

### 7.2 Cho phép deep SNN huấn luyện trực tiếp

Trước đây, việc huấn luyện SNN sâu còn khá hạn chế. SEW-ResNet giúp việc này khả thi, mở ra khả năng mạng SNN mạnh hơn và đa lớp hơn.

### 7.3 Tăng hiệu quả và tính khả thi thực nghiệm

- độ chính xác cao hơn các phương pháp SNN trực tiếp trước đó;
- có thể nhân rộng độ sâu thêm;
- phù hợp với việc mô hình hóa dữ liệu temporal/event-based.

### 7.4 Tính hợp lý về mặt lý thuyết và thực nghiệm

Paper không chỉ nêu cảm giác “dùng residual và thấy tốt”, mà còn xây dựng lập luận rằng residual learning cần phải phù hợp với domain spike. Đó là một đóng góp quan trọng trong lý thuyết SNN.

---

## 8. Nhược điểm và hạn chế

### 8.1 Chi phí tính toán vẫn lớn

Mặc dù SNN có tiềm năng tiết kiệm năng lượng trên phần cứng sự kiện, nhưng trong huấn luyện trên GPU/CPU, SNN vẫn cần nhiều timestep và tính toán theo thời gian. Điều này làm tăng chi phí tính toán so với ANN đơn giản.

### 8.2 Chưa hoàn toàn giải quyết mọi vấn đề về gradient

SEW-ResNet làm giảm hiệu quả của vanishing/exploding gradient, nhưng vẫn không loại bỏ hoàn toàn khó khăn của việc học SNN theo thời gian. Vẫn có những vấn đề như:

- chọn ngưỡng neuron;
- thiết kế decay factor;
- tuning timestep;
- ảnh hưởng của surrogate gradient.

### 8.3 Đòi hỏi kiến thức và thiết kế cẩn thận

Việc thiết kế SEW block, chọn phép element-wise phù hợp, điều chỉnh mô hình theo độ sâu và timestep đều là các quyết định không hề đơn giản. Điều này có thể gây khó khăn khi triển khai trên các bài toán mới.

### 8.4 Khó so sánh trực tiếp với ANN trên phần cứng thực tế

Nhiều báo cáo về SNN nói đến tiết kiệm năng lượng, nhưng hiệu quả thực tế phụ thuộc lớn vào phần cứng triển khai. Vì vậy, đánh giá trên GPU mô phỏng chưa phải là thước đo hoàn toàn chính xác về hiệu quả năng lượng thực tế.

---

## 9. Ý nghĩa khoa học của paper

Paper này có giá trị quan trọng vì nó đi từ một nhận định rất rõ: SNN sâu là cần thiết nhưng hiện tại rất khó huấn luyện. Nó chứng minh rằng nếu muốn xây dựng deep SNN, không thể chỉ “lấy một mạng ResNet ANN rồi thay activation”, mà phải thiết kế lại residual block phù hợp với spike.

Nói cách khác, paper đặt nền tảng cho một hướng nghiên cứu rất quan trọng:

> deep SNN có thể học được nếu residual learning được thiết kế đúng với tính chất của spike và thời gian.

Đây là một bước tiến lớn trong việc biến SNN từ mô hình nghiên cứu lý thuyết trở thành mạng có khả năng học sâu hơn và ứng dụng thực tiễn tốt hơn.

---

## 10. Hướng nghiên cứu tương lai

Dựa trên paper này, các hướng nghiên cứu sau có thể tiếp tục mở rộng:

### 10.1 Kết hợp SEW với kiến trúc mới hơn

- Transformer spike-based;
- ViT/SNN hybrid;
- ConvNext hoặc attention-based SNN;
- mô hình sâu hơn nhưng vẫn giữ tính sparse và temporal.

### 10.2 Cải thiện surrogate gradient và training stability

- surrogate gradient học được;
- adaptive threshold;
- điều chỉnh timestep theo mẫu dữ liệu;
- giảm hiện tượng gradient mismatch.

### 10.3 Nghiên cứu hardware-aware SNN

- kiểm tra hiệu quả trên neuromorphic hardware như Loihi, SpiNNaker, Intel/BrainChip, v.v.;
- đo chính xác latency, energy và throughput thực tế;
- so sánh hiệu quả với ANN, không chỉ trên mô phỏng phần mềm.

### 10.4 Ứng dụng dữ liệu event-based

- camera sự kiện;
- cảm biến thần kinh;
- nhận diện thời gian thực;
- robot và hệ thống tự động.

### 10.5 Tối ưu hóa độ sâu và sparsity cùng lúc

Một hướng nghiên cứu quan trọng là không chỉ tăng số layer, mà còn bảo toàn spike sparsity nhằm tối ưu năng lượng tính toán và độ chính xác.

---

## 11. Kết luận

Paper “Deep Residual Learning in Spiking Neural Networks” là một bài báo quan trọng vì nó giải quyết một vấn đề cốt lõi của SNN: làm sao để huấn luyện được mạng SNN sâu thay vì chỉ giữ ở mạng hẹp hoặc mạng không ổn định. Bằng cách đề xuất SEW-ResNet, paper cho thấy rằng residual learning có thể được áp dụng hiệu quả cho SNN nếu được thiết kế phù hợp với đặc tính spike.

Về mặt ý nghĩa, bài báo mở ra khả năng cho SNN phát triển theo hướng deep learning, không chỉ là mô hình thử nghiệm trong các bài toán nhỏ. Nó cũng đặt nền móng cho nhiều nghiên cứu sau này về deep SNN, surrogate gradient, sparsity và hardware-efficient neuromorphic computing.

Đây là một paper quan trọng trong chuỗi nghiên cứu SNN vì nó gắn liền trực tiếp với vấn đề “deep SNN training” — một bài toán trung tâm của lĩnh vực này.

---

## 12. Tổng kết ngắn theo kiểu báo cáo nhóm

- Paper này nghiên cứu cách huấn luyện SNN sâu bằng residual learning.
- Vấn đề chính là SNN có spike nhị phân, temporal dynamics và gradient khó truyền khi mạng sâu.
- Phương pháp đề xuất là SEW-ResNet, thiết kế residual block phù hợp với tín hiệu spike.
- Đóng góp chính là cho phép training direct deep SNN vượt 100 lớp và đạt hiệu quả tốt hơn các phương pháp cũ.
- Ưu điểm: giảm gradient problem, mở rộng độ sâu, cải thiện accuracy.
- Nhược điểm: tính toán và huấn luyện vẫn nặng, hiệu quả năng lượng thực tế phụ thuộc vào phần cứng.
- Hướng nghiên cứu tiếp theo: kết hợp SNN với kiến trúc mới, tối ưu surrogate gradient, tập trung vào event-based data và neuromorphic hardware.

---

## 13. Câu hỏi thảo luận cho nhóm nghiên cứu

1. Vì sao việc chỉ thay ReLU bằng LIF neuron trong ResNet không đủ cho deep SNN?
2. SEW-ResNet giải quyết vấn đề gì một cách cốt lõi mà plain Spiking ResNet chưa làm được?
3. Tại sao residual learning lại đặc biệt quan trọng đối với SNN sâu?
4. Nếu tăng độ sâu của SNN, liệu có phải lúc nào cũng cải thiện accuracy? Tại sao hoặc tại sao không?
5. Năng lượng tiết kiệm của SNN thực sự đến từ đâu: từ sparsity, từ hardware, hay từ cả hai?

---

## 14. Đọc hiểu theo kiểu ngắn gọn cho người mới

Nếu chỉ cần hiểu đúng 1 câu:

> Paper 3 cho thấy deep SNN không thể chỉ học như ANN bình thường; cần một residual block phù hợp với tín hiệu spike, và SEW-ResNet là cách làm đó.

