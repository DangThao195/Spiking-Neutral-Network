# Kịch bản thuyết trình: Thông tin thời gian trong SNN

**Đi cùng:** `slide-two-papers-v8.html`  
**Thời lượng gợi ý:** 17–18 phút nếu chỉ hình và giải thích theo gợi ý; nên tập nói và điều chỉnh theo tốc độ thực tế.  
**Mục tiêu:** Giáo viên và các bạn học hiểu vấn đề, cách hoạt động và bằng chứng của hai paper mà không cần biết trước TIC.

## Slide 1 — Mở đầu · 20 giây

Em xin chào thầy cô và các bạn. Nhóm em trình bày một câu hỏi trong mạng nơ-ron xung: làm thế nào để thông tin xuất hiện trước đó vẫn giúp dự đoán ở hiện tại? Nhóm tập trung vào hai paper năm 2026, MD-Mixer và DMP-SNN. Hai paper cùng quan tâm đến thời gian nhưng can thiệp ở hai vị trí khác nhau.

## Slide 2 — Nội dung · 25 giây

Em sẽ đi từ ANN sang SNN và giải thích ngắn cách neuron xung hoạt động. Sau đó, với từng paper, nhóm sẽ nói rõ vấn đề, ý tưởng, pipeline và kết quả. Cuối cùng chúng em so sánh hai phương pháp. Nhóm không dành nhiều thời gian cho các chỉ số chuyên biệt như TIC vì điều quan trọng nhất ở đây là hiểu cơ chế.

## Slide 3 — Pipeline ANN và SNN · 55 giây

Đọc hàng trên trước. ANN nhận giá trị pixel liên tục, qua các lớp trọng số và hàm kích hoạt, tạo activation liên tục rồi dùng nó để dự đoán. Tiếp theo đọc hàng dưới. Với cùng ví dụ ảnh đầu vào, SNN cần mã hóa ảnh thành chuỗi spike theo từng timestep. Neuron LIF tích lũy điện thế, có thể không phát spike ở một bước và phát ở bước khác. Hình 0–0–1–0 chỉ là ví dụ, không phải đầu ra cố định của SNN. Sau T bước, mô hình tổng hợp thông tin để dự đoán. Điểm cần nhớ là SNN lặp qua thời gian, còn trạng thái của neuron nối các bước lại với nhau. Nếu spike thưa và phần cứng khai thác được sự thưa đó, tính toán có thể tiết kiệm hơn; không nên nói SNN luôn tiết kiệm năng lượng hơn ANN.

## Slide 4 — Pipeline neuron LIF · 70 giây

Hãy đọc pipeline từ trái sang phải. Ở timestep t, neuron nhận spike từ lớp trước. Trọng số synapse biến spike này thành dòng đầu vào I_t. Neuron cộng dòng mới với phần điện thế màng còn lại từ bước t−1, rồi so điện thế mới với ngưỡng. Nếu đạt ngưỡng, neuron phát spike; nếu chưa đạt, đầu ra ở bước đó bằng 0. Sau phép so ngưỡng, có hai thông tin đi theo hai hướng. Spike 0 hoặc 1 được chuyển sang lớp tiếp theo. Điện thế màng sau khi xét ngưỡng và reset được giữ lại làm trạng thái cho timestep t+1. Quy trình lặp T lần, nhưng điều đó không có nghĩa neuron phát spike T lần. Mối liên hệ giữa các timestep chính là điện thế màng được chuyển tiếp này. Công thức trên slide mô tả phiên bản LIF giản lược để giải thích trực quan; paper có thể dùng biến thể cập nhật và reset khác.

## Slide 5 — Bài toán chung · 45 giây

Hãy tưởng tượng ở bước 1 xuất hiện tín hiệu A, còn đến bước 8 mới phải quyết định. Nếu mô hình chỉ giữ một trạng thái đang suy giảm, thông tin về A có thể yếu đi. Có hai câu hỏi khác nhau. Một là ở bước 8, attention có thể lấy trực tiếp đặc trưng của các bước trước không? Hai là neuron có thể giữ một bản tóm tắt lịch sử dài mà không phải lưu nguyên cả chuỗi không? MD-Mixer trả lời câu hỏi đầu, DMP-SNN trả lời câu hỏi sau. Ví dụ A trên slide chỉ để minh họa, không phải dữ liệu thí nghiệm.

## Slide 6 — Vấn đề của MD-Mixer · 50 giây

Hãy theo hình từ trái sang phải. Ngôi sao đỏ ở t1 tượng trưng cho một đặc trưng quan trọng đã xuất hiện sớm; các biểu tượng còn lại là feature ở những bước sau. Tại t4, feature màu xanh lá tạo Q, K và V rồi đi vào attention. Không có đường nối trực tiếp từ ngôi sao đỏ ở t1 hay tam giác ở t3 đến K/V tại t4. Hình chỉ diễn giải đường đi trực tiếp của feature, không có nghĩa baseline hoàn toàn quên quá khứ: trạng thái neuron vẫn có thể mang ảnh hưởng gián tiếp. Vấn đề paper muốn giải quyết là cho attention ở hiện tại truy cập tường minh tới feature từ một timestep cũ được chọn. Đây là hình minh họa tạo bằng AI, không phải hình trong paper.

## Slide 7 — Hướng giải quyết MD-Mixer · 45 giây

Giữ nguyên bốn mốc thời gian để so sánh trực tiếp với slide 6. Bây giờ hai đường màu đỏ và xanh đưa feature cũ từ t1 và t3 tới vị trí trộn trước K/V. Trong K và V xuất hiện cả biểu tượng cũ lẫn feature hiện tại; Q vẫn lấy feature từ t4. Đây là ý tưởng cốt lõi của MD-Mixer: nhiều nhánh delay lấy feature từ bộ đệm thời gian, mô hình học vị trí delay và trọng số trộn theo channel, rồi dùng kết quả để tạo K/V. Mốc t1 và t3 chỉ là ví dụ trong hình diễn giải, không phải độ trễ cố định của paper. Sang slide 8, ta sẽ xem sơ đồ cơ chế gốc của tác giả để thấy rõ mixer và self-attention.

## Slide 8 — Pipeline MD-Mixer · 65 giây

Đọc hình trái trước, hình phải sau. Ở hình trái, mỗi channel có nhiều đường delay và trọng số alpha để cộng các nhánh. Ở hình phải, hai MD-Mixer tạo Key/Value, còn Linear tạo Query.

Ở nhánh trên, X tại timestep t đi qua linear, batch normalization và spiking neuron để tạo Query. Ở nhánh dưới, buffer giữ một số đặc trưng cũ. Với mỗi channel, K nhánh delay lấy K vị trí thời gian đã học. Những đặc trưng này được trộn theo trọng số alpha. Đầu ra của MD-Mixer lại đi qua batch normalization và spiking neuron để tạo Key và Value. Cuối cùng Query, Key và Value đi vào spiking self-attention. Hai thao tác khác nhau diễn ra: MD-Mixer trộn thông tin theo thời gian trong channel; attention kết hợp token sau đó. Triển khai streaming cần buffer cho các timestep cũ, nhưng paper chưa đo đầy đủ chi phí buffer này.

## Slide 9 — Quy trình học delay · 65 giây

Hãy nhìn hàng trên của hình. Bốn biểu tượng là feature đã lưu trong buffer ở các thời điểm t−3, t−2, t−1 và t. Trong lúc huấn luyện, một nhánh delay dùng phân bố mềm để lấy và cộng feature từ nhiều thời điểm với trọng số khác nhau. Các luồng màu đi vào dấu tổng Σ, tạo feature đã trộn rồi đưa sang nhánh tạo K/V. Paper biểu diễn trọng số của mỗi delay ứng viên bằng một phân bố tam giác quanh tâm delay có thể học d*. Vì cách chọn còn mềm, gradient có thể cập nhật tâm này.

Mũi tên ở giữa biểu thị nhiệt độ τ giảm dần qua các epoch. Nhìn xuống hàng dưới: phân bố trở nên gần một lựa chọn rời rạc. Trong ví dụ minh họa, chỉ feature ở t−2 còn đường đi sang K/V. Mốc t−2 và bốn biểu tượng trên hình không phải delay cụ thể hay dữ liệu thực nghiệm của paper. Đây là hình diễn giải tạo bằng AI cho một nhánh; MD-Mixer thật có nhiều nhánh và nhiều channel, với delay và trọng số trộn được học. Khi suy luận, delay đã học được giữ cố định thay vì chọn lại theo từng input. Figure 4 gốc của paper minh họa phân bố hẹp dần; bản slide trước vẫn giữ hình đó nếu cần đối chiếu.

## Slide 10 — Kết quả MD-Mixer · 60 giây

Chỉ vào ba hàng s-CIFAR10 của Table 5: baseline, random delay và MD-Mixer. Các con số ở bảng bên phải là thí nghiệm khác, phải đọc theo từng hàng riêng.

Đầu tiên xem ablation trên s-CIFAR10 với SDT-V1. Baseline đạt 83,65%. Gắn delay ngẫu nhiên đạt 84,39%, còn MD-Mixer với delay học được đạt 86,65%. Điều này cho thấy không chỉ việc thêm delay, mà cách chọn delay theo dữ liệu cũng quan trọng. Ở ImageNet cùng backbone SDT-V2 8-512, top-1 tăng từ 79,49 lên 80,02%. Với s-CIFAR100 và QKFormer chạy 32 bước, tăng từ 55,99 lên 64,33%. Chỉ so từng hàng với baseline tương ứng; không so trực tiếp các bài toán khác nhau. Paper chưa công bố đo buffer, latency và năng lượng thực tế đủ để kết luận lợi ích triển khai.

## Slide 11 — Vấn đề của DMP-SNN · 50 giây

Đọc Hình 1a từ trên xuống: FSNN chỉ có trạng thái neuron, RSNN thêm vòng hồi tiếp, DSNN thêm đường delay, DMP-SNN thêm khối memory nhỏ phía trên soma. Không cần giải từng ký hiệu trong ảnh; sơ đồ vị trí bộ nhớ là ý chính.

Chuyển sang DMP-SNN. S-MNIST biến ảnh 28 nhân 28 thành chuỗi 784 pixel, mỗi bước đọc một pixel. Dự đoán cuối cùng phải tích hợp thông tin qua cả chuỗi. Điện thế LIF có thể quên dần tín hiệu đầu chuỗi. Kết nối hồi tiếp giúp giữ thông tin nhưng tăng số kết nối. Delay dài có thể đưa spike cũ đến muộn, song cần buffer để giữ các spike đang chờ. DMP-SNN tìm một cách lưu ngữ cảnh dài gọn hơn mà vẫn giữ đường xử lý spike hiện tại.

## Slide 12 — Hướng giải quyết DMP-SNN · 55 giây

Chỉ nửa trái của Hình 1b trước: mũi tên Fast path chạy thẳng đến soma, còn Slow path đi qua shared slow state rồi điều biến soma. Nửa phải là cách phần cứng bố trí riêng hai đường dữ liệu này.

DMP-SNN chia xử lý thành hai đường song song. Fast path biến spike hiện tại thành dòng điện trực tiếp cho neuron, giúp phản ứng tức thời. Slow memory path nhận cùng spike, nén nó rồi cập nhật vector trạng thái m. Vector này giữ một tóm tắt của lịch sử và được chiếu trở lại thành dòng điện bổ sung. Neuron cộng dòng nhanh, dòng nhớ và điện thế cũ trước khi quyết định phát spike. Memory ở đây là ngữ cảnh điều biến, không phải một classifier riêng có thể thay thế fast path.

## Slide 13 — Pipeline DMP-SNN · 70 giây

Ta theo dõi một timestep. Trước tiên layer nhận vector spike từ layer trước. Từ cùng đầu vào, nhánh nhanh tính trọng số feedforward để tạo dòng điện nhanh. Nhánh chậm nén cả vector spike thành một scalar, dùng scalar này cập nhật vector nhớ m theo một phương trình state-space, rồi chiếu m trở lại không gian neuron để tạo dòng nhớ. Hai dòng điện gặp nhau khi cập nhật điện thế màng. Neuron vượt ngưỡng sẽ phát spike sang layer tiếp theo. Sơ đồ có hai nhánh song song; không phải fast path chạy xong mới đến slow path. State m có số chiều d nhỏ hơn số neuron N, nên memory là bản tóm tắt gọn chứ không phải toàn bộ lịch sử spike.

## Slide 14 — Ví dụ hai đường phối hợp · 45 giây

Ví dụ trực giác: ở t=1 có tín hiệu A, ở t=8 có tín hiệu B. Fast path tại t=8 phản ứng trực tiếp với B. Slow state đã đi qua nhiều bước, nên có thể còn biểu diễn một phần ngữ cảnh liên quan đến A. Khi cộng cả hai dòng, neuron có thể xử lý B khác đi tùy lịch sử. Không nên hiểu state nhỏ này giữ chính xác mọi chi tiết của A. Việc nén đổi độ chi tiết để lấy chi phí bộ nhớ thấp hơn. Đây là ví dụ minh họa cơ chế, không phải một thí nghiệm của paper.

## Slide 15 — Kết quả thuật toán DMP-SNN · 60 giây

Trong Table 1, chỉ đúng hai nhóm PS-MNIST và S-MNIST. Đọc hàng DSNN rồi DMP-SNN (II) và nhắc người nghe nhìn thêm cột Parameters để tránh hiểu là so sánh cùng ngân sách.

Trên S-MNIST 784 bước, FSNN đạt 59%, DSNN đạt 88,79%, DMP-SNN Solution II đạt 99,20%. Trên PS-MNIST, thứ tự pixel bị hoán vị làm phụ thuộc dài khó hơn: FSNN 11,30%, DSNN 72,06%, DMP-SNN II 96,65%. Các kết quả cho thấy slow memory giúp rất mạnh trong hai task này. Tuy nhiên DMP dùng số tham số cao hơn các baseline trong bảng, nên không thể nói nó thắng tuyệt đối ở cùng ngân sách. Ta cần xem cả accuracy lẫn số tham số.

## Slide 16 — Ablation fast path · 45 giây

Ở Extended Data Table 2, tìm hai cột “DMP (Wf=0) Accuracy” và “DMP Accuracy”, rồi lần theo hai hàng S-MNIST và PS-MNIST. Ký hiệu Wf=0 nghĩa là bỏ đường feedforward nhanh.

Một thí nghiệm giúp hiểu vai trò hai đường: trên S-MNIST, DMP-SNN II đầy đủ đạt 99,20%. Khi bỏ fast spike path, accuracy còn 10%, ngang mức đoán ngẫu nhiên của bài toán mười lớp. PS-MNIST cũng giảm xuống 10% khi bỏ fast path. Điều này không phủ nhận giá trị memory; nó cho thấy state chậm cung cấp ngữ cảnh cho đường phản ứng trực tiếp chứ không thể tự thay thế toàn bộ xử lý đầu vào.

## Slide 17 — Phần cứng DMP-SNN · 65 giây

Hình trên mã màu ba đường: đỏ cho spike, xanh dương/tím cho memory, xanh lá cho cập nhật state. Hình dưới lần lượt là tốc độ, năng lượng mỗi timestep và diện tích; chỉ nêu kết luận trong phạm vi setup của paper, không đọc mọi cột.

Paper còn đồng thiết kế phần cứng. Spike path thưa, nên phần cứng tập trung đọc trọng số của các spike đang hoạt động. Memory path dense nhưng nhỏ, nên phù hợp với cách giữ kết quả trung gian gần neuron. Các phép leak, tích hợp spike và tích hợp memory được gộp để giảm truy cập SRAM trung gian. Trong setup đánh giá của paper, throughput cao hơn trên bốn lần so với Loihi2 delay implementation và energy efficiency cao hơn trên năm lần so với các baseline Loihi2 và DenRAM sau chuẩn hóa. Cần nói rõ đây là post-layout simulation ở 22FDX, tập trung inference single-core trên SHD. Một số số liệu đối chứng lấy từ nghiên cứu khác; chưa phải kết quả đo chip DMP chế tạo.

## Slide 18 — So sánh hai paper · 55 giây

MD-Mixer giữ lịch sử dưới dạng feature ở một số delay và đưa vào Key, Value của spiking attention. DMP-SNN nén lịch sử thành vector trạng thái nhỏ rồi thêm ngữ cảnh vào điện thế neuron. MD-Mixer có bằng chứng accuracy trên nhiều Spiking Transformer, nhưng chưa đo đầy đủ chi phí hệ thống. DMP mạnh ở task chuỗi dài và có thiết kế phần cứng cụ thể, song kết quả phần cứng hiện là mô phỏng với phạm vi đánh giá nhất định. Hai paper khác bài toán và protocol, nên không thể lấy accuracy của chúng đặt cạnh nhau để tuyên bố phương pháp nào tốt hơn.

## Slide 19 — Kết luận · 40 giây

Điểm nhóm muốn nhấn mạnh là SNN có trục thời gian và điện thế màng, nhưng chỉ có khả năng nhớ tự nhiên chưa đủ. Mô hình cần cách tổ chức lịch sử phù hợp với vị trí sử dụng. MD-Mixer cho attention truy cập có địa chỉ đến các thời điểm cũ. DMP-SNN nén bối cảnh dài và đưa nó trở lại neuron qua memory nhỏ. Các kết quả đã thử đều đáng chú ý, còn lựa chọn tốt nhất cho một ứng dụng cụ thể cần đo đồng thời accuracy, latency, bộ nhớ và năng lượng.

## Slide 20 — Cảm ơn · 10 giây

Nhóm em xin cảm ơn thầy cô và các bạn. Nhóm mong nhận được câu hỏi và góp ý.

---

## Câu hỏi có thể được hỏi

1. **ANN khác SNN cốt lõi ở đâu?** ANN thường truyền activation liên tục. SNN truyền spike theo timestep, còn neuron tích lũy trạng thái và phát xung khi vượt ngưỡng. Lợi ích năng lượng phụ thuộc triển khai.
2. **Baseline MD-Mixer có hoàn toàn không nhớ quá khứ không?** Không. LIF vẫn có trạng thái theo thời gian. MD-Mixer bổ sung đường lấy feature quá khứ tường minh vào K/V.
3. **Delay MD-Mixer có thay đổi theo từng input không?** Không ở phương pháp gốc. Delay được học khi training rồi dùng cố định khi inference.
4. **Vì sao DMP vừa có fast vừa có slow path?** Fast path xử lý spike hiện tại. Slow state cung cấp ngữ cảnh; ablation bỏ fast path rơi về mức đoán ngẫu nhiên trên S-MNIST và PS-MNIST.
5. **Có thể nói DMP tiết kiệm năng lượng hơn mọi SNN không?** Không. Hệ số hơn năm lần thuộc setup so sánh và cách chuẩn hóa của paper, từ post-layout simulation chứ chưa phải phép đo chip DMP thật.
6. **Paper nào tốt hơn?** Không thể xếp hạng trực tiếp vì điểm can thiệp, dataset, backbone và protocol khác nhau.

## Nguồn chính

- Shi et al. (2026), *Temporal Interaction in Spiking Transformers with Multi-Delay Mixer*, CVPR 2026. File gốc: `Temporal Interaction in Spiking Transformers with Multi-Delay Mixer.pdf`.
- Sun et al. (2026), *Algorithm–hardware co-design of neuromorphic networks with dual memory pathways*, *Nature Machine Intelligence* 8, 901–912, DOI 10.1038/s42256-026-01255-3. File gốc: `s42256-026-01255-3.pdf`.
- Guo, Huang & Ma (2023), *Direct learning-based deep spiking neural networks: a review*, cho phần nền tảng SNN.
