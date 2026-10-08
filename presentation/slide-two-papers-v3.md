# Kịch bản thuyết trình: Thông tin thời gian trong SNN

**Đi cùng:** `slide-two-papers-v3.html`  
**Thời lượng gợi ý:** 17–18 phút nếu chỉ hình và giải thích theo gợi ý; nên tập nói và điều chỉnh theo tốc độ thực tế.  
**Mục tiêu:** Giáo viên và các bạn học hiểu vấn đề, cách hoạt động và bằng chứng của hai paper mà không cần biết trước TIC.

## Slide 1 — Mở đầu · 20 giây

Em xin chào thầy cô và các bạn. Nhóm em trình bày một câu hỏi trong mạng nơ-ron xung: làm thế nào để thông tin xuất hiện trước đó vẫn giúp dự đoán ở hiện tại? Nhóm tập trung vào hai paper năm 2026, MD-Mixer và DMP-SNN. Hai paper cùng quan tâm đến thời gian nhưng can thiệp ở hai vị trí khác nhau.

## Slide 2 — Nội dung · 25 giây

Em sẽ đi từ ANN sang SNN và giải thích ngắn cách neuron xung hoạt động. Sau đó, với từng paper, nhóm sẽ nói rõ vấn đề, ý tưởng, pipeline và kết quả. Cuối cùng chúng em so sánh hai phương pháp. Nhóm không dành nhiều thời gian cho các chỉ số chuyên biệt như TIC vì điều quan trọng nhất ở đây là hiểu cơ chế.

## Slide 3 — ANN sang SNN · 45 giây

Trong ANN thông thường, các neuron xử lý giá trị liên tục. Ví dụ một lớp nhận giá trị pixel, tính tổng có trọng số, áp dụng hàm kích hoạt rồi chuyển kết quả sang lớp sau. SNN truyền tín hiệu dưới dạng spike theo từng timestep. Neuron tích lũy tín hiệu đến, chỉ phát spike khi đạt ngưỡng. Vì nhiều timestep không có spike, SNN có tiềm năng giảm tính toán trên phần cứng hỗ trợ xử lý sự kiện. Nhưng không nên khẳng định SNN luôn tiết kiệm năng lượng hơn ANN: số timestep, tỷ lệ spike và kiến trúc phần cứng đều ảnh hưởng.

## Slide 4 — Neuron LIF · 50 giây

Trên hình, chỉ từ trái sang phải: rò điện thế, cộng dòng đầu vào, so ngưỡng và reset. Các số 0,5 và 1 trên sơ đồ chỉ là ví dụ trong báo cáo SNN của project, không phải tham số của hai paper mới.

Neuron LIF là mô hình đơn giản để thấy SNN có trí nhớ. Ở thời điểm t, điện thế màng nhận một phần điện thế từ t−1 và cộng thêm dòng do spike mới tạo ra. Nếu điện thế vượt ngưỡng, neuron phát spike, rồi trạng thái được reset theo quy tắc của mô hình. Ví dụ hai spike đến gần nhau có thể cộng dồn đủ để neuron phát xung, còn một spike riêng lẻ thì chưa đủ. Tuy nhiên phần điện thế cũ bị suy giảm dần, nên tín hiệu rất xa trong quá khứ có thể khó duy trì. Công thức trên slide là bản giản lược, không bao gồm đầy đủ ngưỡng và reset.

## Slide 5 — Bài toán chung · 45 giây

Hãy tưởng tượng ở bước 1 xuất hiện tín hiệu A, còn đến bước 8 mới phải quyết định. Nếu mô hình chỉ giữ một trạng thái đang suy giảm, thông tin về A có thể yếu đi. Có hai câu hỏi khác nhau. Một là ở bước 8, attention có thể lấy trực tiếp đặc trưng của các bước trước không? Hai là neuron có thể giữ một bản tóm tắt lịch sử dài mà không phải lưu nguyên cả chuỗi không? MD-Mixer trả lời câu hỏi đầu, DMP-SNN trả lời câu hỏi sau. Ví dụ A trên slide chỉ để minh họa, không phải dữ liệu thí nghiệm.

## Slide 6 — Vấn đề của MD-Mixer · 50 giây

Chỉ vào panel 3b của paper: bốn khối màu ứng với bốn timestep. Ở T=4, khối K/V màu xanh lá chưa có lát màu từ timestep cũ đi vào trực tiếp.

Trong spiking self-attention thông thường, tại timestep t, Query, Key và Value được tạo từ đặc trưng ở t. Attention kết hợp tốt các token ở thời điểm này. Thông tin cũ vẫn có thể ảnh hưởng gián tiếp nhờ trạng thái neuron, nên không thể nói baseline hoàn toàn không có bộ nhớ. Điểm tác giả nhấn mạnh là attention thiếu một đường truy cập tường minh đến đặc trưng ở t−1, t−3 hoặc một thời điểm cũ khác. Khi tín hiệu hữu ích xuất hiện xa trước hiện tại, đường gián tiếp ấy có thể không đủ.

## Slide 7 — Hướng giải quyết MD-Mixer · 45 giây

Đối chiếu với slide trước: panel 3c xuất hiện các lát màu khác trong khối K/V hiện tại. Dùng ngón tay theo một mũi tên nét đứt để minh họa feature cũ được đưa đến bước mới.

Giải pháp là đặt nhiều nhánh delay trước attention. Mỗi nhánh lấy đặc trưng từ một thời điểm trong buffer. Ví dụ trên hình tô ba mốc quá khứ để hình dung, chứ paper không cố định các delay 1, 3, 5 cho mọi channel. Mô hình học cả vị trí delay và trọng số trộn từng nhánh. Kết quả được dùng để tạo Key và Value, còn Query vẫn hỏi về thời điểm hiện tại. Như vậy attention có ngữ cảnh cũ để đối chiếu với tín hiệu đang đến.

## Slide 8 — Pipeline MD-Mixer · 65 giây

Đọc hình trái trước, hình phải sau. Ở hình trái, mỗi channel có nhiều đường delay và trọng số alpha để cộng các nhánh. Ở hình phải, hai MD-Mixer tạo Key/Value, còn Linear tạo Query.

Ở nhánh trên, X tại timestep t đi qua linear, batch normalization và spiking neuron để tạo Query. Ở nhánh dưới, buffer giữ một số đặc trưng cũ. Với mỗi channel, K nhánh delay lấy K vị trí thời gian đã học. Những đặc trưng này được trộn theo trọng số alpha. Đầu ra của MD-Mixer lại đi qua batch normalization và spiking neuron để tạo Key và Value. Cuối cùng Query, Key và Value đi vào spiking self-attention. Hai thao tác khác nhau diễn ra: MD-Mixer trộn thông tin theo thời gian trong channel; attention kết hợp token sau đó. Triển khai streaming cần buffer cho các timestep cũ, nhưng paper chưa đo đầy đủ chi phí buffer này.

## Slide 9 — Học delay · 55 giây

Ở hình 4, đi từ các đường phân bố rộng ở dưới lên đường nhọn ở trên. Trục ngang là delay, trục dọc là quá trình huấn luyện. Hai panel minh họa phân bố dần sắc và tâm delay có thể di chuyển.

Nếu delay là chỉ số nguyên, ta không thể cập nhật trực tiếp bằng gradient như một trọng số liên tục. Paper đặt một tâm delay có thể học, rồi trong training trải trọng số mềm lên các delay nguyên gần đó. Ban đầu phân bố rộng để mô hình thử nhiều vị trí. Khi giảm nhiệt độ, phân bố hẹp dần và gần một lựa chọn rời rạc. Ví dụ tâm nằm giữa 3 và 4 thì lúc đầu cả hai mốc có thể đóng góp. Đây chỉ là ví dụ giảng giải. Sau training, delay đã học dùng cố định khi inference; phương pháp gốc không tìm delay mới cho từng input.

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
