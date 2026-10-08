# Kịch bản thuyết trình: SNN và hai paper 2026

**Slide đi kèm:** `slide-two-papers.html`  
**Thời lượng dự kiến:** khoảng 9–10 phút  
**Trọng tâm:** MD-Mixer và DMP-SNN. Phần SNN chỉ cung cấp kiến thức cần để hiểu hai paper.

## Slide 1 — Mở đầu · 15 giây

Em xin chào thầy cô và các bạn. Nhóm em là Đặng Thị Ngọc Thảo và Đặng Nguyễn Hữu Huy. Hôm nay nhóm trình bày về cách SNN xử lý thông tin theo thời gian, tập trung vào hai paper năm 2026: MD-Mixer và DMP-SNN.

## Slide 2 — Lộ trình · 20 giây

Bài nói bắt đầu bằng một giới thiệu ngắn về SNN. Sau đó nhóm phân tích hai cách sử dụng lịch sử: MD-Mixer truy cập feature tại nhiều độ trễ trong attention, còn DMP-SNN nén lịch sử vào một trạng thái nhớ nhỏ. Cuối cùng nhóm đặt hai phương pháp cạnh nhau để thấy phạm vi và giới hạn của từng paper.

## Slide 3 — SNN hoạt động thế nào · 35 giây

Neuron SNN tích lũy điện thế màng. Khi điện thế vượt ngưỡng, neuron phát spike. Tín hiệu vì vậy là các sự kiện rời rạc qua nhiều timestep. Điện thế màng còn giữ một phần trạng thái cũ, nên SNN có khả năng xử lý thời gian tự nhiên. Nếu spike thưa và phần cứng khai thác được sự thưa đó, mô hình có tiềm năng tiết kiệm phép toán. Điều này vẫn phụ thuộc số timestep, tỷ lệ spike và thiết bị.

## Slide 4 — Bài toán chung · 40 giây

Hai paper xuất phát từ hai biểu hiện của cùng vấn đề. Trong Spiking Transformer, attention thường tính tương tác giữa token ở cùng timestep; thông tin cũ chỉ đi vào gián tiếp qua trạng thái neuron. Với chuỗi dài, trạng thái LIF lại có xu hướng quên dần. Có thể thêm recurrence hoặc delay dài để nhớ, nhưng trọng số và buffer tăng. MD-Mixer giải phần truy cập lịch sử trong attention; DMP-SNN giải phần lưu ngữ cảnh dài một cách gọn hơn.

## Slide 5 — MD-Mixer phát hiện vấn đề gì · 35 giây

MD-Mixer không chỉ giả định temporal interaction yếu. Tác giả đo output ở thời điểm hiện tại nhạy đến đâu với input ở từng thời điểm trước. Họ gọi entropy của phân bố độ nhạy này là TIC. Ở các attention baseline được khảo sát, độ nhạy tập trung mạnh vào thời điểm hiện tại. TIC thấp nghĩa là sự phụ thuộc theo thời gian khá hẹp. Tuy nhiên TIC cao không tự động có nghĩa mô hình dùng đúng thông tin; vẫn phải xem accuracy và ablation.

## Slide 6 — MD-Mixer hoạt động thế nào · 45 giây

Ở mỗi channel, MD-Mixer có nhiều nhánh lấy feature từ các thời điểm trước với những độ trễ khác nhau. Mô hình học độ trễ và trọng số trộn của từng nhánh. Feature đã trộn được đưa vào Key và Value; Query vẫn giữ thông tin của thời điểm hiện tại. Attention nhờ vậy đối chiếu hiện tại với ngữ cảnh thời gian phong phú hơn. Các ký hiệu d một, d hai, d K trên hình là các delay được học, không phải các giá trị 1, 3, 7 cố định cho mọi trường hợp.

## Slide 7 — MD-Mixer học delay · 35 giây

Delay là chỉ số thời gian nguyên, khó tối ưu trực tiếp bằng gradient. Paper dùng một phân bố mềm trên các delay ứng viên trong lúc train, rồi giảm nhiệt độ để phân bố tập trung vào một delay rời rạc. Điều quan trọng là các delay được học theo dữ liệu nhưng sau train dùng cố định cho các input mới. Paper không chọn lại tập delay riêng cho từng mẫu.

## Slide 8 — Bằng chứng của MD-Mixer · 45 giây

Trên CIFAR10-DVS, TIC của SSA tại timestep 16 tăng từ 2,10 lên 3,39 sau MD-Mixer. Điều này cho thấy output nhạy với nhiều thời điểm quá khứ hơn. Trong cùng backbone SDT-V2 trên ImageNet, top-1 tăng từ 79,49 lên 80,02% ở T bằng 4. Trên s-CIFAR100 với QKFormer, accuracy tăng từ 55,99 lên 64,33%. Hai bài toán khác nhau nên không so trực tiếp các mức tăng đó. Paper còn cho thấy learned delay tốt hơn random delay trong ablation. Giới hạn là chưa có phép đo đầy đủ về buffer, latency và năng lượng thực tế.

## Slide 9 — DMP-SNN đặt vấn đề gì · 35 giây

Khi chuỗi dài, ảnh hưởng của tín hiệu cũ trong điện thế màng LIF suy giảm theo thời gian. Recurrent SNN có thể thêm bộ nhớ nhưng thường cần ma trận kết nối lớn. Delay-based SNN giữ sự kiện đến muộn, nhưng delay dài cần buffer sâu. DMP-SNN hỏi liệu có thể giữ một tóm tắt lịch sử nhỏ, dùng chung trong layer, trong khi vẫn giữ đường spike nhanh hay không.

## Slide 10 — Cơ chế DMP-SNN · 50 giây

Cùng vector spike đầu vào đi theo hai đường. Đường nhanh tạo dòng điện trực tiếp cho neuron. Đường chậm nén spike thành một giá trị, cập nhật vector trạng thái m qua thời gian, rồi chiếu state đó trở lại làm dòng điện bổ sung. Memory vì thế điều biến phản ứng của neuron theo ngữ cảnh, không tự làm classifier tách biệt. Vector m có số chiều nhỏ hơn số neuron, giúp tóm tắt lịch sử mà không phải lưu toàn bộ spike quá khứ.

## Slide 11 — Kết quả thuật toán DMP-SNN · 35 giây

Trên S-MNIST, paper báo FSNN đạt 59%, delay-based SNN đạt 88,79%, DMP-SNN Solution II đạt 99,20%. Các model trong bảng có số tham số khác nhau, nên đây là bằng chứng DMP mạnh trên task này chứ không phải so sánh cùng ngân sách hoàn hảo. Một ablation rất quan trọng là khi bỏ fast spike path, DMP rơi về 10%, bằng mức đoán ngẫu nhiên. Điều đó cho thấy slow memory cần đường phản ứng tức thời.

## Slide 12 — Phần cứng DMP-SNN · 45 giây

Điểm đặc biệt của DMP-SNN là thiết kế phần cứng cùng thuật toán. Đường spike thưa và đường memory nhỏ nhưng dense được bố trí dataflow khác nhau. Paper còn gộp các phép cập nhật neuron để giảm đọc ghi trạng thái trung gian. Trong setup của họ, throughput cao hơn hơn bốn lần và energy efficiency hơn năm lần so với các baseline được chọn. Các số này đến từ mô phỏng post-layout 22FDX, tập trung inference single-core trên SHD; một số baseline lấy từ công bố khác rồi chuẩn hóa. Chưa phải phép đo chip DMP đã chế tạo.

## Slide 13 — So sánh hai paper · 55 giây

MD-Mixer giữ lịch sử dưới dạng feature ở các delay cụ thể và đưa vào K/V của attention. DMP-SNN nén lịch sử vào vector slow state rồi tác động lên điện thế màng. MD-Mixer có thí nghiệm trên nhiều Spiking Transformer và làm rõ temporal interaction, nhưng còn thiếu đo chi phí hệ thống. DMP có bằng chứng mạnh hơn về đồng thiết kế phần cứng, nhưng tập trung các task chuỗi và phần cứng single-core. Chúng ta không thể đặt accuracy của hai paper cạnh nhau để kết luận ai tốt hơn, vì dataset, backbone và protocol khác nhau.

## Slide 14 — Kết luận · 35 giây

Hai paper cho thấy trạng thái thời gian sẵn có của SNN chưa đủ để giải mọi bài toán. MD-Mixer cho mạng truy cập nhiều thời điểm quá khứ rõ ràng hơn. DMP-SNN giữ ngữ cảnh dài qua một state nhỏ và thiết kế dataflow phù hợp. Câu hỏi nghiên cứu tiếp theo là chọn cơ chế nào cho loại dữ liệu nào, với mức chi phí thời gian, bộ nhớ và năng lượng chấp nhận được. Đây là câu hỏi mở rút ra từ hai paper, chưa phải kết quả nhóm đã kiểm chứng.

## Slide 15 — Cảm ơn · 10 giây

Nhóm em xin cảm ơn thầy cô và các bạn. Nhóm mong nhận được câu hỏi và góp ý.

---

# Câu hỏi có thể được hỏi

1. **TIC đo gì?** Nó là entropy của phân bố độ nhạy của output hiện tại đối với input ở các timestep trước. Nó phản ánh mức trải rộng của phụ thuộc theo thời gian, không trực tiếp đo tính hữu ích hay quan hệ nhân quả.
2. **MD-Mixer khác LIF memory ở đâu?** LIF giữ một trạng thái tích lũy và suy giảm. MD-Mixer lấy feature từ các độ trễ cụ thể đã học, tạo đường truy cập lịch sử tường minh.
3. **Delay của MD-Mixer có thay đổi theo từng input không?** Không trong thiết kế gốc. Delay được học trong training và dùng cố định ở inference.
4. **Slow memory của DMP có đủ để dự đoán không?** Không trong ablation của paper. Khi bỏ fast spike path, accuracy trên S-MNIST rơi về mức đoán ngẫu nhiên.
5. **Có thể nói DMP tiết kiệm năng lượng hơn mọi SNN không?** Không. Kết quả hơn năm lần thuộc setup phần cứng và baseline được paper chọn, dùng post-layout simulation chứ chưa đo chip DMP thật.
6. **Paper nào tốt hơn?** Chưa thể xếp hạng trực tiếp. Chúng giải hai vị trí khác nhau và dùng task, backbone, protocol khác nhau.

# Nguồn chính

- Shi et al. (2026), *Temporal Interaction in Spiking Transformers with Multi-Delay Mixer*, CVPR 2026. PDF trong project: `Temporal Interaction in Spiking Transformers with Multi-Delay Mixer.pdf`.
- Sun et al. (2026), *Algorithm–hardware co-design of neuromorphic networks with dual memory pathways*, Nature Machine Intelligence 8, 901–912. DOI: 10.1038/s42256-026-01255-3. PDF trong project: `s42256-026-01255-3.pdf`.
- Guo, Huang & Ma (2023), *Direct learning-based deep spiking neural networks: a review*, cho phần nền tảng SNN.
