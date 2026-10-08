# Kịch bản thuyết trình: Spiking Neural Networks

**Bản đi cùng:** `slide.html` (18 slide).  
**Thời lượng dự kiến:** khoảng 9–10 phút.  
**Cấu trúc:** 1. Nền tảng và bài toán; 2. Các hướng nghiên cứu; 3. Hướng đề xuất.

Các đoạn sau là gợi ý lời nói. Không cần đọc nguyên văn chữ trên slide.

## Slide 1 — Mở đầu · 15 giây

Em xin chào thầy cô và các bạn. Nhóm em là Đặng Thị Ngọc Thảo và Đặng Nguyễn Hữu Huy. Hôm nay nhóm trình bày về Spiking Neural Networks, gọi tắt là SNN, và một hướng thử nghiệm nhằm giảm các phép tính thời gian không cần thiết.

## Slide 2 — Tổng quan · 20 giây

Bài nói gồm ba phần. Đầu tiên, nhóm giải thích SNN hoạt động thế nào và vì sao phải đo chi phí bên cạnh accuracy. Tiếp theo là các paper giải quyết từng điểm nghẽn. Cuối cùng nhóm xác định một câu hỏi hẹp, có thể kiểm chứng bằng thí nghiệm.

## Slide 3 — Vì sao SNN? · 30 giây

Neuron SNN tích lũy điện thế màng và phát spike khi vượt ngưỡng. Spike là tín hiệu rời rạc theo thời gian. Nếu hoạt động đủ thưa, phần cứng có thể bỏ qua nhiều thao tác và truyền dữ liệu theo sự kiện. Hình trên slide minh họa sự khác nhau giữa tính toán dense và dòng spike thưa. Đây là tiềm năng, chưa phải kết quả năng lượng đo được của mọi SNN.

## Slide 4 — Ba điểm nghẽn · 35 giây

Có ba nơi lợi thế đó có thể mất đi. Trong training, hàm phát spike không cho đạo hàm thuận tiện, nên thường phải dùng surrogate gradient. Trong suy luận, mô hình có thể phải chạy nhiều timestep; mỗi bước lại cập nhật trạng thái. Cuối cùng, nếu backend vẫn dùng phép toán dense và truy cập bộ nhớ như cũ, spike thưa không tự biến thành tốc độ hay năng lượng tốt hơn.

## Slide 5 — Đo hiệu quả · 30 giây

Ví dụ A và B ở đây là giả định. B hơn một điểm accuracy, nhưng cần bốn lần số timestep và spike rate cũng cao hơn. Không thể kết luận model nào hiệu quả chỉ từ bảng này. Khi đọc paper, chúng ta cần đi từ accuracy sang mức hoạt động, số phép toán, rồi latency, memory và energy trên hệ thống cụ thể. Các paper khác dataset hoặc protocol không nên xếp hạng trực tiếp.

## Slide 6 — Hai hướng training · 30 giây

Một hướng là train ANN rồi chuyển sang SNN. Hướng còn lại train trực tiếp SNN qua nhiều timestep bằng backpropagation through time. Điểm khó của hướng thứ hai là hàm spike không khả vi. Surrogate gradient giữ spike thật ở forward nhưng thay đạo hàm của nó bằng một hàm xấp xỉ ở backward.

## Slide 7 — Bốn paper về training · 45 giây

Bốn paper ở slide này can thiệp vào bốn vị trí khác nhau. LSG cho độ rộng vùng surrogate gradient thích nghi với dynamics của neuron. SEW-ResNet thiết kế residual connection có đường identity để mạng sâu dễ train hơn. SML nối một ANN phụ từ feature trung gian để gradient về lớp trước bằng đường ngắn hơn; nhánh phụ bị bỏ khi inference. MSG mask một phần gradient surrogate, còn TWO đặt trọng số cho output ở các timestep. Vì khác vị trí, chúng có thể bổ sung nhau, nhưng không nên nói một paper giải quyết toàn bộ bài toán training.

## Slide 8 — MSG và TWO · 35 giây

Ví dụ spike, gradient SG và gradient MSG cho thấy forward thưa chưa có nghĩa backward cũng thưa. MSG dùng mask ngẫu nhiên để một phần tín hiệu cập nhật bằng không. TWO thì tập trung ở bước đọc output, cho các timestep mức đóng góp khác nhau. Trong bảng kết quả của bản PDF nhóm có, CIFAR-10 đạt 95,40 ± 0,12% và DVS-CIFAR10 đạt 83,97 ± 0,12%. Abstract ghi số khác, nên khi trích dẫn nhóm chọn số của bảng và nêu rõ nguồn. Gradient thưa không tự chứng minh GPU chạy nhanh hơn.

## Slide 9 — Kiến trúc Transformer · 35 giây

CNN-SNN xử lý tốt quan hệ cục bộ. Spikformer và các Spiking Transformer đưa attention vào để các vị trí xa nhau tương tác. Spike-driven Transformer điều chỉnh phép attention cho tín hiệu spike. Tuy vậy, các cơ chế attention được MD-Mixer phân tích vẫn chủ yếu tương tác giữa token ở cùng timestep, còn lịch sử đi vào gián tiếp qua trạng thái neuron. MD-Mixer thêm một đường lấy lịch sử rõ ràng vào K và V.

## Slide 10 — MD-Mixer · 40 giây

MD-Mixer có nhiều nhánh delay. Ở mỗi channel, mô hình học nên lấy feature từ thời điểm nào và đặt trọng số cho từng nhánh bao nhiêu. Khi train, delay được học theo cơ chế soft-to-hard; sau train, các giá trị đó cố định cho input mới. Vì vậy nhánh một, hai, K trên hình là ký hiệu, không phải bộ delay 1, 3, 7 được đặt sẵn. Paper báo accuracy và chỉ số tương tác thời gian TIC cải thiện trên nhiều benchmark. Tuy nhiên paper chưa chứng minh đầy đủ latency, buffer cost hay energy trên chip.

## Slide 11 — DMP-SNN · 40 giây

DMP-SNN giải bài toán nhớ dài hơn bằng cách giữ fast spike path và một slow state nhỏ. Spike đầu vào cũng được nén để cập nhật slow state; state này quay lại điều biến điện thế màng. Slow path không phải bộ dự đoán độc lập. Paper còn thiết kế dataflow riêng cho đường spike thưa và memory dense nhưng nhỏ. Các mức hơn bốn lần throughput và hơn năm lần hiệu quả năng lượng thuộc phép so sánh của paper trong mô phỏng post-layout 22FDX, chủ yếu cho inference single-core trên SHD. Đây không phải phép đo silicon cho mọi bài toán.

## Slide 12 — Tổng hợp · 25 giây

Nhìn chung, các paper tạo ba tầng. Tầng học xử lý gradient và mạng sâu. Tầng biểu diễn khai thác quan hệ không gian và thời gian. Tầng hệ thống tổ chức bộ nhớ và phần cứng. Nhóm rút ra một hướng suy luận: nếu không phải mọi input cần cùng lượng tính toán thời gian, có thể thử cho mô hình chọn phần việc cần chạy. Đây là suy luận của nhóm, không phải kết luận sẵn trong các paper.

## Slide 13 — Công trình liên quan · 35 giây

Tính toán thích nghi nói chung đã có trước. DT-SNN có thể dừng sớm theo độ tin cậy sau mỗi timestep. STAS và STEG-AIW giảm token, hoạt động hoặc số timestep theo input. CADAD điều chỉnh delay theo tín hiệu. Vì vậy nhóm thu hẹp câu hỏi vào chọn tập con nhánh delay đã học trong K/V của MD-Mixer theo từng input. Tính mới của hướng này vẫn cần rà soát literature sâu hơn trước khi khẳng định.

## Slide 14 — Điều kiện để tiết kiệm · 30 giây

Hình trái cho thấy vấn đề thường gặp: nếu đã tính cả ba nhánh rồi mới nhân trọng số, phép tính không giảm. Hình phải là giả thuyết của nhóm: router quyết định trước, chỉ nhánh được chọn mới chạy. Khi đo chi phí cần cộng cả overhead của router. Nhóm ưu tiên quyết định ở mức sample hoặc block để bỏ được cả một cụm phép tính, thay vì mask lẻ tẻ từng channel.

## Slide 15 — Kiến trúc đề xuất · 35 giây

Đầu vào đi qua phần tạo spike feature. Một router nhẹ đọc feature này và chọn top-k nhánh delay. Những nhánh đó mới tạo K/V đã trộn lịch sử trước attention. Loss vừa tối ưu phân loại, vừa phạt số nhánh bật trung bình. Chúng em sẽ thay đổi hệ số lambda để xem khi giảm phép tính thì accuracy thay đổi ra sao. Hiện đây là giả thuyết, chưa phải kết quả.

## Slide 16 — Thí nghiệm · 45 giây

Đối chứng quan trọng nhất là MD-Mixer gốc với tất cả nhánh. Static top-k chọn cùng một tập nhánh cho mọi input; random gate kiểm tra lợi ích của giảm nhánh đơn thuần; soft gate kiểm tra trường hợp có trọng số nhưng không thật sự bỏ phép tính. Hard router được so ở cùng ngân sách kích hoạt. Nhóm đo accuracy qua nhiều seed, số nhánh được bật, spike rate, SOPs tính cả router, latency và memory trên cùng thiết bị. Timestep không được gọi là giảm nếu mô hình vẫn chạy đủ T bước.

## Slide 17 — Kết luận · 20 giây

Các paper cho thấy SNN có tiềm năng hiệu quả nhưng còn nhiều chi phí ở training, xử lý thời gian và phần cứng. Hướng nhóm đề xuất là một phép kiểm chứng cụ thể: liệu chọn nhánh delay theo từng input có giữ accuracy trong mức chấp nhận được và giảm cả phép toán lẫn latency hay không?

## Slide 18 — Cảm ơn · 10 giây

Nhóm em xin cảm ơn thầy cô và các bạn đã theo dõi. Nhóm rất mong nhận được câu hỏi và góp ý.

---

# Câu hỏi có thể được hỏi

1. **SNN có luôn tiết kiệm năng lượng hơn ANN không?** Không. Điều đó phụ thuộc spike rate, timestep, cách triển khai và phần cứng có bỏ được phép toán, truy cập bộ nhớ không cần thiết hay không.
2. **Surrogate gradient đổi forward hay backward?** Forward vẫn dùng spike theo ngưỡng. Backward thay đạo hàm không thuận tiện của hàm bước bằng xấp xỉ.
3. **MSG có làm inference sparse hơn không?** Không trực tiếp. MSG mask gradient trong training; spike rate và chi phí inference phải đo riêng.
4. **MD-Mixer có chọn delay theo từng input không?** Bản gốc học delay và trọng số trong training, rồi dùng cố định sau training. Đề xuất của nhóm chỉ chọn tập con nhánh theo input.
5. **Router có thể làm chậm mô hình không?** Có. Vì vậy cần tính cả overhead router và benchmark latency thực tế, bên cạnh số phép toán lý thuyết.
6. **Vì sao không chỉ đặt K nhỏ cố định?** Static K nhỏ là baseline bắt buộc. Hard router chỉ có giá trị nếu ở cùng ngân sách nó tốt hơn static top-k hoặc tạo điểm đánh đổi accuracy–chi phí tốt hơn.
7. **Tính mới của đề xuất nằm đâu?** Ở phép chọn tập con nhánh delay đã học trong K/V của MD-Mixer theo từng input. Tính mới tuyệt đối chưa được khẳng định nếu chưa rà literature đầy đủ.

# Paper chính

- Guo, Huang & Ma (2023), *Direct learning-based deep spiking neural networks: a review*.
- Lian et al. (2023), *Learnable Surrogate Gradient for Direct Training Spiking Neural Networks*.
- Fang et al. (2021), *Deep Residual Learning in Spiking Neural Networks*.
- Deng et al. (2023), *Surrogate Module Learning*.
- Li et al. (2024), *Directly Training Temporal Spiking Neural Network with Sparse Surrogate Gradient*.
- Shi et al. (2026), *Temporal Interaction in Spiking Transformers with Multi-Delay Mixer*.
- Sun et al. (2026), *Algorithm–hardware co-design of neuromorphic networks with dual memory pathways*.
