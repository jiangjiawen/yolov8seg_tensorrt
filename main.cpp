#include "yolov8_seg.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
	using namespace std::literals;

	SampleYoloV8Seg sample("yourpath\\yolov8s-seg.trt");
	sample.InitInfo();

	cv::Mat img = cv::imread("yourpath\\bus.jpg");

	int output_bbox_size = sample.get_outputsize_bbox();
	int output_seg_size = sample.get_outputsize_seg();

	auto output_bbox_buffer = std::unique_ptr<float>{ new float[output_bbox_size / sizeof(float)] };
	auto output_seg_buffer = std::unique_ptr<float>{ new float[output_seg_size / sizeof(float)] };

	sample.infer(img.clone(), output_bbox_buffer, output_seg_buffer);

	auto bbox_data = output_bbox_buffer.get();
	auto seg_data = output_seg_buffer.get();
	std::vector<DetectionSeg> res_bbox = sample.post_process(bbox_data, seg_data, 10);

	sample.draw_result_bboxes(res_bbox, img.clone(), "res.jpg");

	return 0;
}