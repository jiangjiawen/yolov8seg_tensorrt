#pragma once

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <random>

#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "logger.h"
#include "util.h"

using sample::gLogError;
using sample::gLogInfo;

static inline float fast_exp(float x)
{
	union {
		uint32_t i;
		float f;
	} v{};
	v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
	return v.f;
}

inline float sigmoid(float x)
{
	return 1.0f / (1.0f + fast_exp(-x));
}


struct DetectionSeg
{
	int class_id;
	float confidence;
	cv::Rect box;
	std::vector<cv::Point> most_cnt;
};

class SampleYoloV8Seg
{

public:
	SampleYoloV8Seg(const std::string& engineFilename);
	~SampleYoloV8Seg();

	bool InitInfo();

	bool infer(const cv::Mat& img, std::unique_ptr<float>& output_bbox_buffer, std::unique_ptr<float>& output_seg_buffer);

	cv::Mat preprocessedImage(const cv::Mat& img);
	cv::Mat format_yolov5(const cv::Mat& sourceImg);

	std::vector<DetectionSeg> post_process(float* outs_bbox, float* outs_seg, int minDis);

	void draw_result_bboxes(const std::vector<DetectionSeg>& bboxes, cv::Mat img, const std::string& output_filename);

	int32_t get_outputsize_bbox();
	int32_t get_outputsize_seg();

private:

	std::vector<std::string> classes{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

	int W = 640;
	int H = 640;
	int MASK_SIZE = 160;            //!yolov8 default

	int O_W = 1280;
	int O_H = 1280;
	float scale_w = 1.6;
	float scale_h = 1.6;

	const float SCORE_THRESHOLD = 0.5f;
	const float NMS_THRESHOLD = 0.35f;

	std::string mEngineFilename;                    //!< Filename of the serialized engine.

	util::UniquePtr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

	util::UniquePtr<nvinfer1::IExecutionContext> context;
	int32_t input_idx;
	nvinfer1::Dims4 input_dims;
	size_t input_size;

	//output0
	int32_t output_bbox_idx;
	nvinfer1::Dims output_bbox_dims;
	size_t output_bbox_size;

	//output1
	int32_t output_seg_idx;
	nvinfer1::Dims output_seg_dims;
	size_t output_seg_size;

	void* input_mem{ nullptr };
	void* output_bbox_mem{ nullptr };
	void* output_seg_mem{ nullptr };

	cudaStream_t stream;
};

inline SampleYoloV8Seg::SampleYoloV8Seg(const std::string& engineFilename)
	: mEngineFilename(engineFilename)
	, mEngine(nullptr)
{
	// De-serialize engine from file
	std::ifstream engineFile(engineFilename, std::ios::binary);
	if (engineFile.fail())
	{
		return;
	}

	engineFile.seekg(0, std::ifstream::end);
	auto fsize = engineFile.tellg();
	engineFile.seekg(0, std::ifstream::beg);

	std::vector<char> engineData(fsize);
	engineFile.read(engineData.data(), fsize);

	util::UniquePtr<nvinfer1::IRuntime> runtime{ nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()) };
	mEngine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
	assert(mEngine.get() != nullptr);
}

inline SampleYoloV8Seg::~SampleYoloV8Seg()
{
	// Free CUDA resources
	cudaFree(input_mem);
	cudaFree(output_bbox_mem);
	cudaFree(output_seg_mem);
}

inline int32_t SampleYoloV8Seg::get_outputsize_bbox()
{
	return this->output_bbox_size;
}

inline int32_t SampleYoloV8Seg::get_outputsize_seg()
{
	return this->output_seg_size;
}

inline cv::Mat SampleYoloV8Seg::preprocessedImage(const cv::Mat& imageBGR) {
	this->O_W = imageBGR.cols;
	this->O_H = imageBGR.rows;

	cv::Mat input_image = format_yolov5(imageBGR);


	// the same of scale_w and scale_h;
	this->scale_w = input_image.cols * 1.0 / W;
	this->scale_h = input_image.rows * 1.0 / H;

	cv::Mat blob;
	cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(W, H), cv::Scalar(), true, false);

	return blob;
}

inline cv::Mat SampleYoloV8Seg::format_yolov5(const cv::Mat& sourceImg)
{
	int col = sourceImg.cols;
	int row = sourceImg.rows;
	int _max = MAX(col, row);
	cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
	sourceImg.copyTo(result(cv::Rect(0, 0, col, row)));
	return result;
}

inline bool SampleYoloV8Seg::InitInfo()
{

	context = util::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
	if (!context)
	{
		return false;
	}

	input_idx = mEngine->getBindingIndex("images");
	if (input_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(input_idx) == nvinfer1::DataType::kFLOAT);
	input_dims = nvinfer1::Dims4{ 1, 3, H, W };
	context->setBindingDimensions(input_idx, input_dims);
	input_size = util::getMemorySize(input_dims, sizeof(float));
	std::cout << "input size " << input_size << std::endl;

	output_bbox_idx = mEngine->getBindingIndex("output0");
	//output_idx = mEngine->getBindingIndex("prob");
	if (output_bbox_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(output_bbox_idx) == nvinfer1::DataType::kFLOAT);
	output_bbox_dims = context->getBindingDimensions(output_bbox_idx);

	std::cout << "output_bbox_dims: ";
	for (int i = 0; i < output_bbox_dims.nbDims; i++) {
		std::cout << output_bbox_dims.d[i] << " ";
	}
	std::cout << std::endl;

	output_bbox_size = util::getMemorySize(output_bbox_dims, sizeof(float));
	std::cout << "output bbox size " << output_bbox_size << std::endl;
	std::cout << "output bbox idx " << output_bbox_idx << std::endl;


	output_seg_idx = mEngine->getBindingIndex("output1");
	if (output_seg_idx == -1)
	{
		return false;
	}
	assert(mEngine->getBindingDataType(output_seg_idx) == nvinfer1::DataType::kFLOAT);
	output_seg_dims = context->getBindingDimensions(output_seg_idx);

	std::cout << "output_seg_dims: ";
	for (int i = 0; i < output_seg_dims.nbDims; i++) {
		std::cout << output_seg_dims.d[i] << " ";
	}
	std::cout << std::endl;

	output_seg_size = util::getMemorySize(output_seg_dims, sizeof(float));
	std::cout << "output seg size " << output_seg_size << std::endl;
	std::cout << "output seg idx " << output_seg_idx << std::endl;


	// Allocate CUDA memory for input and output bindings
	if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
	{
		gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
		return false;
	}

	if (cudaMalloc(&output_bbox_mem, output_bbox_size) != cudaSuccess)
	{
		gLogError << "ERROR: output cuda memory allocation failed, size = " << output_bbox_size << " bytes" << std::endl;
		return false;
	}

	if (cudaMalloc(&output_seg_mem, output_seg_size) != cudaSuccess)
	{
		gLogError << "ERROR: output cuda memory allocation failed, size = " << output_seg_size << " bytes" << std::endl;
		return false;
	}

	if (cudaStreamCreate(&stream) != cudaSuccess)
	{
		gLogError << "ERROR: cuda stream creation failed." << std::endl;
		return false;
	}

}

inline bool SampleYoloV8Seg::infer(const cv::Mat& img, std::unique_ptr<float>& output_buffer, std::unique_ptr<float>& output_seg_buffer)
{
	cv::Mat input_img_mat = preprocessedImage(img);

	if (cudaMemcpyAsync(input_mem, input_img_mat.data, input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
	{
		gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
		return false;
	}

	// Run TensorRT inference
	//according to index sort
	void* bindings[] = { input_mem, output_seg_mem, output_bbox_mem };
	bool status = context->enqueueV2(bindings, stream, nullptr);
	if (!status)
	{
		gLogError << "ERROR: TensorRT inference failed" << std::endl;
		return false;
	}

	//auto output_buffer = std::unique_ptr<float>{ new float[output_size] };
	if (cudaMemcpyAsync(output_buffer.get(), output_bbox_mem, output_bbox_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
	{
		gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_bbox_size << " bytes" << std::endl;
		return false;
	}


	if (cudaMemcpyAsync(output_seg_buffer.get(), output_seg_mem, output_seg_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
	{
		gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_bbox_size << " bytes" << std::endl;
		return false;
	}

	cudaStreamSynchronize(stream);


	return true;
}

inline std::vector<DetectionSeg> SampleYoloV8Seg::post_process(float* outs_bbox, float* outs_seg, int minDis)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> nums_sort;

	const int nums_of_one = output_bbox_dims.d[1];
	const int nums_of_result = output_bbox_dims.d[2];

	for (int i = 0; i < nums_of_result; ++i) {

		std::vector<float> classes_scores(classes.size(), 0);
		for (int j = 0; j < classes.size(); j++) {
			classes_scores[j] = outs_bbox[(4 + j) * nums_of_result + i];
		}

		cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores.data());
		cv::Point class_id;
		double maxClassScore;

		cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

		if (maxClassScore > SCORE_THRESHOLD) {
			float x = outs_bbox[0 * nums_of_result + i];
			float w = outs_bbox[2 * nums_of_result + i];
			float y = outs_bbox[1 * nums_of_result + i];
			float h = outs_bbox[3 * nums_of_result + i];

			int left = int((x - 0.5 * w) * scale_w);
			int top = int((y - 0.5 * h) * scale_h);

			int width = int(w * scale_w);
			int height = int(h * scale_h);

			boxes.push_back(cv::Rect(left, top, width, height));
			confidences.emplace_back(maxClassScore);

			class_ids.emplace_back(class_id.x);
			nums_sort.emplace_back(i);
		}
	}


	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

	std::vector<DetectionSeg> output;

	for (const auto idx : nms_result) {
		DetectionSeg result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];

		int index_i = nums_sort[idx];
		// do mask calculate; yolact method
		// output shape different from yolov5
		std::vector<uint8_t> mask_u8(MASK_SIZE * MASK_SIZE, 0);
		for (int mi = 0; mi < MASK_SIZE; mi++) {
			for (int mj = 0; mj < MASK_SIZE; mj++) {
				float m_sum = 0.0f;
				for (int k = 0; k < 32; k++) {
					float mask_loc = outs_bbox[(k + classes.size() + 4) * nums_of_result + index_i] * outs_seg[k * (MASK_SIZE * MASK_SIZE) + mi * MASK_SIZE + mj];
					m_sum += mask_loc;
				}
				float m_sigmoid = sigmoid(m_sum);
				// 0.8 
				if (m_sigmoid > 0.8) {
					mask_u8[mi * MASK_SIZE + mj] = static_cast<uint8_t>(1);
				}
				else {
					mask_u8[mi * MASK_SIZE + mj] = static_cast<uint8_t>(0);
				}
			}
		}
		cv::Mat mat(MASK_SIZE, MASK_SIZE, CV_8UC1, mask_u8.data());
		cv::Mat mat_r;
		cv::resize(mat, mat_r,
			cv::Size(W*scale_w, H*scale_h),
			cv::InterpolationFlags::INTER_NEAREST);

		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(mat_r, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		//keep the bigest contour as result
		int max_j = -1;
		float max_ratio = -1.0;
		for (size_t j = 0; j < contours.size(); j++)
		{
			cv::Rect cnt_bbox = cv::boundingRect(contours[j]);
			float area_ratio = (boxes[idx] & cnt_bbox).area()*1.0 / boxes[idx].area();
			if (area_ratio > max_ratio) {
				max_j = j;
				max_ratio = area_ratio;
			}
		}
		result.most_cnt = contours[max_j];

		output.emplace_back(result);
	}

	return output;
}

inline void SampleYoloV8Seg::draw_result_bboxes(const std::vector<DetectionSeg>& bboxes, cv::Mat img, const std::string& output_filename)
{
	cv::Mat mask= cv::Mat::zeros(img.size(), CV_8UC3);

	for (const auto& bbox : bboxes)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(100, 255);
		cv::Scalar color = cv::Scalar(dis(gen),
			dis(gen),
			dis(gen));

		cv::rectangle(img, bbox.box, color, 2);

		// DetectionSeg box text
		std::string classString = classes[bbox.class_id] + ' ' + std::to_string(bbox.confidence).substr(0, 4);
		cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
		cv::Rect textBox(bbox.box.x, bbox.box.y - 40, textSize.width + 10, textSize.height + 20);

		cv::rectangle(img, textBox, color, cv::FILLED);
		cv::putText(img, classString, cv::Point(bbox.box.x + 5, bbox.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

		cv::drawContours(mask, std::vector<std::vector<cv::Point>>{bbox.most_cnt}, -1, color, -1);
	}

	cv::Mat outimg;
	cv::addWeighted(mask, 0.7, img, 1.0, 0.0, outimg);

	cv::imshow(output_filename, outimg);
	cv::waitKey(0);
}


