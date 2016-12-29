#include <caffe/caffe.hpp>
#include <caffe/util/io.hpp>
#include <caffe/blob.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
//#include <boost/shared_ptr.hpp>
//#include <stdio>

using namespace caffe;
using namespace std;
using std::string;
//using cv::Mat;


/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

/*****************************************/
/*		CLASSES
/*****************************************/
class Network {
public:
	Network(const string& model_file, const string& weight_file,
                const string& mean_file,  const string& label_file);

	// Return Top 5 prediction of image 
	std::vector<Prediction > Classify(const cv::Mat& img, int N = 5);

private:
	void SetMean(const string& mean_file);
	void WrapInputLayer(std::vector<cv::Mat>* input_channels);
	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels);
	std::vector<float> Predict(const cv::Mat& img);

	int num_channels;
        shared_ptr<Net<float> > net;

	cv::Mat mean_;
	std::vector<string> labels;
	cv::Size input_geometry;		// size of network - width and height
};

/************************************************************************/	
// Function Network
// Load network, mean file and labels
/************************************************************************/
Network::Network(const string& model_file, const string& weight_file,
                 const string& mean_file, const string& label_file){

        // Load Network and set phase (TRAIN / TEST)
        net.reset(new Net<float>(model_file, TEST));

	// Load pre-trained net 
	net->CopyTrainedLayersFrom(weight_file);

	// Set input layer and check number of channels
	Blob<float>* input_layer = net->input_blobs()[0];
	num_channels = input_layer->channels();
	CHECK(num_channels == 3 || num_channels == 1)
		  << "Input layer should have 1 or 3 channels";

	input_geometry = cv::Size(input_layer->width(), input_layer->height());

	// Load mean file
	SetMean(mean_file);

	// Load labels
	std::ifstream labels2(label_file.c_str());   // vector with labels
	CHECK(labels2) << "Unable to open labels file " << label_file;
	std::string line;
	while (std::getline(labels2, line))
		labels.push_back(string(line));
	

	/*std::ifstream label_file;
	label_file.open(.c_str);
*/


	Blob<float>* output_layer = net->output_blobs()[0];
        /*CHECK_EQ(labels.size(), output_layer->channels())
                << "Number of labels is different from the output layer dimension.";*/
}

/************************************************************************/
// Function SetMean
// Create a mean image
/************************************************************************/
void Network::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	// Convert from BlobProto to Blob<float> 
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);			// make copy
	CHECK_EQ(mean_blob.channels(), num_channels)
		<< "Number of channels of mean file doesn't match input layer";

	// The format of the mean file is planar 32-bit float BGR or grayscale
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels; ++i) {
		// Extract an individual channel
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	// Merge the separate channels into a single image
	cv::Mat mean;
	cv::merge(channels, mean);

	// Compute the global mean pixel value and create a mean image filled with this value 
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry, mean.type(), channel_mean);
}


/************************************************************************/
// Function PairCompare
// Compare 2 pairs
/************************************************************************/
static bool PairCompare(const std::pair<float, int>& lhs,
						const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}


/************************************************************************/
// Function Argmax
// Return the indices of the top N values of vector v where N = 5
/************************************************************************/
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}


/************************************************************************/
// Function Classify 
// Return the top N predictions 
/************************************************************************/
std::vector<Prediction> Network::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);  // output is a float vector

	N = std::min<int>(labels.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels[idx], output[idx]));  
	}

	return predictions;

}

/************************************************************************/
// Function Predict
// wrap input layers and make preprocessing
/************************************************************************/
std::vector<float> Network::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net->input_blobs()[0];

	input_layer->Reshape(1, num_channels, input_geometry.height, input_geometry.width);
	
	// Forward dimension change to all layers
	net->Reshape();		
 
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	// Convert the input image to the input image format of the network
	Preprocess(img, &input_channels);

	net->Forward();

	// Copy the output layer to a std::vector 
	Blob<float>* output_layer = net->output_blobs()[0];
	const float* begin = output_layer->cpu_data();      // output of forward pass
	const float* end = begin + output_layer->channels();

	return std::vector<float>(begin, end);
}

/************************************************************************/
// Function WrapInputLayer
// Wrap the input layer of the network in separate cv::Mat objects (one per channel)
// The last preprocessing operation will write the separate channels directly to the input layer. 
/************************************************************************/
void Network::WrapInputLayer(std::vector<cv::Mat>* input_channels){
	Blob<float>* input_layer = net->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}


/************************************************************************/
// Function Preprocess
// Subtract mean, swap channels, resize input
/************************************************************************/
void Network::Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels){
	
	// Convert the input image to the input image format of the network
	// swap channels from RGB to BGR
	cv::Mat sample;
	if (img.channels() == 3 && num_channels == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	// Resize if geometry of image != input geometry of the network
	cv::Mat sample_resized;
	if (sample.size() != input_geometry)
		cv::resize(sample, sample_resized, input_geometry);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels == 3)		// RGB
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	// Subtract the dataset-mean value in each channel
	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	   input layer of the network because it is wrapped by the cv::Mat
	   objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


/*****************************************/
//					MAIN
/*****************************************/

int main(int argc, char** argv){

        // Init
	::google::InitGoogleLogging(argv[0]);

        const string absolute_path_folder = string(argv[1]);
        const string model_file = absolute_path_folder + string(argv[2]);
        const string weight_file = absolute_path_folder + string(argv[3]);
        const string mean_file = absolute_path_folder + string(argv[4]);
        const string label_file = absolute_path_folder + string(argv[5]);


        // Set mode
        if (strcmp(argv[6], "CPU") == 0){
            Caffe::set_mode(Caffe::CPU);
            //cout << "Using CPU\n" << endl;
        }
        else{
            Caffe::set_mode(Caffe::GPU);
            int device_id = atoi(argv[7]);
            Caffe::SetDevice(device_id);
            //cout << "Using GPU, device_id\n" << device_id << "\n" << endl;
        }

        Network Network(model_file, weight_file, mean_file, label_file);

        string file = "/home/filipa/Documents/Validation_Set/ILSVRC2012_val_00000001.JPEG"; // load image
        cout << "\n---------- Prediction for " << file << " ----------\n" << endl;

	cv::Mat img = cv::imread(file, -1);		 // Read image

	// Predict top 5, return vector predictions with pair (labels, output)
	std::vector<Prediction> predictions = Network.Classify(img); 
										
	// Print the top N predictions
        cout << "Scores \t" << " Predicted Image" << endl;
	for (size_t i = 0; i < predictions.size(); ++i) {
		Prediction p = predictions[i];
		cout << std::fixed << std::setprecision(4) << p.second << " - \""
			 << p.first << "\"" << endl;
	}
}
