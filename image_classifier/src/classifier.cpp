//
// Created by jintian on 17-12-14.
//

#include <fstream>
#include "classifier.h"


/**
 * the construct method of Classifier
 * @param model_file
 * @param trained_file
 * @param mean_file
 * @param label_file
 */
Classifier::Classifier(const string &model_file, const string &trained_file, const string &mean_file,
                       const string &label_file) {
    // we assume you are using cpu
    Caffe::set_mode(Caffe::CPU);

    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "We have only one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "We have only one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();

    CHECK(num_channels_ == 3 || num_channels_ == 1) << "Input layer should have 1 or 3 channels.";

    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

    SetMean(mean_file);

    ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open file " << label_file;
    string line;
    while (getline(labels, line)){
        labels_.push_back(string(line));
    }

    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels()) << "Number of labels is different from the output layer dim.";
}

vector<Prediction> Classifier::Classify(const cv::Mat &img, int N) {
    vector<float> output = Predict(img);
    N = min<int>((int) labels_.size(), N);
    vector<int> maxN = Classifier::Argmax(output, N);
    vector<Prediction> predictions;
    for (int i=0; i<N; ++i) {
        int idx = maxN[i];
        predictions.push_back(make_pair(labels_[idx], output[idx]));
    }
    return predictions;
}

/**
 * This method not define in class, will it be called??????
 * @param lhs
 * @param rhs
 * @return
 */
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

vector<int> Classifier::Argmax(const vector<float> &v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

void Classifier::SetMean(const string &mean_file) {
    BlobProto blobProto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blobProto);

    Blob<float> meanBlob;
    meanBlob.FromProto(blobProto);
    CHECK_EQ(meanBlob.channels(), num_channels_) << "num of channels in mean file must same with input layer.";

    vector<cv::Mat> channels;
    float* data = meanBlob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
        cv::Mat channel(meanBlob.height(), meanBlob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanBlob.height() * meanBlob.width();
    }
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    cv::Scalar channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

vector<float> Classifier::Predict(const cv::Mat &img) {
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(1, num_channels_, input_geometry_.height, input_geometry_.width);

    net_->Reshape();

    vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);
    Preprocess(img, &input_channels);
    net_->Forward();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels->push_back(channel);
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    cv::split(sample_normalized, *input_channels);

    CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
          == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}
