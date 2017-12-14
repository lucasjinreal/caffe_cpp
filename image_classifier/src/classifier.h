//
// Created by jintian on 17-12-14.
//

#ifndef IMAGE_CLASSIFIER_CLASSIFIER_H
#define IMAGE_CLASSIFIER_CLASSIFIER_H

#include <iostream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "caffe/caffe.hpp"
#include "caffe/net.hpp"

using namespace std;
using namespace caffe;


typedef pair<string, float> Prediction;

class Classifier {
public:
    Classifier(const string& model_file,
            const string& trained_file,
            const string& mean_file,
    const string& label_file);


    vector<Prediction> Classify(const cv::Mat& img, int N = 5);


private:
    void SetMean(const string& mean_file);

    vector<float> Predict(const cv::Mat& img);
    void WrapInputLayer(vector<cv::Mat>* input_channels);
    void Preprocess(const cv::Mat& img, vector<cv::Mat>* input_channels);


private:
    std::shared_ptr<Net<float> > net_;
    cv::Size input_geometry_;
    int num_channels_;
    cv::Mat mean_;
    vector<string> labels_;

    vector<int> Argmax(const vector<float>& v, int N);

};


#endif //IMAGE_CLASSIFIER_CLASSIFIER_H
