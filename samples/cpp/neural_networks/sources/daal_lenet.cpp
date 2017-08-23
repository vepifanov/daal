/* file: daal_lenet.cpp */
/*******************************************************************************
* Copyright 2017 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "daal_lenet.h"
#include "service.h"
#include "image_dataset.h"
#include <cmath>
#include <iostream>

using namespace std;

void train();
void test();
bool checkResult();

TensorPtr _trainingData;
TensorPtr _trainingGroundTruth;
TensorPtr _testingData;
TensorPtr _testingGroundTruth;
size_t TrainDataCount = 50000;
size_t TestDataCount = 100;

prediction::ModelPtr _predictionModel;
prediction::ResultPtr _predictionResult;

const string datasetFileNames[] =
{
    "./data/train-images.idx3-ubyte",
    "./data/train-labels.idx1-ubyte",
    "./data/t10k-images.idx3-ubyte",
    "./data/t10k-labels.idx1-ubyte"
};

int main(int argc, char *argv[])
{

    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    printf("Data loading started... \n");

    DatasetReader_MNIST<double> reader;
    reader.setTrainBatch(datasetFileNames[0], datasetFileNames[1], TrainDataCount);
    reader.setTestBatch(datasetFileNames[2], datasetFileNames[3], TestDataCount);
    reader.read();

    printf("Data loaded \n");

    _trainingData = reader.getTrainData();
    _trainingGroundTruth = reader.getTrainGroundTruth();
    _testingData = reader.getTestData();
    _testingGroundTruth = reader.getTestGroundTruth();

    printf("LeNet training started... \n");

    train();

    printf("LeNet training completed \n");
    printf("LeNet testing started \n");

    test();

    if (checkResult())
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

/*LeNet training*/
void train()
{
    const size_t _batchSize = 10;
    double learningRate = 0.01;

    SharedPtr<optimization_solver::sgd::Batch<float> > sgdAlgorithm(new optimization_solver::sgd::Batch<float>());
    (*(HomogenNumericTable<double>::cast(sgdAlgorithm->parameter.learningRateSequence)))[0][0] = learningRate;

    training::TopologyPtr topology = configureNet();

    training::Batch<> net;

    net.parameter.batchSize = _batchSize;
    net.parameter.optimizationSolver = sgdAlgorithm;

    net.initialize(_trainingData->getDimensions(), *topology);

    net.input.set(training::data, _trainingData);
    net.input.set(training::groundTruth, _trainingGroundTruth);
    net.compute();
    checkPtr(net.getResult().get());
    checkPtr(net.getResult()->get(training::model).get());
    _predictionModel = net.getResult()->get(training::model)->getPredictionModel<double>();
    checkPtr(_predictionModel.get());
}

/*LeNet testing*/
void test()
{
    prediction::Batch<> net;

    net.input.set(prediction::model, _predictionModel);
    net.input.set(prediction::data, _testingData);

    net.compute();

    _predictionResult = net.getResult();

    printPredictedClasses(_predictionResult, _testingGroundTruth);
}

/*check prediction results*/
bool checkResult()
{
    TensorPtr prediction = _predictionResult->get(prediction::prediction);
    if (!prediction) return false;
    const Collection<size_t> &predictionDimensions = prediction->getDimensions();

    SubtensorDescriptor<double> predictionBlock;
    prediction->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, predictionBlock);
    double *predictionPtr = predictionBlock.getPtr();
    if (!predictionPtr) return false;

    SubtensorDescriptor<int> testGroundTruthBlock;
    _testingGroundTruth->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, testGroundTruthBlock);
    int *testGroundTruthPtr = testGroundTruthBlock.getPtr();
    if (!testGroundTruthPtr) return false;
    size_t maxPIndex = 0;
    size_t trueCount = 0;

    /*validation accuracy finding*/
    for (size_t i = 0; i < predictionDimensions[0]; i++)
    {
        double maxP = 0;
        maxPIndex = 0;
        for (size_t j = 0; j < predictionDimensions[1]; j++)
        {
            double p = predictionPtr[i * predictionDimensions[1] + j];
            if (maxP < p)
            {
                maxP = p;
                maxPIndex = j;
            }
        }
        if ( maxPIndex == testGroundTruthPtr[i] )
        trueCount ++;
    }

    prediction->releaseSubtensor(predictionBlock);
    _testingGroundTruth->releaseSubtensor(testGroundTruthBlock);

    if ( (double)trueCount / (double)TestDataCount > 0.9 )
    {
        return true;
    }
    else
    {
        return false;
    }
}
