/* Copyright 2024 Chirale, TensorFlow Authors. All Rights Reserved.

This sketch is derived from the classic Hello World example of the general 
TensorFlow Lite Micro library. It has been adapted and simplified by Chirale 
to conform to the typical style of Arduino sketches. 
It has been tested on an Arduino Nano 33 BLE.
The sketch implements a Deep Neural Network pre-trained on calculating 
the function sin(x). 
By sending a value between 0 and 2*Pi via the Serial Monitor, 
both the value inferred by the DNN model and the actual value 
calculated using the Arduino math library are displayed.

It shows how to use TensorFlow Lite Library on Arduino.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>
#include "model.h"  // Include the generated header file
#include "mnist_samples.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"


const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 16*1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("MNIST Digit Recognition example.");
  Serial.println("Initializing TensorFlow Lite Micro Interpreter...");

  // Map the model into a usable data structure
  model = tflite::GetModel(g_model);

  // Check model schema version
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version is not compatible!");
    while (true);
  }

  // AllOpsResolver
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true);
  }

  // Obtain pointers to model's input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Initialization done.");
  Serial.println("Ready for input data.");
}

void loop() {
  if (Serial.available()) {
    String inputValue = Serial.readString();
    
    // Quantize input data
    for (int i = 0; i < 28 * 28; i++) {
      input->data.f[i] = image_0[i];
    }

    // Perform inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Get output and determine the predicted digit
    float max_value = output->data.f[0];
    int predicted_digit = 0;
    for (int i = 1; i < 10; i++) {
      if (output->data.f[i] > max_value) {
        max_value = output->data.f[i];
        predicted_digit = i;
      }
    }

    // Print the predicted digit
    Serial.print("Predicted digit: ");
    Serial.println(predicted_digit);
    Serial.print("Actual digit: ");
    Serial.println(labels[0]);
  }
}