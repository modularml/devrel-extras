/*******************************************************************************
 * Copyright (c) 2024, Modular Inc. All rights reserved.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions:
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/pytorch/config.h"
#include "max/c/tensor.h"
#include "max/c/value.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void logHelper(const char *level, const char *message, const char delimiter) {
  printf("%s: %s%c", level, message, delimiter);
}
void logDebug(const char *message) { logHelper("DEBUG", message, ' '); }
void logInfo(const char *message) { logHelper("INFO", message, '\n'); }
void logError(const char *message) { logHelper("ERROR", message, '\n'); }

#define CHECK(x)                                                               \
  if (M_isError(x)) {                                                          \
    logError(M_getError(x));                                                   \
    return EXIT_FAILURE;                                                       \
  }

// Read file at the given path. On failure abort.
char *readFileOrExit(const char *filepath) {
  FILE *file;
  file = fopen(filepath, "rb");
  if (!file) {
    printf("failed to open %s. Aborting.\n", filepath);
    abort();
  }
  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  rewind(file);

  char *buffer = (char *)malloc(fileSize * sizeof(char));
  fread(buffer, fileSize, 1, file);
  fclose(file);
  return buffer;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: clip-vit <path to bert saved model>");
    return EXIT_FAILURE;
  }

  M_Status *status = M_newStatus();

  M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
  M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
  CHECK(status);

  logInfo("Compiling Model");
  M_CompileConfig *compileConfig = M_newCompileConfig();
  const char *modelPath = argv[1];
  M_setModelPath(compileConfig, /*path=*/modelPath);

  logInfo("Setting InputSpecs for compilation");
  int64_t *inputIdsShape =
      (int64_t *)readFileOrExit("inputs/input_ids_shape.bin");
  M_TorchInputSpec *inputIdsInputSpec =
      M_newTorchInputSpec(inputIdsShape, /*rankSize=*/2, /*dtype=*/M_INT64);

  int64_t *pixelValuesShape =
      (int64_t *)readFileOrExit("inputs/pixel_values_shape.bin");
  M_TorchInputSpec *pixelValuesInputSpec =
      M_newTorchInputSpec(pixelValuesShape, /*rankSize=*/4, /*dtype=*/M_FLOAT32);

  int64_t *attentionMaskShape =
      (int64_t *)readFileOrExit("inputs/attention_mask_shape.bin");
  M_TorchInputSpec *attentionMaskInputSpec = M_newTorchInputSpec(
      attentionMaskShape, /*rankSize=*/2, /*dtype=*/M_INT64);


  M_TorchInputSpec *inputSpecs[3] = {inputIdsInputSpec, pixelValuesInputSpec, attentionMaskInputSpec};
  M_setTorchInputSpecs(compileConfig, inputSpecs, 3);

  M_AsyncCompiledModel *compiledModel =
      M_compileModel(context, &compileConfig, status);
  CHECK(status);

  logInfo("Initializing Model");
  M_AsyncModel *model = M_initModel(context, compiledModel, status);
  CHECK(status);

  logInfo("Waiting for model compilation to finish");
  M_waitForModel(model, status);
  CHECK(status);

  logInfo("Inspecting model metadata");
  size_t numInputs = M_getNumModelInputs(compiledModel, status);
  CHECK(status);
  printf("Num inputs: %ld\n", numInputs);

  M_TensorNameArray *tensorNames = M_getInputNames(compiledModel, status);
  CHECK(status);
  logDebug("Model input names:");
  for (size_t i = 0; i < numInputs; i++) {
    const char *tensorName = M_getTensorNameAt(tensorNames, i);
    printf("%s ", tensorName);
  }
  printf("\n");

  logInfo("Preparing inputs...");
  M_AsyncTensorMap *inputToModel = M_newAsyncTensorMap(context);

  M_TensorSpec *inputIdsSpec =
      M_newTensorSpec(inputIdsShape, /*rankSize=*/2, /*dtype=*/M_INT64,
                      /*tensorName=*/"input_ids");

  int64_t *inputIdsTensor = (int64_t *)readFileOrExit("inputs/input_ids.bin");
  M_borrowTensorInto(inputToModel, inputIdsTensor, inputIdsSpec, status);
  CHECK(status);

  M_TensorSpec *pixelValuesSpec =
      M_newTensorSpec(pixelValuesShape, /*rankSize=*/4, /*dtype=*/M_FLOAT32,
                      /*tensorName=*/"pixel_values");
  float *pixelValuesTensor =
      (float *)readFileOrExit("inputs/pixel_values.bin");
  M_borrowTensorInto(inputToModel, pixelValuesTensor, pixelValuesSpec,
                     status);
  CHECK(status);

  M_TensorSpec *attentionMaskSpec =
      M_newTensorSpec(attentionMaskShape, /*rankSize=*/2, /*dtype=*/M_INT64,
                      /*tensorName=*/"attention_mask");
  int64_t *attentionMaskTensor =
      (int64_t *)readFileOrExit("inputs/attention_mask.bin");
  M_borrowTensorInto(inputToModel, attentionMaskTensor, attentionMaskSpec,
                     status);
  CHECK(status);

  logInfo("Running Inference...");
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputToModel, status);
  CHECK(status);

  M_AsyncValue *resultValue =
      M_getValueByNameFrom(outputs, /*tensorName=*/"result0", status);
  CHECK(status);

  logInfo("Extracting output values");
  // Convert the value we found to a tensor and save it to disk.
  M_AsyncTensor *result = M_getTensorFromValue(resultValue);
  size_t numElements = M_getTensorNumElements(result);
  printf("Tensor size: %ld\n", numElements);
  M_Dtype dtype = M_getTensorType(result);
  const char *outputFilePath = "outputs.bin";
  FILE *file = fopen(outputFilePath, "wb");
  if (!file) {
    printf("failed to open %s. Aborting.\n", outputFilePath);
    return EXIT_FAILURE;
  }
  fwrite(M_getTensorData(result), M_sizeOf(dtype), numElements, file);
  fclose(file);

  // free memory buffer
  free(inputIdsTensor);
  free(pixelValuesTensor);
  free(attentionMaskTensor);
  free(inputIdsShape);
  free(pixelValuesShape);
  free(attentionMaskShape);

  // free resources
  M_freeTensor(result);

  M_freeValue(resultValue);

  M_freeAsyncTensorMap(outputs);

  M_freeTensorSpec(attentionMaskSpec);
  M_freeTensorSpec(pixelValuesSpec);
  M_freeTensorSpec(inputIdsSpec);

  M_freeAsyncTensorMap(inputToModel);

  M_freeTensorNameArray(tensorNames);

  M_freeModel(model);

  M_freeCompileConfig(compileConfig);
  M_freeCompiledModel(compiledModel);

  M_freeTorchInputSpec(pixelValuesInputSpec);
  M_freeTorchInputSpec(attentionMaskInputSpec);
  M_freeTorchInputSpec(inputIdsInputSpec);

  M_freeCompileConfig(compileConfig);
  M_freeRuntimeContext(context);

  M_freeStatus(status);

  logInfo("Inference successfully completed");
  return EXIT_SUCCESS;
}
