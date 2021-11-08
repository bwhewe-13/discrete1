#include <stdio.h>
#include <stdlib.h>
// #include <stddef.h>
#include <string.h>
#include <tensorflow/c/c_api.h>
#include "autoencode.h"

void NoOpDeallocator(void* data, size_t a, void* b) {}

float* encode(float data[]){
    // printf("Hello from TensorFlow C library version %s\n",TF_Version());
    char* new_environment = "TF_CPP_MIN_LOG_LEVEL=3";
    int ret;
    ret = putenv(new_environment);

    TF_Graph* Graph = TF_NewGraph(); 
    TF_Status* Status = TF_NewStatus(); 

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "ae_models/encoder";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
    // Load Session
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    // Check to See if That worked
    // if(TF_GetCode(Status) == TF_OK)
    // {
    //     printf("TF_LoadSessionFromSavedModel OK\n");
    // }
    // else
    // {
    //     printf("%s",TF_Message(Status));
    // }

    //****** Get input tensor
    // One input tensor of length 87
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
   
    // if(t0.oper == NULL)
    //     printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    // else
    // printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0; 
  
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t1 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};   
    Output[0] = t1; 
 
    // if(t1.oper == NULL)
    //     printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    // else    
    // printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    int length = 87;
    int ndims = 2;
    int64_t dims[] = {1,length};

    int ndata = sizeof(int64_t)*length ;// This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    // if (int_tensor != NULL)
    // {
    //     printf("TF_NewTensor is OK\n");
    // }
    // else
    // printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;
    
    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

    // if(TF_GetCode(Status) == TF_OK)
    // {
    //     printf("Session is OK\n");
    // }
    // else
    // {
    //     printf("%s",TF_Message(Status));
    // }

    //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    TF_Tensor* tout = OutputValues[0];
    size_t nout;
    nout = (size_t)TF_Dim(tout,1);

    void* buff = TF_TensorData(tout);
    float* offsets = buff;

    return offsets;
}


float* decode(float data[]){
    // printf("Hello from TensorFlow C library version %s\n",TF_Version());
    char* new_environment = "TF_CPP_MIN_LOG_LEVEL=3";
    int ret;
    ret = putenv(new_environment);
    
    TF_Graph* Graph = TF_NewGraph(); 
    TF_Status* Status = TF_NewStatus(); 

    TF_SessionOptions* SessionOpts = TF_NewSessionOptions();
    TF_Buffer* RunOpts = NULL;

    const char* saved_model_dir = "ae_models/decoder";
    const char* tags = "serve"; // default model serving tag; can change in future
    int ntags = 1;
    // Load Session
    TF_Session* Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
    // Check to See if That worked
    // if(TF_GetCode(Status) == TF_OK)
    // {
    //     printf("TF_LoadSessionFromSavedModel OK\n");
    // }
    // else
    // {
    //     printf("%s",TF_Message(Status));
    // }

    //****** Get input tensor
    // One input tensor of length 87
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_2"), 0};
   
    // if(t0.oper == NULL)
    //     printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    // else
    // printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0; 
  
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t1 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};   
    Output[0] = t1; 
 
    // if(t1.oper == NULL)
    //     printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    // else    
    // printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    //********* Allocate data for inputs & outputs
    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);

    int length = 20;
    int ndims = 2;
    int64_t dims[] = {1,length};

    int ndata = sizeof(int64_t)*length ;// This is tricky, it number of bytes not number of element

    TF_Tensor* int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    // if (int_tensor != NULL)
    // {
    //     printf("TF_NewTensor is OK\n");
    // }
    // else
    // printf("ERROR: Failed TF_NewTensor\n");
    
    InputValues[0] = int_tensor;
    
    // //Run the Session
    TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

    // if(TF_GetCode(Status) == TF_OK)
    // {
    //     printf("Session is OK\n");
    // }
    // else
    // {
    //     printf("%s",TF_Message(Status));
    // }

    //Free memory
    TF_DeleteGraph(Graph);
    TF_DeleteSession(Session, Status);
    TF_DeleteSessionOptions(SessionOpts);
    TF_DeleteStatus(Status);

    TF_Tensor* tout = OutputValues[0];
    size_t nout;
    nout = (size_t)TF_Dim(tout,1);

    void* buff = TF_TensorData(tout);
    float* offsets = buff;

    return offsets;
}
