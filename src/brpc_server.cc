/*
 # Copyright (c) 2020 Qihoo Inc. All rights reserved.
 * @File    : 2022/07/28 brpc_server.cc
 * @Author  : sunxiaodong (sunxiaodong@360.cn)
 */

// #include "triton/common/logging.h"

#include "brpc_server.h"

#include <algorithm>
#include <functional>
#include <future>
#include <iostream>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "google/protobuf/util/json_util.h"
#include "triton/common/logging.h"

#define RET_IF_ERR(X)                                                        \
  do {                                                                       \
    TRITONSERVER_Error* err__ = (X);                                         \
    if (err__ != nullptr) {                                                  \
      std::cerr << "error: " << TRITONSERVER_ErrorCodeString(err__) << " - " \
                << TRITONSERVER_ErrorMessage(err__) << std::endl;            \
      TRITONSERVER_ErrorDelete(err__);                                       \
      return;                                                                \
    }                                                                        \
  } while (false)

namespace triton { namespace server {


//
// ShmInfo
//
// Simple structure that carries the shared memory information
//
struct ShmInfo {
  ShmInfo(
      void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_id, char* cuda_ipc_handle)
      : base_(base), byte_size_(byte_size), memory_type_(memory_type),
        memory_type_id_(memory_type_id), cuda_ipc_handle_(cuda_ipc_handle)
  {
  }
  void* base_;
  size_t byte_size_;
  TRITONSERVER_MemoryType memory_type_;
  int64_t memory_type_id_;
  char* cuda_ipc_handle_;
};
using TensorShmMap = std::unordered_map<std::string, ShmInfo>;

//
// A simple queue holding the responses to be written. Uses a
// vector of persistent message objects to prevent allocating
// memory for each response to be written.
//
template <typename ResponseType>
class ResponseQueue {
 public:
  ResponseQueue() { Reset(); }

  ~ResponseQueue()
  {
    for (auto response : responses_) {
      delete response;
    }
  }

  // Resets the queue
  void Reset()
  {
    alloc_count_ = 0;
    ready_count_ = 0;
    current_index_ = 0;
    for (auto response : responses_) {
      response->Clear();
    }
  }
  // Gets the response for the non-decoupled models.
  // Note that there will be a single response in
  // non-decoupled cases.
  ResponseType* GetNonDecoupledResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    alloc_count_ = 1;
    if (responses_.size() < 1) {
      responses_.push_back(new ResponseType());
    }
    return responses_[0];
  }

  // Allocates a response on the head of the queue
  void AllocateResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    alloc_count_++;
    if (responses_.size() < alloc_count_) {
      responses_.push_back(new ResponseType());
    }
  }

  // Gets the last allocated response
  ResponseType* GetLastAllocatedResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (responses_.size() < alloc_count_) {
      LOG(ERROR)
          << "[INTERNAL] Attempting to access the response not yet allocated";
      return nullptr;
    }
    return responses_[alloc_count_ - 1];
  }

  // Marks the next non-ready response complete
  bool MarkNextResponseComplete()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (alloc_count_ <= ready_count_) {
      LOG(ERROR)
          << "[INTERNAL] Attempting to mark an unallocated response complete";
      return false;
    }
    ready_count_++;

    return true;
  }

  // Gets the current response from the tail of
  // the queue.
  ResponseType* GetCurrentResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (current_index_ >= ready_count_) {
      LOG(ERROR) << "[INTERNAL] Attempting to access current response when it "
                    "is not ready";
      return nullptr;
    }
    return responses_[current_index_];
  }

  // Gets the response at the specified index
  ResponseType* GetResponseAt(const uint32_t index)
  {
    std::lock_guard<std::mutex> lock(mtx_);
    if (index >= alloc_count_) {
      LOG(ERROR) << "[INTERNAL] Attempting to access response which is not yet "
                    "allocated";
      return nullptr;
    }
    return responses_[index];
  }

  // Pops the response from the tail of the queue
  void PopResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    current_index_++;
  }

  // Returns whether the queue is empty
  bool IsEmpty()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return ((alloc_count_ == ready_count_) && (alloc_count_ == current_index_));
  }

  // Returns whether the queue has responses
  // ready to be written.
  bool HasReadyResponse()
  {
    std::lock_guard<std::mutex> lock(mtx_);
    return (ready_count_ > current_index_);
  }

 private:
  std::vector<ResponseType*> responses_;
  std::mutex mtx_;

  // There are three indices to track the responses in the queue
  // Tracks the allocated response
  uint32_t alloc_count_;
  // Tracks the response that is ready to be written
  uint32_t ready_count_;
  // Tracks the response next in the queue to be written
  uint32_t current_index_;
};


//
// AllocPayload
//
// Simple structure that carries the userp payload needed for
// allocation.
//
template <typename ResponseType>
struct AllocPayload {
  using ClassificationMap = std::unordered_map<std::string, uint32_t>;

  AllocPayload() : response_queue_(nullptr) {}
  ~AllocPayload() {
  }  // Don't delete 'response_'.. it is owned by the InferHandlerState

  std::shared_ptr<ResponseQueue<ResponseType>> response_queue_;
  uint32_t response_alloc_count_;
  TensorShmMap shm_map_;
  ClassificationMap classification_map_;

  // Used to extend the lifetime of the serialized data in case
  // non-raw contents were provided in the request. Serialized data's
  // actual lifetime is that of the request whereas AllocPayload's
  // lifetime is that of a response... but it is convenient to keep it
  // here.
  std::list<std::string> serialized_data_;
};


// Make sure to keep InferResponseAlloc and OutputBufferQuery logic in sync
TRITONSERVER_Error*
InferResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
    int64_t preferred_memory_type_id, void* userp, void** buffer,
    void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
    int64_t* actual_memory_type_id)
{
  LOG(INFO) << "ModelInfer::InferResponseAlloc";
  *actual_memory_type = preferred_memory_type;
  *actual_memory_type_id = preferred_memory_type_id;

  // inference::ModelInferResponse::InferOutputTensor* output_tensor =
  //     response->add_outputs();
  // output_tensor->set_name(tensor_name);

  if (byte_size == 0) {
    *buffer = nullptr;
    *buffer_userp = nullptr;
    LOG(INFO) << "allocated " << byte_size << " bytes for result tensor "
              << tensor_name;
  } else {
    void* allocated_ptr = nullptr;
    switch (*actual_memory_type) {
#ifdef TRITON_ENABLE_GPU
#endif
      case TRITONSERVER_MEMORY_CPU:
      default: {
        *actual_memory_type = TRITONSERVER_MEMORY_CPU;
        allocated_ptr = malloc(byte_size);
        break;
      }
    }
    if (allocated_ptr != nullptr) {
      *buffer = allocated_ptr;
      *buffer_userp = new std::string(tensor_name);
      LOG(INFO) << "allocated " << byte_size << " bytes in "
                << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                << " for result tensor " << tensor_name;
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
InferResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  // Don't do anything when releasing a buffer since InferResponseAlloc
  // wrote directly into the response protobuf.
  LOG(INFO) << "ModelInfer::InferResponseRelease";
  return nullptr;
}


TRITONSERVER_Error*
InferResponseStart(TRITONSERVER_ResponseAllocator* allocator, void* userp)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  // ModelInfer RPC expects exactly one response per request. Hence, always call
  // GetNonDecoupledResponse() to create one response object on response start.
  payload->response_queue_->GetNonDecoupledResponse();

  return nullptr;  // success
}


TRITONSERVER_Error*
OutputBufferQueryHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t* byte_size, const TensorShmMap& shm_map,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  // Check if shared memory is used if named tensor is provided
  if (tensor_name != nullptr) {
    const auto& pr = shm_map.find(tensor_name);
    if (pr != shm_map.end()) {
      // The output is in shared memory so check that shared memory
      // size is at least large enough for the output, if byte size is provided
      if ((byte_size != nullptr) && (*byte_size > pr->second.byte_size_)) {
        // Don't return error yet and just set to the default properties for
        // GRPC buffer, error will be raised when allocation happens
        *memory_type = TRITONSERVER_MEMORY_CPU;
        *memory_type_id = 0;
      } else {
        *memory_type = pr->second.memory_type_;
        *memory_type_id = pr->second.memory_type_id_;
      }
      return nullptr;  // Success
    }
  }

  // Not using shared memory so a buffer created directly in
  // the response protobuf will be used, and the type will be CPU.
  *memory_type = TRITONSERVER_MEMORY_CPU;
  *memory_type_id = 0;
  return nullptr;  // Success
}


TRITONSERVER_Error*
OutputBufferAttributesHelper(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    const TensorShmMap& shm_map,
    TRITONSERVER_BufferAttributes* buffer_attributes)
{
  // We only need to set the cuda ipc handle here. The rest of the buffer
  // attributes have been properly populated by triton core.
  if (tensor_name != nullptr) {
    const auto& pr = shm_map.find(tensor_name);

    if (pr != shm_map.end()) {
      if (pr->second.memory_type_ == TRITONSERVER_MEMORY_GPU) {
        RETURN_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
            buffer_attributes, pr->second.cuda_ipc_handle_));
      }
    }
  }

  return nullptr;  // Success
}

TRITONSERVER_Error*
OutputBufferQuery(
    TRITONSERVER_ResponseAllocator* allocator, void* userp,
    const char* tensor_name, size_t* byte_size,
    TRITONSERVER_MemoryType* memory_type, int64_t* memory_type_id)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  return OutputBufferQueryHelper(
      allocator, tensor_name, byte_size, payload->shm_map_, memory_type,
      memory_type_id);
}

// Make sure to keep InferResponseAlloc, OutputBufferQuery, and
// OutputBufferAttributes logic in sync
TRITONSERVER_Error*
OutputBufferAttributes(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    TRITONSERVER_BufferAttributes* buffer_attributes, void* userp,
    void* buffer_userp)
{
  AllocPayload<inference::ModelInferResponse>* payload =
      reinterpret_cast<AllocPayload<inference::ModelInferResponse>*>(userp);

  return OutputBufferAttributesHelper(
      allocator, tensor_name, payload->shm_map_, buffer_attributes);
  return nullptr;  // Success
}

void
InferRequestComplete(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  LOG(INFO) << "ModelInferHandler::InferRequestComplete";

  if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
    LOG(INFO) << "Calling TRITONSERVER_InferenceRequestDelete";
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(request),
        "deleting BRPC inference request");
  }
}

void
InferResponseComplete(
    TRITONSERVER_InferenceResponse* iresponse, const uint32_t flags,
    void* userp)
{
  LOG(INFO) << "ModelInferHandler::InferResponseComplete";
  std::promise<TRITONSERVER_InferenceResponse*>* p =
      reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
  p->set_value(iresponse);
  delete p;
  if ((flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) == 0) {
    LOG(ERROR)
        << "[INTERNAL] ModelInfer received a response without FINAL flag";
    return;
  }
}

BRPCServer::BRPCServer(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port)
    : server_(server), trace_manager_(trace_manager), shm_manager_(shm_manager),
      impl_(server), port_(port)
{
  brpc_server_ = std::make_unique<brpc::Server>();
  brpc_server_->AddService(&impl_, brpc::SERVER_DOESNT_OWN_SERVICE);
}


TRITONSERVER_Error*
BRPCServer::Create(
    const std::shared_ptr<TRITONSERVER_Server>& server,
    triton::server::TraceManager* trace_manager,
    const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
    std::unique_ptr<BRPCServer>* brpc_server)
{
  brpc_server->reset(new BRPCServer(server, trace_manager, shm_manager, port));
  return nullptr;
}

BRPCServer::~BRPCServer()
{
  Stop();
}


TRITONSERVER_Error*
BRPCServer::Start()
{
  LOG(INFO) << "Started BRPC Service at 0.0.0.0:" << port_;
  brpc_server_->Start(port_, &options_);
  return nullptr;
}

TRITONSERVER_Error*
BRPCServer::Stop()
{
  brpc_server_->Stop(0);
  return nullptr;
}

// InferenceServiceImpl
InferenceServiceImpl::InferenceServiceImpl(
    const std::shared_ptr<TRITONSERVER_Server>& tritonserver)
    : tritonserver_(tritonserver)
{
}

InferenceServiceImpl::~InferenceServiceImpl() {}

void
InferenceServiceImpl::ServerLive(
    google::protobuf::RpcController* cntl_base,
    const inference::ServerLiveRequest* request,
    inference::ServerLiveResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  bool live = false;
  TRITONSERVER_Error* err =
      TRITONSERVER_ServerIsLive(tritonserver_.get(), &live);
  response->set_live((err == nullptr) && live);
  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ServerReady(
    google::protobuf::RpcController* cntl_base,
    const inference::ServerReadyRequest* request,
    inference::ServerReadyResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  bool ready = false;
  TRITONSERVER_Error* err =
      TRITONSERVER_ServerIsReady(tritonserver_.get(), &ready);
  response->set_ready((err == nullptr) && ready);
  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ModelReady(
    google::protobuf::RpcController* cntl_base,
    const inference::ModelReadyRequest* request,
    inference::ModelReadyResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  bool is_ready = false;
  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  if (err == nullptr) {
    err = TRITONSERVER_ServerModelIsReady(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        &is_ready);
  }

  response->set_ready(is_ready);
  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ServerMetadata(
    google::protobuf::RpcController* cntl_base,
    const inference::ServerMetadataRequest* request,
    inference::ServerMetadataResponse* response,
    google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);

  TRITONSERVER_Message* server_metadata_message = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_ServerMetadata(
      tritonserver_.get(), &server_metadata_message);
  RET_IF_ERR(err);

  const char* buffer;
  size_t byte_size;
  err = TRITONSERVER_MessageSerializeToJson(
      server_metadata_message, &buffer, &byte_size);
  RET_IF_ERR(err);
  {
    triton::common::TritonJson::Value server_metadata_json;
    err = server_metadata_json.Parse(buffer, byte_size);
    RET_IF_ERR(err);

    const char* name;
    size_t namelen;
    err = server_metadata_json.MemberAsString("name", &name, &namelen);
    RET_IF_ERR(err);

    const char* version;
    size_t versionlen;
    err = server_metadata_json.MemberAsString("version", &version, &versionlen);
    RET_IF_ERR(err);

    response->set_name(std::string(name, namelen));
    response->set_version(std::string(version, versionlen));

    if (server_metadata_json.Find("extensions")) {
      triton::common::TritonJson::Value extensions_json;
      err = server_metadata_json.MemberAsArray("extensions", &extensions_json);
      RET_IF_ERR(err);

      for (size_t idx = 0; idx < extensions_json.ArraySize(); ++idx) {
        const char* ext;
        size_t extlen;
        err = extensions_json.IndexAsString(idx, &ext, &extlen);
        RET_IF_ERR(err);
        response->add_extensions(std::string(ext, extlen));
      }
    }
    TRITONSERVER_MessageDelete(server_metadata_message);
  }
  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ModelMetadata(
    google::protobuf::RpcController* cntl_base,
    const inference::ModelMetadataRequest* request,
    inference::ModelMetadataResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  RET_IF_ERR(err);

  {
    TRITONSERVER_Message* model_metadata_message = nullptr;
    err = TRITONSERVER_ServerModelMetadata(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        &model_metadata_message);
    RET_IF_ERR(err);

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        model_metadata_message, &buffer, &byte_size);
    RET_IF_ERR(err);

    triton::common::TritonJson::Value model_metadata_json;
    err = model_metadata_json.Parse(buffer, byte_size);
    RET_IF_ERR(err);

    const char* name;
    size_t namelen;
    err = model_metadata_json.MemberAsString("name", &name, &namelen);
    RET_IF_ERR(err);

    response->set_name(std::string(name, namelen));

    if (model_metadata_json.Find("versions")) {
      triton::common::TritonJson::Value versions_json;
      err = model_metadata_json.MemberAsArray("versions", &versions_json);
      RET_IF_ERR(err);

      for (size_t idx = 0; idx < versions_json.ArraySize(); ++idx) {
        const char* version;
        size_t versionlen;
        err = versions_json.IndexAsString(idx, &version, &versionlen);
        RET_IF_ERR(err);
        response->add_versions(std::string(version, versionlen));
      }
    }

    const char* platform;
    size_t platformlen;
    err =
        model_metadata_json.MemberAsString("platform", &platform, &platformlen);
    RET_IF_ERR(err);
    response->set_platform(std::string(platform, platformlen));

    if (model_metadata_json.Find("inputs")) {
      triton::common::TritonJson::Value inputs_json;
      err = model_metadata_json.MemberAsArray("inputs", &inputs_json);
      RET_IF_ERR(err);

      for (size_t idx = 0; idx < inputs_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value io_json;
        err = inputs_json.IndexAsObject(idx, &io_json);
        RET_IF_ERR(err);

        inference::ModelMetadataResponse::TensorMetadata* io =
            response->add_inputs();

        const char* name;
        size_t namelen;
        err = io_json.MemberAsString("name", &name, &namelen);
        RET_IF_ERR(err);

        const char* datatype;
        size_t datatypelen;
        err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
        RET_IF_ERR(err);

        io->set_name(std::string(name, namelen));
        io->set_datatype(std::string(datatype, datatypelen));

        if (io_json.Find("shape")) {
          triton::common::TritonJson::Value shape_json;
          err = io_json.MemberAsArray("shape", &shape_json);
          RET_IF_ERR(err);

          for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
            int64_t d;
            err = shape_json.IndexAsInt(sidx, &d);
            RET_IF_ERR(err);

            io->add_shape(d);
          }
        }
      }
    }

    if (model_metadata_json.Find("outputs")) {
      triton::common::TritonJson::Value outputs_json;
      err = model_metadata_json.MemberAsArray("outputs", &outputs_json);
      RET_IF_ERR(err);

      for (size_t idx = 0; idx < outputs_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value io_json;
        err = outputs_json.IndexAsObject(idx, &io_json);
        RET_IF_ERR(err);

        inference::ModelMetadataResponse::TensorMetadata* io =
            response->add_outputs();

        const char* name;
        size_t namelen;
        err = io_json.MemberAsString("name", &name, &namelen);
        RET_IF_ERR(err);

        const char* datatype;
        size_t datatypelen;
        err = io_json.MemberAsString("datatype", &datatype, &datatypelen);
        RET_IF_ERR(err);

        io->set_name(std::string(name, namelen));
        io->set_datatype(std::string(datatype, datatypelen));

        if (io_json.Find("shape")) {
          triton::common::TritonJson::Value shape_json;
          err = io_json.MemberAsArray("shape", &shape_json);
          RET_IF_ERR(err);

          for (size_t sidx = 0; sidx < shape_json.ArraySize(); ++sidx) {
            int64_t d;
            err = shape_json.IndexAsInt(sidx, &d);
            RET_IF_ERR(err);

            io->add_shape(d);
          }
        }
      }
    }

    TRITONSERVER_MessageDelete(model_metadata_message);
  }
  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ModelConfig(
    google::protobuf::RpcController* cntl_base,
    const inference::ModelConfigRequest* request,
    inference::ModelConfigResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  if (err == nullptr) {
    TRITONSERVER_Message* model_config_message = nullptr;
    err = TRITONSERVER_ServerModelConfig(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        1 /* config_version */, &model_config_message);
    if (err == nullptr) {
      const char* buffer;
      size_t byte_size;
      err = TRITONSERVER_MessageSerializeToJson(
          model_config_message, &buffer, &byte_size);
      if (err == nullptr) {
        ::google::protobuf::util::JsonStringToMessage(
            {buffer, static_cast<int>(byte_size)}, response->mutable_config());
      }
      TRITONSERVER_MessageDelete(model_config_message);
    }
  }

  TRITONSERVER_ErrorDelete(err);
}

void
InferenceServiceImpl::ModelStatistics(
    google::protobuf::RpcController* cntl_base,
    const inference::ModelStatisticsRequest* request,
    inference::ModelStatisticsResponse* response,
    google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
#ifdef TRITON_ENABLE_STATS
  triton::common::TritonJson::Value model_stats_json;

  int64_t requested_model_version;
  auto err =
      GetModelVersionFromString(request->version(), &requested_model_version);
  RET_IF_ERR(err);

  {
    TRITONSERVER_Message* model_stats_message = nullptr;
    err = TRITONSERVER_ServerModelStatistics(
        tritonserver_.get(), request->name().c_str(), requested_model_version,
        &model_stats_message);
    RET_IF_ERR(err);

    const char* buffer;
    size_t byte_size;
    err = TRITONSERVER_MessageSerializeToJson(
        model_stats_message, &buffer, &byte_size);
    RET_IF_ERR(err);

    err = model_stats_json.Parse(buffer, byte_size);
    RET_IF_ERR(err);

    TRITONSERVER_MessageDelete(model_stats_message);
  }

  if (model_stats_json.Find("model_stats")) {
    triton::common::TritonJson::Value stats_json;
    err = model_stats_json.MemberAsArray("model_stats", &stats_json);
    RET_IF_ERR(err);

    for (size_t idx = 0; idx < stats_json.ArraySize(); ++idx) {
      triton::common::TritonJson::Value model_stat;
      err = stats_json.IndexAsObject(idx, &model_stat);
      RET_IF_ERR(err);

      auto statistics = response->add_model_stats();

      const char* name;
      size_t namelen;
      err = model_stat.MemberAsString("name", &name, &namelen);
      RET_IF_ERR(err);

      const char* version;
      size_t versionlen;
      err = model_stat.MemberAsString("version", &version, &versionlen);
      RET_IF_ERR(err);

      statistics->set_name(std::string(name, namelen));
      statistics->set_version(std::string(version, versionlen));

      uint64_t ucnt;
      err = model_stat.MemberAsUInt("last_inference", &ucnt);
      RET_IF_ERR(err);
      statistics->set_last_inference(ucnt);

      err = model_stat.MemberAsUInt("inference_count", &ucnt);
      RET_IF_ERR(err);
      statistics->set_inference_count(ucnt);

      err = model_stat.MemberAsUInt("execution_count", &ucnt);
      RET_IF_ERR(err);
      statistics->set_execution_count(ucnt);

      triton::common::TritonJson::Value infer_stats_json;
      err = model_stat.MemberAsObject("inference_stats", &infer_stats_json);
      RET_IF_ERR(err);

      {
        triton::common::TritonJson::Value success_json;
        err = infer_stats_json.MemberAsObject("success", &success_json);
        RET_IF_ERR(err);

        err = success_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_success()->set_count(
            ucnt);
        err = success_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_success()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value fail_json;
        err = infer_stats_json.MemberAsObject("fail", &fail_json);
        RET_IF_ERR(err);

        err = fail_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_fail()->set_count(ucnt);
        err = fail_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_fail()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value queue_json;
        err = infer_stats_json.MemberAsObject("queue", &queue_json);
        RET_IF_ERR(err);

        err = queue_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_queue()->set_count(ucnt);
        err = queue_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_queue()->set_ns(ucnt);
      }

      {
        triton::common::TritonJson::Value compute_input_json;
        err = infer_stats_json.MemberAsObject(
            "compute_input", &compute_input_json);
        RET_IF_ERR(err);

        err = compute_input_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()
            ->mutable_compute_input()
            ->set_count(ucnt);
        err = compute_input_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_compute_input()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value compute_infer_json;
        err = infer_stats_json.MemberAsObject(
            "compute_infer", &compute_infer_json);
        RET_IF_ERR(err);

        err = compute_infer_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()
            ->mutable_compute_infer()
            ->set_count(ucnt);
        err = compute_infer_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_compute_infer()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value compute_output_json;
        err = infer_stats_json.MemberAsObject(
            "compute_output", &compute_output_json);
        RET_IF_ERR(err);

        err = compute_output_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()
            ->mutable_compute_output()
            ->set_count(ucnt);
        err = compute_output_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_compute_output()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value cache_hit_json;
        err = infer_stats_json.MemberAsObject("cache_hit", &cache_hit_json);
        RET_IF_ERR(err);

        err = cache_hit_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_cache_hit()->set_count(
            ucnt);
        err = cache_hit_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_cache_hit()->set_ns(
            ucnt);
      }

      {
        triton::common::TritonJson::Value cache_miss_json;
        err = infer_stats_json.MemberAsObject("cache_miss", &cache_miss_json);
        RET_IF_ERR(err);

        err = cache_miss_json.MemberAsUInt("count", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_cache_miss()->set_count(
            ucnt);
        err = cache_miss_json.MemberAsUInt("ns", &ucnt);
        RET_IF_ERR(err);
        statistics->mutable_inference_stats()->mutable_cache_miss()->set_ns(
            ucnt);
      }


      triton::common::TritonJson::Value batches_json;
      err = model_stat.MemberAsArray("batch_stats", &batches_json);
      RET_IF_ERR(err);

      for (size_t idx = 0; idx < batches_json.ArraySize(); ++idx) {
        triton::common::TritonJson::Value batch_stat;
        err = batches_json.IndexAsObject(idx, &batch_stat);
        RET_IF_ERR(err);

        auto batch_statistics = statistics->add_batch_stats();

        uint64_t ucnt;
        err = batch_stat.MemberAsUInt("batch_size", &ucnt);
        RET_IF_ERR(err);
        batch_statistics->set_batch_size(ucnt);

        {
          triton::common::TritonJson::Value compute_input_json;
          err = batch_stat.MemberAsObject("compute_input", &compute_input_json);
          RET_IF_ERR(err);

          err = compute_input_json.MemberAsUInt("count", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_input()->set_count(ucnt);
          err = compute_input_json.MemberAsUInt("ns", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_input()->set_ns(ucnt);
        }

        {
          triton::common::TritonJson::Value compute_infer_json;
          err = batch_stat.MemberAsObject("compute_infer", &compute_infer_json);
          RET_IF_ERR(err);

          err = compute_infer_json.MemberAsUInt("count", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_infer()->set_count(ucnt);
          err = compute_infer_json.MemberAsUInt("ns", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_infer()->set_ns(ucnt);
        }

        {
          triton::common::TritonJson::Value compute_output_json;
          err =
              batch_stat.MemberAsObject("compute_output", &compute_output_json);
          RET_IF_ERR(err);

          err = compute_output_json.MemberAsUInt("count", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_output()->set_count(ucnt);
          err = compute_output_json.MemberAsUInt("ns", &ucnt);
          RET_IF_ERR(err);
          batch_statistics->mutable_compute_output()->set_ns(ucnt);
        }
      }
    }
  }
#else
  auto err = TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNAVAILABLE,
      "the server does not suppport model statistics");
#endif
  TRITONSERVER_ErrorDelete(err);
}

TRITONSERVER_Error*
InferGRPCToInputHelper(
    const std::string& input_name, const std::string& model_name,
    const TRITONSERVER_DataType tensor_dt, const TRITONSERVER_DataType input_dt,
    const size_t binary_data_byte_size)
{
  if (binary_data_byte_size != 0) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name +
            "', binary data was already supplied.")
            .c_str());
  }

  if (tensor_dt != input_dt) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string(
            "unexpected explicit tensor data for input tensor '" + input_name +
            "' for model '" + model_name + "' of type '" +
            TRITONSERVER_DataTypeString(tensor_dt) + "', expected datatype '" +
            TRITONSERVER_DataTypeString(input_dt) + "'")
            .c_str());
  }

  return nullptr;  // success
}

void
InferenceServiceImpl::ModelInfer(
    google::protobuf::RpcController* cntl_base,
    const inference::ModelInferRequest* request,
    inference::ModelInferResponse* response, google::protobuf::Closure* done)
{
  brpc::ClosureGuard done_guard(done);
  TRITONSERVER_Error* err = nullptr;
  LOG(INFO) << "Processing for " << request->model_name();

  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  RET_IF_ERR(TRITONSERVER_ResponseAllocatorNew(
      &allocator, InferResponseAlloc, InferResponseRelease, nullptr));

  int64_t requested_model_version;
  err = GetModelVersionFromString(
      request->model_version(), &requested_model_version);

  TRITONSERVER_InferenceRequest* irequest = nullptr;
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestNew(
        &irequest, tritonserver_.get(), request->model_name().c_str(),
        requested_model_version);
  }

  // SetInferenceRequestMetadata
  // Parameters is not supported by default.
  if (err == nullptr) {
    RET_IF_ERR(
        TRITONSERVER_InferenceRequestSetId(irequest, request->id().c_str()));

    for (const auto& input : request->inputs()) {
      RET_IF_ERR(TRITONSERVER_InferenceRequestAddInput(
          irequest, input.name().c_str(),
          TRITONSERVER_StringToDataType(input.datatype().c_str()),
          input.shape().data(), input.shape_size()));
    }

    for (const auto& output : request->outputs()) {
      RET_IF_ERR(TRITONSERVER_InferenceRequestAddRequestedOutput(
          irequest, output.name().c_str()));
    }
  }

  // InferGRPCToInput
  if (err == nullptr) {
    // int index = 0;
    for (const auto& io : request->inputs()) {
      // ParseSharedMemoryParams
      // SharedMemoryParams is not supported by default.

      const void* base = nullptr;
      base = nullptr;
      size_t byte_size = 0;
      TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t memory_type_id = 0;

      TRITONSERVER_BufferAttributes* buffer_attributes;
      RET_IF_ERR(TRITONSERVER_BufferAttributesNew(&buffer_attributes));
      auto buffer_attributes_del =
          [](TRITONSERVER_BufferAttributes* buffer_attributes) {
            TRITONSERVER_BufferAttributesDelete(buffer_attributes);
          };
      std::unique_ptr<
          TRITONSERVER_BufferAttributes, decltype(buffer_attributes_del)>
          buffer_attrsl(buffer_attributes, buffer_attributes_del);
      char* cuda_ipc_handle = nullptr;

      if (io.has_contents() && (!request->raw_input_contents().empty())) {
        std::string err_msg =
            "contents field must not be specified when using "
            "raw_input_contents for '" +
            io.name() + "' for model '" + request->model_name() + "'";
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, err_msg.c_str());
        break;
      } else if (io.has_contents()) {
        TRITONSERVER_DataType dtype =
            TRITONSERVER_StringToDataType(io.datatype().c_str());
        const size_t elem_byte_size = TRITONSERVER_DataTypeByteSize(dtype);

        if (io.contents().bool_contents_size() != 0) {
          RET_IF_ERR(InferGRPCToInputHelper(
              io.name(), request->model_name(), TRITONSERVER_TYPE_BOOL, dtype,
              byte_size));
          base = static_cast<const void*>(io.contents().bool_contents().data());
          byte_size = io.contents().bool_contents_size() * elem_byte_size;
        }
        if (io.contents().int_contents_size() != 0) {
          // INT8 and INT16 is not supported by default.
          RET_IF_ERR(InferGRPCToInputHelper(
              io.name(), request->model_name(), TRITONSERVER_TYPE_INT32, dtype,
              byte_size));
          base = (const void*)io.contents().int_contents().data();
          byte_size = io.contents().int_contents_size() * elem_byte_size;
        }
        if (io.contents().fp32_contents_size() != 0) {
          RET_IF_ERR(InferGRPCToInputHelper(
              io.name(), request->model_name(), TRITONSERVER_TYPE_FP32, dtype,
              byte_size));
          base = (const void*)io.contents().fp32_contents().data();
          byte_size = io.contents().fp32_contents_size() * elem_byte_size;
        }
        if (io.contents().int64_contents_size() != 0) {
          RET_IF_ERR(InferGRPCToInputHelper(
              io.name(), request->model_name(), TRITONSERVER_TYPE_INT64, dtype,
              byte_size));
          base = (const void*)io.contents().int64_contents().data();
          byte_size = io.contents().int64_contents_size() * elem_byte_size;
        }
      } else {
        std::string err_msg = "unable to find data for input tensor '" +
                              io.name() + "' for model '" +
                              request->model_name() + "' in request.";
        err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, err_msg.c_str());
        break;
      }
      if (cuda_ipc_handle != nullptr) {
        RET_IF_ERR(TRITONSERVER_BufferAttributesSetCudaIpcHandle(
            buffer_attributes, reinterpret_cast<void*>(cuda_ipc_handle)));
      }

      RET_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryType(
          buffer_attributes, memory_type));
      RET_IF_ERR(TRITONSERVER_BufferAttributesSetMemoryTypeId(
          buffer_attributes, memory_type_id));
      RET_IF_ERR(TRITONSERVER_BufferAttributesSetByteSize(
          buffer_attributes, byte_size));
      RET_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputDataWithBufferAttributes(
              irequest, io.name().c_str(), base, buffer_attributes));
    }
  }

  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetReleaseCallback(
        irequest, InferRequestComplete, nullptr /* request_release_userp */);
  }

  auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
  std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();
  if (err == nullptr) {
    err = TRITONSERVER_InferenceRequestSetResponseCallback(
        irequest, allocator, nullptr, InferResponseComplete,
        reinterpret_cast<void*>(p));
  }

  // TRITONSERVER_InferenceRequestId
  const char* request_id = nullptr;
  if (err == nullptr) {
    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestId(irequest, &request_id),
        "unable to retrieve request ID string");
  }
  if ((request_id == nullptr) || (request_id[0] == '\0')) {
    request_id = "<id_unknown>";
  }

  if (err == nullptr) {
    TRITONSERVER_InferenceTrace* triton_trace = nullptr;
    err = TRITONSERVER_ServerInferAsync(
        tritonserver_.get(), irequest, triton_trace);
  }

  TRITONSERVER_InferenceResponse* iresponse = completed.get();
  RET_IF_ERR(TRITONSERVER_InferenceResponseError(iresponse));

  // Set response
  response->set_id(request_id);
  response->set_model_name(request->model_name());
  response->set_model_version(std::to_string(requested_model_version));

  uint32_t output_count;
  RET_IF_ERR(
      TRITONSERVER_InferenceResponseOutputCount(iresponse, &output_count));
  for (size_t output_idx = 0; output_idx < output_count; ++output_idx) {
    const char* cname;
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint64_t dim_count;
    const void* base;
    size_t byte_size;
    TRITONSERVER_MemoryType memory_type;
    int64_t memory_type_id;
    void* userp;

    RET_IF_ERR(TRITONSERVER_InferenceResponseOutput(
        iresponse, output_idx, &cname, &datatype, &shape, &dim_count, &base,
        &byte_size, &memory_type, &memory_type_id, &userp));

    inference::ModelInferResponse::InferOutputTensor* tensor =
        response->add_outputs();
    tensor->set_name(cname);
    tensor->set_datatype(TRITONSERVER_DataTypeString(datatype));
    for (size_t i = 0; i < dim_count; ++i) {
      tensor->add_shape(shape[i]);
    }

    if (datatype == TRITONSERVER_TYPE_FP32) {
      const float* p = reinterpret_cast<const float*>(base);
      google::protobuf::RepeatedField<float> copy(
          p, p + byte_size / sizeof(float));
      tensor->mutable_contents()->mutable_fp32_contents()->Swap(&copy);
    }
  }

  RET_IF_ERR(TRITONSERVER_InferenceResponseDelete(iresponse));

  if (err != nullptr) {
    LOG(INFO) << "[request id: " << request_id << "] "
              << "Infer failed: " << TRITONSERVER_ErrorMessage(err);

    LOG_TRITONSERVER_ERROR(
        TRITONSERVER_InferenceRequestDelete(irequest),
        "deleting BRPC inference request");
    TRITONSERVER_ErrorDelete(err);
  }

  LOG_TRITONSERVER_ERROR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");
  if (err == nullptr) {
    LOG(INFO) << "err is nullptr";
  } else {
    LOG(INFO) << "err is not nullptr";
  }
}
}}  // namespace triton::server
