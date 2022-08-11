/*
 # Copyright (c) 2020 Qihoo Inc. All rights reserved.
 * @File    : 2022/07/28 brpc_server.h
 * @Author  : sunxiaodong (sunxiaodong@360.cn)
 */

#pragma once

#include <memory>
#include <string>

#include "brpc/server.h"
#include "grpc_service.pb.h"
#include "shared_memory_manager.h"
#include "tracer.h"
#include "triton/core/tritonserver.h"


namespace triton { namespace server {

class InferenceServiceImpl : public inference::GRPCInferenceService {
 public:
  explicit InferenceServiceImpl(
      const std::shared_ptr<TRITONSERVER_Server>& tritonserver);
  ~InferenceServiceImpl();
  void ServerLive(
      google::protobuf::RpcController* cntl_base,
      const inference::ServerLiveRequest* request,
      inference::ServerLiveResponse* response, google::protobuf::Closure* done);

  void ServerReady(
      google::protobuf::RpcController* cntl_base,
      const inference::ServerReadyRequest* request,
      inference::ServerReadyResponse* response,
      google::protobuf::Closure* done);

  void ModelReady(
      google::protobuf::RpcController* cntl_base,
      const inference::ModelReadyRequest* request,
      inference::ModelReadyResponse* response, google::protobuf::Closure* done);

  void ServerMetadata(
      google::protobuf::RpcController* cntl_base,
      const inference::ServerMetadataRequest* request,
      inference::ServerMetadataResponse* response,
      google::protobuf::Closure* done);

  void ModelMetadata(
      google::protobuf::RpcController* cntl_base,
      const inference::ModelMetadataRequest* request,
      inference::ModelMetadataResponse* response,
      google::protobuf::Closure* done);

  void ModelInfer(
      google::protobuf::RpcController* cntl_base,
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* response, google::protobuf::Closure* done);

  void ModelConfig(
      google::protobuf::RpcController* cntl_base,
      const inference::ModelConfigRequest* request,
      inference::ModelConfigResponse* response,
      google::protobuf::Closure* done);

  void ModelStatistics(
      google::protobuf::RpcController* cntl_base,
      const inference::ModelStatisticsRequest* request,
      inference::ModelStatisticsResponse* response,
      google::protobuf::Closure* done);

 private:
  std::shared_ptr<TRITONSERVER_Server> tritonserver_;
};

class BRPCServer {
 public:
  static TRITONSERVER_Error* Create(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port,
      std::unique_ptr<BRPCServer>* brpc_server);
  ~BRPCServer();
  TRITONSERVER_Error* Start();
  TRITONSERVER_Error* Stop();

 private:
  BRPCServer(
      const std::shared_ptr<TRITONSERVER_Server>& server,
      triton::server::TraceManager* trace_manager,
      const std::shared_ptr<SharedMemoryManager>& shm_manager, int32_t port);

  std::shared_ptr<TRITONSERVER_Server> server_;
  TraceManager* trace_manager_;
  std::shared_ptr<SharedMemoryManager> shm_manager_;

  InferenceServiceImpl impl_;
  brpc::ServerOptions options_;
  std::unique_ptr<brpc::Server> brpc_server_;
  int32_t port_;
  bool running_;
};

}}  // namespace triton::server
