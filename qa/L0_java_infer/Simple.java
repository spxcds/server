// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
public class Infer {
    
    static TRITONSERVER_Error
    ParseModelMetadata(
        JsonObject model_metadata, String datatype, String framework_type)
    {
      String seen_data_type = null;
      for (JsonElement input_element : model_metadata.get("inputs").getAsJsonArray()) {
        JsonObject input = input_element.getAsJsonObject();
        if (!input.get("datatype").getAsString().equals("INT32") &&
            !input.get("datatype").getAsString().equals("FP32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "simple lib example only supports model with data type INT32 or " +
              "FP32");
        }
        if (seen_data_type == null) {
          seen_data_type = input.get("datatype").getAsString();
        } else if (!seen_data_type.equals(input.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of 'simple' model must have the data type");
        }
      }
      for (JsonElement output_element : model_metadata.get("outputs").getAsJsonArray()) {
        JsonObject output = output_element.getAsJsonObject();
        if (!output.get("datatype").getAsString().equals("INT32") &&
            !output.get("datatype").getAsString().equals("FP32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "simple lib example only supports model with data type INT32 or " +
              "FP32");
        } else if (!seen_data_type.equals(output.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of 'simple' model must have the data type");
        }
      }

      is_int[0] = seen_data_type.equals("INT32");
      is_torch_model[0] =
          model_metadata.get("platform").getAsString().equals("pytorch_libtorch");
      return null;
    }
    
    
    
    
    static class TRITONSERVER_ServerDeleter extends TRITONSERVER_Server {
      public TRITONSERVER_ServerDeleter(TRITONSERVER_Server p) {
        super(p);
        deallocator(new DeleteDeallocator(this)); 
        }
      protected static class DeleteDeallocator extends TRITONSERVER_Server implements Deallocator {
        DeleteDeallocator(Pointer p) { 
          super(p); 
          } @Override public void deallocate() { TRITONSERVER_ServerDelete(this); }
      }
    }



      static class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {
        @Override public TRITONSERVER_Error call (
            TRITONSERVER_ResponseAllocator allocator, String tensor_name,
            long byte_size, int preferred_memory_type,
            long preferred_memory_type_id, Pointer userp, PointerPointer buffer,
            PointerPointer buffer_userp, IntPointer actual_memory_type,
            LongPointer actual_memory_type_id)
        {
          // Initially attempt to make the actual memory type and id that we
          // allocate be the same as preferred memory type
          actual_memory_type.put(0, preferred_memory_type);
          actual_memory_type_id.put(0, preferred_memory_type_id);

          // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
          // need to do any other book-keeping.
          if (byte_size == 0) {
            buffer.put(0, null);
            buffer_userp.put(0, null);
          } else {
            Pointer allocated_ptr = new Pointer();
            if (enforce_memory_type) {
              actual_memory_type.put(0, requested_memory_type);
            }

            actual_memory_type.put(0, TRITONSERVER_MEMORY_CPU);
            allocated_ptr = Pointer.malloc(byte_size);

            // Pass the tensor name with buffer_userp so we can show it when
            // releasing the buffer.
            if (!allocated_ptr.isNull()) {
              buffer.put(0, allocated_ptr);
              buffer_userp.put(0, Loader.newGlobalRef(tensor_name));
            }
          }

          return null;  // Success
        }
    }
    
        static class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {
        @Override public TRITONSERVER_Error call (
            TRITONSERVER_ResponseAllocator allocator, Pointer buffer, Pointer buffer_userp,
            long byte_size, int memory_type, long memory_type_id)
        {
          String name = null;
          if (buffer_userp != null) {
            name = (String)Loader.accessGlobalRef(buffer_userp);
          } else {
            name = "<unknown>";
          }
          Pointer.free(buffer);
          Loader.deleteGlobalRef(buffer_userp);

          return null;  // Success
        }
    }

        static class InferRequestComplete extends TRITONSERVER_InferenceRequestReleaseFn_t {
        @Override public void call (
            TRITONSERVER_InferenceRequest request, int flags, Pointer userp)
        {
          // We reuse the request so we don't delete it here.
        }
    }

    static class InferResponseComplete extends TRITONSERVER_InferenceResponseCompleteFn_t {
        @Override public void call (
            TRITONSERVER_InferenceResponse response, int flags, Pointer userp)
        {
          if (response != null) {
            // Send 'response' to the future.
            futures.get(userp).complete(response);
          }
        }
    }

    static ConcurrentHashMap<Pointer, CompletableFuture<TRITONSERVER_InferenceResponse>> futures = new ConcurrentHashMap<>();
    static ResponseAlloc responseAlloc = new ResponseAlloc();
    static ResponseRelease responseRelease = new ResponseRelease();
    static InferRequestComplete inferRequestComplete = new InferRequestComplete();
    static InferResponseComplete inferResponseComplete = new InferResponseComplete();

    static void GenerateInputData(
        IntPointer[] input0_data, IntPointer[] input1_data) {
      input0_data[0] = new IntPointer(16);
      input1_data[0] = new IntPointer(16);
      for (int i = 0; i < 16; ++i) {
        input0_data[0].put(i, i);
        input1_data[0].put(i, 1);
      }
    }

    static void GenerateInputData(
        FloatPointer[] input0_data, FloatPointer[] input1_data)
    {
      input0_data[0] = new FloatPointer(16);
      input1_data[0] = new FloatPointer(16);
      for (int i = 0; i < 16; ++i) {
        input0_data[0].put(i, i);
        input1_data[0].put(i, 1);
      }
    }

    static void Usage(String msg) {
      if (msg != null) {
        System.err.println(msg);
      }

      System.err.println("Usage: java " + Simple.class.getSimpleName() + " [options]");
      System.err.println("\t-i Set number of iterations");
      System.err.println("\t-m <\"system\"|\"pinned\"|gpu>"
                       + " Enforce the memory type for input and output tensors."
                       + " If not specified, inputs will be in system memory and outputs"
                       + " will be based on the model's preferred type.");
      System.err.println("\t-v Enable verbose logging");
      System.err.println("\t-r [model repository absolute path]");

      System.exit(1);
    }


    public static void
    main(String[] args) throws Exception
    {
      int num_iterations = 1000000;
      String model_repository_path = null;
      int verbose_level = 0;
      boolean checkAccuracy = false;
      String input_data_path = "";

      // Parse commandline...
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "-i":
            i++;
            try {
              num_iterations = Integer.parseInt(args[i]);
            } catch (NumberFormatException e){
              Usage(
                  "-i must be used to specify number of iterations");
            }
            break;
          case "-d":
            input_data_path = args[++i];
          case "-r":
            model_repository_path = args[++i];
            break;
          case "-v":
            verbose_level = 1;
            break;
          case "-c":
            checkAccuracy = true;
            break;
          case "-?":
            Usage(null);
            break;
        }
      }

      if (model_repository_path == null) {
        Usage("-r must be used to specify model repository path");
      }
      // Check API version.
      int[] api_version_major = {0}, api_version_minor = {0};
      FAIL_IF_ERR(
          TRITONSERVER_ApiVersion(api_version_major, api_version_minor),
          "getting Triton API version");
      if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major[0]) ||
          (TRITONSERVER_API_VERSION_MINOR > api_version_minor[0])) {
        FAIL("triton server API version mismatch");
      }

      // Create the server...
      TRITONSERVER_ServerOptions server_options = new TRITONSERVER_ServerOptions(null);
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsNew(server_options),
          "creating server options");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetModelRepositoryPath(
              server_options, model_repository_path),
          "setting model repository path");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
          "setting verbose logging level");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetBackendDirectory(
              server_options, "/opt/tritonserver/backends"),
          "setting backend directory");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
              server_options, "/opt/tritonserver/repoagents"),
          "setting repository agent directory");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
          "setting strict model configuration");
      double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
              server_options, min_compute_capability),
          "setting minimum supported CUDA compute capability");

      TRITONSERVER_Server server_ptr = new TRITONSERVER_Server(null);
      FAIL_IF_ERR(
          TRITONSERVER_ServerNew(server_ptr, server_options), "creating server");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsDelete(server_options),
          "deleting server options");

      TRITONSERVER_ServerDeleter server = new TRITONSERVER_ServerDeleter(server_ptr);

      // Wait until the server is both live and ready.
      int health_iters = 0;
      while (true) {
        boolean[] live = {false}, ready = {false};
        FAIL_IF_ERR(
            TRITONSERVER_ServerIsLive(server, live),
            "unable to get server liveness");
        FAIL_IF_ERR(
            TRITONSERVER_ServerIsReady(server, ready),
            "unable to get server readiness");
        System.out.println("Server Health: live " + live[0] + ", ready " + ready[0]);
        if (live[0] && ready[0]) {
          break;
        }

        if (++health_iters >= 10) {
          FAIL("failed to find healthy inference server");
        }

        Thread.sleep(500);
      }

      // Print status of the server.
      {
        TRITONSERVER_Message server_metadata_message = new TRITONSERVER_Message(null);
        FAIL_IF_ERR(
            TRITONSERVER_ServerMetadata(server, server_metadata_message),
            "unable to get server metadata message");
        BytePointer buffer = new BytePointer((Pointer)null);
        SizeTPointer byte_size = new SizeTPointer(1);
        FAIL_IF_ERR(
            TRITONSERVER_MessageSerializeToJson(
                server_metadata_message, buffer, byte_size),
            "unable to serialize server metadata message");

        System.out.println("Server Status:");
        System.out.println(buffer.limit(byte_size.get()).getString());

        FAIL_IF_ERR(
            TRITONSERVER_MessageDelete(server_metadata_message),
            "deleting status metadata");
      }

      String model_name = "simple";

      // Wait for the model to become available.
      boolean[] is_torch_model = {false};
      boolean[] is_int = {true};
      boolean[] is_ready = {false};
      health_iters = 0;
      while (!is_ready[0]) {
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelIsReady(
                server, model_name, 1, is_ready),
            "unable to get model readiness");
        if (!is_ready[0]) {
          if (++health_iters >= 10) {
            FAIL("model failed to be ready in 10 iterations");
          }
          Thread.sleep(500);
          continue;
        }

        TRITONSERVER_Message model_metadata_message = new TRITONSERVER_Message(null);
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelMetadata(
                server, model_name, 1, model_metadata_message),
            "unable to get model metadata message");
        BytePointer buffer = new BytePointer((Pointer)null);
        SizeTPointer byte_size = new SizeTPointer(1);
        FAIL_IF_ERR(
            TRITONSERVER_MessageSerializeToJson(
                model_metadata_message, buffer, byte_size),
            "unable to serialize model status protobuf");

        JsonParser parser = new JsonParser();
        JsonObject model_metadata = null;
        try {
          model_metadata = parser.parse(buffer.limit(byte_size.get()).getString()).getAsJsonObject();
        } catch (Exception e) {
          FAIL("error: failed to parse model metadata from JSON: " + e);
        }

        FAIL_IF_ERR(
            TRITONSERVER_MessageDelete(model_metadata_message),
            "deleting status protobuf");

        if (!model_metadata.get("name").getAsString().equals(model_name)) {
          FAIL("unable to find metadata for model");
        }

        boolean found_version = false;
        if (model_metadata.has("versions")) {
          for (JsonElement version : model_metadata.get("versions").getAsJsonArray()) {
            if (version.getAsString().equals("1")) {
              found_version = true;
              break;
            }
          }
        }
        if (!found_version) {
          FAIL("unable to find version 1 status for model");
        }

        FAIL_IF_ERR(
            ParseModelMetadata(model_metadata, is_int, is_torch_model),
            "parsing model metadata");
      }


      for(int i = 0; i < num_iterations; i++){
        try (PointerScope scope = new PointerScope()) {
          RunInference(server, model_name, is_int, is_torch_model, checkAccuracy);
        }
      }
      done = true;
      memory_thread.join();

      System.exit(0);
    }
}
