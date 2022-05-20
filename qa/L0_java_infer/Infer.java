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

      Runnable runnable =
        () -> { 
          if(ValidateMemoryGrowth(.05f)){
            System.out.println("Memory growth test passed");
          } else {
            System.out.println("Memory growth test FAILED");
          }
        };
      Thread memory_thread = new Thread(runnable);
      memory_thread.start();

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
