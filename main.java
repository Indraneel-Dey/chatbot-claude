import com.amazonaws.services.bedrock.AWSBedrock;
import com.amazonaws.services.bedrock.AWSBedrockClientBuilder;
import com.amazonaws.services.bedrock.model.InvokeModelRequest;
import com.amazonaws.services.bedrock.model.InvokeModelResult;
import com.amazonaws.services.bedrock.model.ModelInput;
import com.amazonaws.services.bedrock.model.ModelOutput;
import com.google.gson.Gson;
import spark.Request;
import spark.Response;
import spark.Route;

import java.util.HashMap;
import java.util.Map;

import static spark.Spark.*;

public class App {
    private static final String MODEL_ID_CLAUDE = "anthropic.claude-3-haiku-20240307-v1:0";
    private static final String MODEL_ID_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0";
    private static final String REGION = "us-east-1";
    private static final int MAX_ATTEMPTS = 10;

    private static final AWSBedrock bedrockClient = AWSBedrockClientBuilder.standard()
            .withRegion(REGION)
            .withClientConfiguration(clientConfiguration -> clientConfiguration.setMaxErrorRetry(MAX_ATTEMPTS))
            .build();

    private static final Gson gson = new Gson();

    public static void main(String[] args) {
        port(8000);
        get("/", (req, res) -> "Hello, World!");

        post("/query", handleQuery);

        awaitInitialization();
    }

    private static Route handleQuery = (Request req, Response res) -> {
        String question = req.queryParams("question");
        String modelType = req.queryParams("model_type");

        if (modelType == null || modelType.isEmpty()) {
            modelType = "sonnet";
        }

        String modelId;
        switch (modelType) {
            case "sonnet":
                modelId = MODEL_ID_SONNET;
                break;
            case "haiku":
                modelId = MODEL_ID_CLAUDE;
                break;
            default:
                return gson.toJson(Map.of("error", "Invalid model type."));
        }

        Map<String, String> modelInput = new HashMap<>();
        modelInput.put("query", question);

        InvokeModelRequest invokeModelRequest = new InvokeModelRequest()
                .withModelId(modelId)
                .withInput(new ModelInput().withJson(gson.toJson(modelInput)));

        InvokeModelResult invokeModelResult = bedrockClient.invokeModel(invokeModelRequest);
        ModelOutput modelOutput = invokeModelResult.getModelOutput();

        return gson.toJson(Map.of("result", modelOutput.getJson()));
    };
}