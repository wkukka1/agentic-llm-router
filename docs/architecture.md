# Class hierarchies (target architecture)

This is a **target design**, not as-built documentation. It's a blueprint for
the production-grade agentic router service this project is working toward —
drawn in generic typed pseudocode (`String`, `float`, `List~X~`) rather than
literal Python, and naming service-shaped classes (`RoutingPolicy`, `Agent`,
`TrainingHarness`, ...) that mostly don't exist in `src/` yet. Use it to judge
whether a piece of work moves the codebase toward this shape, not to look up
a real class, method signature, or file — for that, see "Where today's code
stands" below and the docs it points to.

The diagram has two layers plus the one type that crosses between them:

- **Serving** (solid boxes, would live under `src/router/`) — the request
  path from a raw prompt to a routed and possibly executed answer: prompt
  decomposition into `Signal`s, candidate scoring by a `RouterModel`, a
  `RoutingPolicy` decision, and an `Agent`/`Orchestrator` layer that can plan,
  execute, spawn child agents, and enforce a shared cost `BudgetLedger`.
- **Offline** (dashed boxes, would live under `src/training/`) — fitting a
  `RouterModelArtifact` (`RouterModelTrainer` + `TrainingHarness`, producing a
  `TrainingRun` with its own `TrainingMetrics`/`ValidationMetrics`).
  `src/evaluation/` isn't modeled in this diagram — scoring a fitted artifact
  against ground truth is a separate concern from fitting one, and today's
  evaluation code (see "Where today's code stands" below) doesn't need this
  redesign the way training and serving do.
- **`RouterModelArtifact`** is the only type both sides touch: offline code
  produces it, `ArtifactStore` persists it, `RouterModelFactory` turns it back
  into a live `RouterModel` for serving. The target keeps the one hard
  boundary the current codebase already enforces: **serving never imports
  training or evaluation.** `training` may import `router`; `evaluation` may
  import both.

```mermaid
classDiagram


    %% =========================================================
    %% CONVENTIONS
    %%   -->  association: the source holds a field of the target type
    %%   ..>  dependency: parameter, return value, or id reference only
    %%   Dashed boxes are OFFLINE (src/training).
    %%   Solid boxes are SERVING (src/router). No solid box points
    %%   at a dashed box. RouterModelArtifact is the only type both
    %%   sides share.
    %% =========================================================


    %% =========================================================
    %% SERVING :: TOP-LEVEL ROUTER            (router/router.py)
    %%   Router owns the pipeline: decompose -> filter -> rank -> decide
    %% =========================================================

    class Router {
        +String name
        +String version
        +PromptDecomposer promptDecomposer
        +RouterModel routerModel
        +RoutingPolicy routingPolicy
        +LLMRegistry llmRegistry
        +AgentFactory agentFactory
        +route(RoutingRequest request) RoutingDecision
        +buildContext(RoutingRequest request) RoutingContext
    }

    Router --> PromptDecomposer : uses
    Router --> RouterModel : ranks with
    Router --> RoutingPolicy : decides with
    Router --> LLMRegistry : accesses
    Router --> AgentFactory : creates agents via
    Router ..> RoutingContext : builds


    %% =========================================================
    %% SERVING :: REQUEST / CONTEXT / OBJECTIVE
    %%   (router/context.py, router/constraints.py)
    %%   RoutingRequest is raw caller input.
    %%   RoutingContext is the enriched object built by Router.
    %% =========================================================

    class RoutingRequest {
        +String requestId
        +String prompt
        +Conversation conversation
        +RoutingConstraints constraints
        +RoutingMode mode
        +String parentTaskId
        +int depth
    }

    class RoutingContext {
        +RoutingRequest request
        +PromptSignals signals
        +List~LLMProfile~ candidates
        +Map~String,Object~ metadata
    }

    class RoutingMode {
        <<enumeration>>
        NORMAL
        MODEL_SELECTION
    }

    class RoutingConstraints {
        +float maxCost
        +float maxLatency
        +float minQuality
        +List~String~ requiredCapabilities
        +OptimizationObjective objective
    }

    class OptimizationObjective {
        +float qualityWeight
        +float costWeight
        +float latencyWeight
        +float riskWeight
        +utility(float quality, float cost, float latency, float risk) float
        +utility(ModelScore score) float
        +utility(PlanScore score) float
    }

    note for OptimizationObjective "The only definition of utility. lambda = costWeight. Offline code imports this class and never reimplements it."

    RoutingRequest --> RoutingConstraints
    RoutingRequest --> RoutingMode
    RoutingRequest --> Conversation
    RoutingConstraints --> OptimizationObjective
    RoutingContext --> RoutingRequest
    RoutingContext --> PromptSignals
    RoutingContext --> "*" LLMProfile : candidates
    OptimizationObjective ..> ModelScore : scores
    OptimizationObjective ..> PlanScore : scores


    %% =========================================================
    %% SERVING :: PROMPT DECOMPOSER / CLASSIFIERS
    %%   (router/decompose/)
    %% =========================================================

    class PromptDecomposer {
        +String name
        +String version
        +List~Classifier~ classifiers
        +List~Embedder~ embedders
        +extractSignals(ClassificationInput input) PromptSignals
        +addClassifier(Classifier classifier)
        +removeClassifier(String name)
    }

    class ClassificationInput {
        +String prompt
        +Conversation conversation
        +Map~String,Object~ metadata
    }

    class Classifier {
        <<interface>>
        +String name
        +String version
        +classify(ClassificationInput input) List~Signal~
    }

    class SimpleClassifier {
        +classify(ClassificationInput input) List~Signal~
    }

    class CompositeClassifier {
        +List~Classifier~ subClassifiers
        +classify(ClassificationInput input) List~Signal~
    }

    Classifier <|.. SimpleClassifier
    Classifier <|.. CompositeClassifier
    PromptDecomposer o-- "*" Classifier : contains
    PromptDecomposer o-- "0..*" Embedder : optionally uses
    CompositeClassifier o-- "*" Classifier : contains
    ClassificationInput --> Conversation
    PromptDecomposer ..> ClassificationInput : consumes
    PromptDecomposer ..> PromptSignals : produces
    Classifier ..> Signal : produces


    %% =========================================================
    %% SERVING :: SIGNALS                     (decompose/signals.py)
    %% =========================================================

    class Signal {
        +String name
        +Object value
        +float confidence
        +String producedBy
        +String producerVersion
        +Map~String,Object~ metadata
    }

    class PromptSignals {
        +Map~String,Signal~ signals
        +Map~String,float[]~ embeddings
        +add(Signal signal)
        +addEmbedding(String name, float[] vector)
        +get(String name) Signal
        +getEmbedding(String name) float[]
    }

    PromptSignals "1" o-- "*" Signal


    %% =========================================================
    %% SERVING :: EMBEDDING                   (decompose/embedders/)
    %%   Embedder returns vectors; the decomposer assembles them.
    %% =========================================================

    class Embedder {
        <<interface>>
        +String name
        +String version
        +int dimension
        +embed(String text) float[]
    }

    class SentenceEmbedder {
        +embed(String text) float[]
    }

    class PromptEmbedder {
        +embed(String text) float[]
    }

    Embedder <|.. SentenceEmbedder
    Embedder <|.. PromptEmbedder


    %% =========================================================
    %% SERVING :: ROUTER MODEL                (router/models/)
    %%   Ranks only. Control flow is the policy's job, not the model's.
    %%   Inference only. No fit(). Fitted state lives in the artifact.
    %%   Concrete variants (KNN, IRT, NIRT, ML) are configuration, not
    %%   design-level types.
    %%   filterCandidates is concrete and inherited, so constraint and
    %%   support checking has one implementation across all variants.
    %% =========================================================

    class RouterModel {
        <<abstract>>
        +String name
        +String version
        +RouterModelArtifact artifact
        +CostModel costModel
        +UnsupportedCandidatePolicy unsupportedPolicy
        +filterCandidates(List~LLMProfile~ candidates, RoutingConstraints constraints) List~LLMProfile~
        +supportedModels() List~String~
        +rank(RoutingContext context, List~LLMProfile~ candidates) List~ModelScore~*
    }

    class UnsupportedCandidatePolicy {
        <<enumeration>>
        ERROR
        DROP
        SCORE_WITH_PRIOR
    }

    note for RouterModel "Concrete ranking models are implementations of this class, registered with RouterModelFactory by routerModelName. They are not modeled here."

    RouterModel --> RouterModelArtifact : loaded from
    RouterModel --> CostModel : predicts cost with
    RouterModel --> UnsupportedCandidatePolicy : applies
    RouterModel ..> RoutingConstraints : enforces
    RouterModel ..> RoutingContext : consumes
    RouterModel ..> ModelScore : produces


    %% =========================================================
    %% SERVING :: PREDICTED COST              (router/llm/cost.py)
    %%   Predicted cost only. Observed cost lives in the dataset
    %%   and in LLMResponse, never here.
    %% =========================================================

    class CostModel {
        +String version
        +Map~String,float~ outputTokenPriors
        +predictOutputTokens(LLMProfile model, int inputTokens) int
        +predictCost(LLMProfile model, int inputTokens, int outputTokens) float
        +predictCost(LLMProfile model, RoutingContext context) float
    }

    CostModel ..> LLMProfile : reads prices


    %% =========================================================
    %% SERVING :: ROUTER MODEL ARTIFACT       (models/artifacts.py)
    %%   The one type shared by serving and offline code.
    %%   Fit in the pipeline, load in serving. Identity is explicit
    %%   so a trace resolves to an exact checkpoint.
    %% =========================================================

    class RouterModelArtifact {
        +String artifactId
        +String contentHash
        +ArtifactFormat format
        +String schemaVersion
        +List~String~ supportedModelIds
        +String routerModelName
        +String routerModelVersion
        +String trainingDatasetVersion
        +String trainingRunId
        +String createdAt
        +Map~String,Object~ hyperparameters
        +ArtifactPayload payload
    }

    class ArtifactPayload {
        <<interface>>
    }

    note for ArtifactPayload "Opaque to serving. A trainer and a model registered under the same routerModelName agree on the payload shape. RouterModelFactory rejects a mismatch."

    class ArtifactFormat {
        <<enumeration>>
        NPZ
        SAFETENSORS
        JSON
        PICKLE
    }

    RouterModelArtifact --> ArtifactFormat
    RouterModelArtifact --> ArtifactPayload


    %% =========================================================
    %% SERVING :: ARTIFACT STORAGE            (models/artifacts.py)
    %%   The store owns file locations; the artifact does not.
    %% =========================================================

    class ArtifactStore {
        <<interface>>
        +save(RouterModelArtifact artifact) String
        +load(String artifactId) RouterModelArtifact
        +exists(String artifactId) boolean
    }

    class LocalArtifactStore {
        +String rootPath
        +save(RouterModelArtifact artifact) String
        +load(String artifactId) RouterModelArtifact
        +exists(String artifactId) boolean
    }

    ArtifactStore <|.. LocalArtifactStore
    ArtifactStore ..> RouterModelArtifact : stores


    %% =========================================================
    %% SERVING :: ROUTER MODEL FACTORY        (models/registry.py)
    %% =========================================================

    class RouterModelFactory {
        +register(String routerModelName, Object modelClass)
        +create(RouterModelArtifact artifact, CostModel costModel) RouterModel
        +create(String artifactId, ArtifactStore store, CostModel costModel) RouterModel
    }

    RouterModelFactory ..> RouterModelArtifact : reads
    RouterModelFactory ..> RouterModel : creates
    RouterModelFactory ..> ArtifactStore : loads from
    RouterModelFactory ..> CostModel : injects


    %% =========================================================
    %% SERVING :: ROUTING POLICY              (router/policy.py)
    %%   Turns scores into a decision and a destination.
    %% =========================================================

    class RoutingPolicy {
        +String name
        +String version
        +float escalationThreshold
        +decide(RoutingContext context, List~ModelScore~ scores) RoutingDecision
        +selectDestination(RoutingContext context, List~ModelScore~ scores) DecisionDestination
    }

    RoutingPolicy ..> RoutingContext : consumes
    RoutingPolicy ..> ModelScore : consumes
    RoutingPolicy ..> RoutingDecision : produces
    RoutingPolicy ..> DecisionDestination : selects
    RoutingPolicy ..> OptimizationObjective : applies


    %% =========================================================
    %% SERVING :: ROUTING DECISION            (router/decision.py)
    %% =========================================================

    class RoutingDecision {
        +String requestId
        +List~ModelScore~ rankedModels
        +String selectedModelId
        +DecisionDestination destination
        +String reason
        +float confidence
        +String artifactId
        +String policyVersion
    }

    class ModelScore {
        +LLMProfile model
        +float score
        +float expectedQuality
        +float expectedCost
        +float expectedLatency
        +float confidence
        +boolean supported
    }

    class DecisionDestination {
        <<enumeration>>
        USER
        PARENT_ORCHESTRATOR
        NEW_ORCHESTRATOR
    }

    RoutingDecision "1" *-- "*" ModelScore : ranked models
    RoutingDecision --> DecisionDestination
    RoutingDecision ..> RouterModelArtifact : produced by
    ModelScore --> LLMProfile


    %% =========================================================
    %% SERVING :: LLM INFRASTRUCTURE          (router/llm/)
    %%   LLMProfile is static registry config only.
    %%   Learned ability estimates live inside the loaded artifact.
    %% =========================================================

    class LLMProfile {
        +String name
        +String provider
        +String modelId
        +String version
        +float inputCostPerToken
        +float outputCostPerToken
        +int contextWindow
        +List~String~ capabilities
    }

    class LLMRegistry {
        +List~LLMProfile~ profiles
        +register(LLMProfile profile)
        +get(String modelId) LLMProfile
        +list() List~LLMProfile~
    }

    class LLMClientFactory {
        +Map~String,ProviderAdapter~ adapters
        +clientFor(LLMProfile profile) LLMClient
    }

    class LLMClient {
        +LLMProfile profile
        +ProviderAdapter adapter
        +complete(List~Message~ messages) LLMResponse
        +stream(List~Message~ messages) Iterator~String~
    }

    class LLMResponse {
        +String content
        +int inputTokens
        +int outputTokens
        +float cost
        +float latency
    }

    class ProviderAdapter {
        <<interface>>
        +String provider
        +complete(String modelId, List~Message~ messages) LLMResponse
        +stream(String modelId, List~Message~ messages) Iterator~String~
    }

    class OpenAIAdapter
    class AnthropicAdapter
    class GoogleAdapter

    ProviderAdapter <|.. OpenAIAdapter
    ProviderAdapter <|.. AnthropicAdapter
    ProviderAdapter <|.. GoogleAdapter

    LLMRegistry "1" *-- "*" LLMProfile
    LLMClientFactory o-- "*" ProviderAdapter : keyed by provider
    LLMClientFactory ..> LLMProfile : resolves
    LLMClientFactory ..> LLMClient : produces
    LLMClient --> LLMProfile
    LLMClient --> ProviderAdapter : delegates to
    LLMClient ..> LLMResponse : returns
    LLMClient ..> Message : sends


    %% =========================================================
    %% SERVING :: CONVERSATION                (router/context.py)
    %% =========================================================

    class Conversation {
        +String conversationId
        +List~Message~ messages
    }

    class Message {
        +MessageRole role
        +String content
        +int tokens
    }

    class MessageRole {
        <<enumeration>>
        SYSTEM
        USER
        ASSISTANT
        TOOL
    }

    Conversation "1" *-- "*" Message
    Message --> MessageRole


    %% =========================================================
    %% SERVING :: AGENTS / ORCHESTRATION      (router/execution/)
    %%   AgentFactory creates Agents. Orchestrator is the Agent
    %%   that plans, forecasts, executes, and spawns child agents.
    %%   Step-level types (LLMStep, ToolStep, SpawnStep, StepAttempt,
    %%   Reservation, AdmissionController) belong in
    %%   router-execution-types.mermaid, not here.
    %% =========================================================

    class Agent {
        <<abstract>>
        +String name
        +String version
        +LLMProfile llm
        +LLMClientFactory clientFactory
        +List~Tool~ tools
        +BudgetLedger ledger
        +ExecutionLimits limits
        +run(AgentTask task) ExecutionResult*
    }

    note for Agent "Concrete agents are selected by AgentConfig.implementation and are not modeled here. Every agent in a task tree shares one BudgetLedger."

    class Orchestrator {
        +AgentFactory factory
        +RouterTool routerTool
        +PlanForecaster planForecaster
        +run(AgentTask task) ExecutionResult
        +plan(AgentTask task) Plan
        +execute(Plan plan) ExecutionResult
        +replan(Plan plan) Plan
        +spawn(AgentTask child) Agent
        +canSpawn(AgentTask child) boolean
    }

    class AgentFactory {
        +createRoot(AgentConfig config) Agent
        +createChild(AgentConfig config, BudgetLedger sharedLedger) Agent
    }

    class AgentConfig {
        +String name
        +String version
        +String implementation
        +LLMProfile llm
        +ExecutionLimits limits
        +List~Tool~ tools
    }

    Agent <|-- Orchestrator

    AgentFactory ..> AgentConfig : configured by
    AgentFactory ..> Agent : creates
    AgentFactory ..> BudgetLedger : creates root ledger
    AgentConfig --> LLMProfile
    AgentConfig --> ExecutionLimits
    AgentConfig o-- "*" Tool

    Agent --> LLMProfile
    Agent --> LLMClientFactory : calls models via
    Agent o-- "*" Tool
    Agent --> BudgetLedger : shares
    Agent --> ExecutionLimits : enforces
    Agent ..> AgentTask : runs
    Agent ..> ExecutionResult : returns

    Orchestrator --> AgentFactory : spawns children via
    Orchestrator --> RouterTool
    Orchestrator --> PlanForecaster : uses
    Orchestrator ..> Plan : creates
    Orchestrator ..> AgentTask : creates
    Orchestrator ..> Agent : spawns (depth bounded)


    %% =========================================================
    %% SERVING :: EXECUTION LIMITS / BUDGET   (execution/budget.py)
    %%   Single owner for money and depth. Agents and tasks
    %%   hold references, not their own copies.
    %% =========================================================

    class ExecutionLimits {
        +float maxCost
        +float maxLatency
        +int maxDepth
        +int maxFanout
        +int maxReplans
    }

    class BudgetLedger {
        +String rootTaskId
        +float total
        +float reserved
        +float spent
        +remaining() float
        +reserve(String taskId, float amount) boolean
        +commit(String taskId, float actual)
        +release(String taskId)
    }

    BudgetLedger ..> ExecutionLimits : bounded by


    %% =========================================================
    %% SERVING :: ROUTER AS A TOOL            (router/tools/)
    %%   MODEL_SELECTION mode is what prevents unbounded re-entry.
    %% =========================================================

    class Tool {
        <<interface>>
        +String name
        +String version
        +execute(Object input) Object
    }

    class MCPTool {
        +String server
        +execute(Object input) Object
    }

    class RouterTool {
        +String name
        +String version
        +Router router
        +execute(Object input) RoutingDecision
        +routeForModelSelection(RoutingRequest request) RoutingDecision
    }

    Tool <|.. MCPTool
    Tool <|.. RouterTool
    RouterTool --> Router : invokes in MODEL_SELECTION


    %% =========================================================
    %% SERVING :: PLANNING                    (execution/plan.py, forecast.py)
    %% =========================================================

    class Plan {
        +String planId
        +String taskId
        +List~PlanStep~ steps
        +float estimatedCost
        +float estimatedQuality
        +float estimatedLatency
        +float utility
        +PlanStatus status
        +int revision
    }

    class PlanStep {
        +String stepId
        +String prompt
        +LLMProfile model
        +int estimatedInputTokens
        +int estimatedOutputTokens
        +float estimatedCost
        +List~String~ dependencies
    }

    class PlanStatus {
        <<enumeration>>
        DRAFT
        EVALUATED
        APPROVED
        EXECUTING
        COMPLETED
        FAILED
    }

    class PlanForecaster {
        +CostModel costModel
        +forecastCost(Plan plan) CostForecast
        +forecastQuality(Plan plan) float
        +forecastLatency(Plan plan) float
        +scorePlan(Plan plan, OptimizationObjective objective) PlanScore
    }

    class CostForecast {
        +float inputCost
        +float outputCost
        +float totalCost
        +float confidence
    }

    class PlanScore {
        +float quality
        +float cost
        +float latency
        +float utility
        +boolean withinBudget
    }

    Plan "1" *-- "*" PlanStep
    Plan --> PlanStatus
    PlanStep --> LLMProfile
    PlanForecaster --> CostModel : predicts cost with
    PlanForecaster ..> Plan : forecasts
    PlanForecaster ..> CostForecast : produces
    PlanForecaster ..> PlanScore : produces
    PlanForecaster ..> OptimizationObjective : applies


    %% =========================================================
    %% SERVING :: AGENT TASK TREE             (execution/)
    %%   Depth lives on the task; the bound lives in ExecutionLimits.
    %% =========================================================

    class AgentTask {
        +String taskId
        +String parentTaskId
        +String rootTaskId
        +String prompt
        +int depth
        +float allocatedBudget
        +TaskStatus status
        +Plan plan
        +ExecutionResult result
        +List~AgentTask~ children
        +aggregateChildren() ExecutionResult
    }

    class TaskStatus {
        <<enumeration>>
        PENDING
        PLANNING
        RUNNING
        COMPLETED
        FAILED
        TERMINATED
    }

    AgentTask --> Plan
    AgentTask --> ExecutionResult
    AgentTask --> TaskStatus
    AgentTask "1" o-- "*" AgentTask : child tasks


    %% =========================================================
    %% SERVING :: EXECUTION RESULT            (execution/results.py)
    %%   STUB - modelled properly in the next pass. Executor, step
    %%   dispatch, retry and fallback are deliberately left out here.
    %% =========================================================

    class ExecutionResult {
        +String output
        +float actualCost
        +float actualLatency
        +int inputTokens
        +int outputTokens
        +boolean success
        +String errorCode
    }


    %% =========================================================
    %% SERVING :: OBSERVABILITY               (router/tracing/)
    %%   A trace resolves to an exact artifact, policy and dataset.
    %% =========================================================

    class ExecutionTrace {
        +String traceId
        +String requestId
        +String taskId
        +String routerVersion
        +String artifactId
        +String policyVersion
        +RoutingRequest request
        +PromptSignals signals
        +RoutingDecision decision
        +ExecutionResult result
        +String timestamp
    }

    ExecutionTrace --> RoutingRequest
    ExecutionTrace --> PromptSignals
    ExecutionTrace --> RoutingDecision
    ExecutionTrace --> ExecutionResult
    ExecutionTrace ..> RouterModelArtifact : produced by


    %% =========================================================
    %% OFFLINE :: TRAINING DATA               (training/datasets/)
    %% =========================================================

    class DatasetRef {
        +String datasetId
        +String version
        +String path
        +String contentHash
        +String description
    }

    class DatasetSplit {
        <<enumeration>>
        TRAIN
        VALIDATION
        TEST
    }

    class TrainingConfig {
        +String runName
        +DatasetSplit trainingSplit
        +DatasetSplit validationSplit
        +int seed
        +int maxEpochs
        +int earlyStoppingPatience
        +Map~String,Object~ hyperparameters
        +Map~String,Object~ preprocessing
    }

    TrainingConfig --> DatasetSplit


    %% =========================================================
    %% OFFLINE :: TRAINERS                    (training/trainers/)
    %% =========================================================

    class RouterModelTrainer {
        <<abstract>>
        +String name
        +String version
        +train(DatasetRef dataset, TrainingConfig config) RouterModelArtifact*
    }

    note for RouterModelTrainer "One implementation per router model, registered under the same routerModelName. Not modeled here."

    RouterModelTrainer ..> DatasetRef : trains on
    RouterModelTrainer ..> TrainingConfig : configured by
    RouterModelTrainer ..> RouterModelArtifact : produces


    %% =========================================================
    %% OFFLINE :: TRAINING HARNESS            (training/pipeline.py)
    %% =========================================================

    class TrainingHarness {
        +run(TrainingJob job) TrainingRun
        +train(RouterModelTrainer trainer, DatasetRef dataset, TrainingConfig config) TrainingRun
    }

    class TrainingJob {
        +String runId
        +RouterModelTrainer trainer
        +DatasetRef dataset
        +TrainingConfig config
        +String outputPath
    }

    class TrainingRun {
        +String runId
        +String trainerName
        +String trainerVersion
        +String datasetVersion
        +RouterModelArtifact artifact
        +TrainingMetrics trainingMetrics
        +ValidationMetrics validationMetrics
        +TrainingConfig config
        +String createdAt
    }

    class TrainingMetrics {
        +float loss
        +float accuracy
        +float bce
        +float auc
        +float spearman
        +float regret
        +float trainingTime
    }

    class ValidationMetrics {
        +float loss
        +float accuracy
        +float bce
        +float auc
        +float spearman
        +float regret
        +Map~String,Object~ selectedParameters
    }

    TrainingHarness ..> TrainingJob : executes
    TrainingHarness ..> TrainingRun : produces
    TrainingHarness ..> ArtifactStore : saves to

    TrainingJob --> RouterModelTrainer
    TrainingJob --> DatasetRef
    TrainingJob --> TrainingConfig

    TrainingRun --> RouterModelArtifact
    TrainingRun --> TrainingMetrics
    TrainingRun --> ValidationMetrics
    TrainingRun --> TrainingConfig


    %% =========================================================
    %% STYLING: offline boxes are dashed
    %% =========================================================

    classDef offline stroke:#888,stroke-width:1px,stroke-dasharray:6 4

    cssClass "DatasetRef,DatasetSplit,TrainingConfig,RouterModelTrainer,TrainingHarness,TrainingJob,TrainingRun,TrainingMetrics,ValidationMetrics" offline
```

## Notes

- **Predicted vs. observed cost, kept as two types.** `CostModel` is the only
  thing serving code consults, and it only ever predicts — `RouterModel`,
  `PlanForecaster`, and `OptimizationObjective` all take a predicted cost as
  input. Observed cost (`LLMResponse.cost`, dataset ground truth) never feeds
  a routing decision; it only shows up offline, in `EvaluationHarness`
  metrics and in training data. Nothing in the target design should blur
  that line by having a `RouterModel` read observed cost.
- **One `utility()`, everywhere.** `OptimizationObjective` is meant to be the
  single definition of utility (`lambda` = `costWeight`); `RoutingPolicy`,
  `PlanForecaster`, and offline evaluation all call the same object rather
  than each re-deriving a quality/cost/latency tradeoff.
- **`Router` bundles more than a decision layer.** Unlike a bare scorer, the
  target `Router` composes prompt decomposition, ranking, policy, and agent
  creation into one entry point — it's the thing a caller talks to, not an
  internal scoring utility.
- **The agent/orchestration layer is new surface area.** `Agent`,
  `Orchestrator`, `AgentTask`, `Plan`/`PlanStep`/`PlanForecaster`,
  `ExecutionLimits`, and `BudgetLedger` describe a full plan → execute →
  replan loop with a shared cost ledger across a spawned task tree. This is
  the most speculative part of the diagram; step-level detail
  (`LLMStep`/`ToolStep`/`SpawnStep`/`StepAttempt`/`Reservation`/
  `AdmissionController`) is deliberately left for a separate diagram rather
  than inflating this one.
- **`TrainingHarness` formalizes what's today an ad hoc script.**
  `RouterModelTrainer` + `TrainingHarness` give every trainer the same
  shape (`train(dataset, config) -> RouterModelArtifact`, wrapped into a
  `TrainingRun`), so a new router kind doesn't need a new pipeline script.
  Evaluation gets no equivalent treatment here — it's deliberately left out
  of this pass.
- **Ranking vs. control flow is a hard split.** `RouterModel.rank()` only
  scores candidates; `filterCandidates` is concrete and inherited so every
  variant filters by `RoutingConstraints` the same way; deciding what to *do*
  with the scores (pick a model, escalate, route to an orchestrator) is
  `RoutingPolicy`'s job, not the model's. A new router kind should never need
  to touch policy code, and a new policy should never need to know how a
  score was produced.
- **A few interfaces get one concrete example, most don't.** `Classifier`,
  `Embedder`, `ArtifactStore`, `ProviderAdapter`, and `Tool` each show one or
  two example implementations (`SimpleClassifier`; `SentenceEmbedder`/
  `PromptEmbedder`; `LocalArtifactStore`; `OpenAIAdapter`/`AnthropicAdapter`/
  `GoogleAdapter`; `MCPTool`) just to make the interface's shape concrete.
  `RouterModel`, `RouterModelTrainer`, and `Agent` deliberately get none —
  their `note for` callouts say so — because the whole point of those three
  is that concrete variants are registry entries / config, not part of the
  class hierarchy worth drawing.
- **`MODEL_SELECTION` is the re-entry guard.** `RouterTool` lets an
  `Orchestrator` call back into `Router` as a tool; `RoutingMode` is what
  stops that from recursing into another full agentic run — a
  `MODEL_SELECTION`-mode request can only ever get a ranked model back, never
  spawn another `Orchestrator`.

## Where today's code stands relative to this

None of the classes above exist yet as written; the current, as-built
equivalents are documented elsewhere and look quite different in shape:

- **Routing decision layer** — `router.routing.RouterModel` (ABC, `kind` +
  `predict_scores` + `route`) and its `matrix`/`nirt`/`knn`/`mlp`/`random`
  implementations. `router.router.Router` composes it with `PromptDecomposer`,
  `RoutingPolicy` and `LLMRegistry` (still no `AgentFactory` wiring — nothing
  constructs it in production yet). See
  [`docs/routing_interface.md`](routing_interface.md).
- **Agentic layer** — `router.agentic.AgenticRouter` wraps a `RouterModel`
  plus `Triage` (`HeuristicTriage`/`LLMTriage`), a `ClientRegistry` of
  `router.llm.client.LLMClient`s (adapter-based, `complete`/`stream` — the
  `invoke`-based duplicate this bullet used to describe has been merged into
  this one), and one of two orchestrators (`RecursiveOrchestrator`,
  `LangChainToolOrchestrator`) that share no common base. It's a flat
  triage → single-or-decompose → synthesize flow: no `AgentTask` tree, no
  `Plan`/`PlanForecaster`, no `BudgetLedger`, no execution limits. See
  [`docs/agentic_router.md`](agentic_router.md).
- **Model definitions and response heads** — `NIRTModel`/`IRTRouterModel`
  (`nn.Module`s chosen by `build_model(cfg)` on an orientation string) and the
  `bernoulli`/`normal`/`beta`/`zoib` `ResponseHead` subclasses (only
  `BaselineNIRT`, the control arm, composes one via `build_response_head`).
  These sit underneath the target's abstract `RouterModel`/`RouterModelArtifact`
  but there's no artifact format, factory, or store around them today — a
  run is loaded straight off a checkpoint path. See
  [`docs/nirt_model.md`](nirt_model.md) and
  [`docs/baseline_control_arm.md`](baseline_control_arm.md).
- **Training** — fitting is `fit`/`fit_mlp_router`, a function per trainer,
  not the `RouterModelTrainer`/`TrainingHarness` classes above. See
  [`docs/nirt_model.md`](nirt_model.md).
- **Evaluation** — scoring against ground truth is
  `evaluate_split`/`routing_evaluation`/oracle labels in
  `evaluation.routing.oracle`, run through scripts
  (`scripts/nirt/route_eval.py`, `route_compare`). This diagram no longer
  models a target shape for it at all — see
  [`docs/routing_evaluation.md`](routing_evaluation.md) and
  [`docs/pool_expansion.md`](pool_expansion.md) for what exists today.
- **Not class hierarchies today, and not modeled as such in the target
  either:** `TextEncoder` branches on a `backend` string field internally
  rather than subclassing; `TrainingData` is a single facade object; the
  agentic `Triage`/`Decomposer`/`Synthesizer` are `Callable` type aliases
  with duck-typed implementations and no shared base.
